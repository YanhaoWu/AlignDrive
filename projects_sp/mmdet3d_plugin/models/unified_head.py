from typing import List, Optional, Union


import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from mmdet.models import HEADS
from mmdet.models import build_head
from mmcv.utils import build_from_cfg

import random
from einops.einops import rearrange
from mmcv.cnn.bricks.registry import ATTENTION, PLUGIN_LAYERS, NORM_LAYERS
from projects_sp.mmdet3d_plugin.models.attention import get_1d_sincos_pos_embed_from_grid
from projects_sp.mmdet3d_plugin.models.instance_bank import Img_instance_queue
from .instance_bank import topk
from projects_sp.mmdet3d_plugin.core.box3d import X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX
from projects_sp.mmdet3d_plugin.models.utils import augment_perception_with_fake_bbox
from projects_sp.mmdet3d_plugin.models.longitudinal_planning.score_module.compute_b2d_score_wyh import (
    compute_corners_torch,
)
from projects_sp.mmdet3d_plugin.models.attention import gen_sineembed_for_position


@HEADS.register_module()
class UnifiedHead(BaseModule):
    """Coordinate perception, map, lateral-path, and longitudinal-planning heads."""

    def __init__(
        self,
        task_config: dict,
        det_head=dict,
        map_head=dict,
        motion_plan_head=dict,
        longitudinal_head=dict,
        dual_plan_head=dict,
        fusion_model=dict,
        temporal_queue=None,
        init_cfg=None,
        num_decoder=6,
        query_drop_rate=0.3,
        **kwargs,
    ):
        super(UnifiedHead, self).__init__(init_cfg)
        self.task_config = task_config

        if self.task_config['with_det']:
            self.det_head = build_head(det_head)
        if self.task_config['with_map']:
            self.map_head = build_head(map_head)
        if self.task_config['with_motion_plan']:
            self.motion_plan_head = build_head(motion_plan_head)

        self.fusion_model = build_head(fusion_model)

        if longitudinal_head is not None:
            self.longitudinal_head = build_head(longitudinal_head)
        else:
            self.longitudinal_head = None

        if dual_plan_head is not None:
            self.dual_plan_head = build_head(dual_plan_head)
        else:
            self.dual_plan_head = None

        self.query_drop_rate = query_drop_rate

        self.temporal_img = temporal_queue['temporal_img']
        self.temporal_gap = temporal_queue['temporal_gap']
        if self.temporal_img > 0:
            self.temporal_queue = Img_instance_queue(
                queue_length=self.temporal_img,
                recurrent_temporal=False,
                temporal_gap=self.temporal_gap,
            )
        else:
            self.temporal_queue = None

        self.num_decoder = num_decoder
        self.num_key_object = kwargs.get('num_key_object', 15)
        self.num_key_map = kwargs.get('num_key_object', 10)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Accept legacy longitudinal-head checkpoint prefixes without duplicating modules."""
        legacy_prefix = prefix + 'odo_head.'
        current_prefix = prefix + 'longitudinal_head.'
        renamed_keys = {
            key: current_prefix + key[len(legacy_prefix):]
            for key in state_dict
            if key.startswith(legacy_prefix)
        }
        for legacy_key, current_key in renamed_keys.items():
            if current_key in state_dict:
                raise RuntimeError(f'Checkpoint contains both {legacy_key} and {current_key}')
        for legacy_key, current_key in renamed_keys.items():
            state_dict[current_key] = state_dict.pop(legacy_key)
        metadata = getattr(state_dict, '_metadata', None)
        if metadata is not None:
            legacy_module = legacy_prefix.rstrip('.')
            current_module = current_prefix.rstrip('.')
            for key in list(metadata):
                if key == legacy_module or key.startswith(legacy_prefix):
                    metadata.setdefault(current_module + key[len(legacy_module):], metadata[key])
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def init_weights(self):
        if self.task_config['with_det']:
            self.det_head.init_weights()
        if self.task_config['with_map']:
            self.map_head.init_weights()
        if self.task_config['with_motion_plan']:
            self.motion_plan_head.init_weights()

    def _forward_task_layer(self, head, features, **kwargs):
        """Run one task decoder stage using its shared feature-state contract."""
        state_keys = (
            'instance_feature',
            'temp_instance_feature',
            'temp_anchor_embed',
            'anchor',
            'anchor_embed',
            'time_interval',
        )
        state = {key: features[key] for key in state_keys}
        return head.forward_one_layer(**state, dn_metas=None, attn_mask=None, **kwargs)

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        unformated_feature_maps: Optional[Union[torch.Tensor, List]],
        metas: dict,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        if self.training:
            query_drop = random.uniform(0, 1) < self.query_drop_rate
        else:
            query_drop = False

        if self.temporal_queue is not None:
            timestamp = metas['timestamp']
            self.temporal_queue.check(feature_maps, timestamp)
            temporal_img_features = self.temporal_queue.get_feature(
                current_feature=feature_maps, current_meta=metas, cat_features=False
            )[0]
            self.temporal_queue.cache(feature_maps, timestamp, meta=metas)
        else:
            temporal_img_features = None

        det_features = self.det_head.get_feature(feature_maps, metas)
        map_features = self.map_head.get_feature(feature_maps, metas)
        plan_features = self.motion_plan_head.get_feature(feature_maps, metas)
        if self.dual_plan_head is not None:
            dual_plan_features = self.dual_plan_head.get_feature(feature_maps, metas)

        all_selected_keypoints = []

        for i in range(self.num_decoder):

            det_instance_feature = self._forward_task_layer(
                self.det_head,
                det_features,
                layer_index=i,
                feature_maps=feature_maps,
                metas=metas,
                stage=0,
            )
            det_features['instance_feature'] = det_instance_feature

            map_instance_feature = self._forward_task_layer(
                self.map_head,
                map_features,
                layer_index=i,
                feature_maps=feature_maps,
                metas=metas,
                stage=0,
            )
            map_features['instance_feature'] = map_instance_feature

            plan_instance_feature = self._forward_task_layer(
                self.motion_plan_head,
                plan_features,
                layer_index=i,
                feature_maps=feature_maps,
                temporal_img_features=temporal_img_features,
                metas=metas,
                stage=0,
                batch_size=batch_size,
                cmd_embedding=plan_features['cmd_embedding'],
                ego_status_embedding=plan_features['ego_status_embedding'],
            )
            plan_features['instance_feature'] = plan_instance_feature

            if self.dual_plan_head is not None:
                dual_plan_instance_features = self._forward_task_layer(
                    self.dual_plan_head,
                    dual_plan_features,
                    layer_index=i,
                    feature_maps=feature_maps,
                    temporal_img_features=temporal_img_features,
                    metas=metas,
                    stage=0,
                    batch_size=batch_size,
                    cmd_embedding=dual_plan_features['cmd_embedding'],
                    ego_status_embedding=dual_plan_features['ego_status_embedding'],
                )
                dual_plan_features['instance_feature'] = dual_plan_instance_features

            if i >= 1:
                (
                    det_features,
                    percepted_trajs,
                    percepted_anchors,
                ) = self.fusion_model.obtain_perception_object_features(
                    det_features, num_objects=self.fusion_model.plan_num_det, data=metas
                )
                det_features, map_features, plan_features = self.fusion_model(
                    det_features=det_features,
                    map_features=map_features,
                    plan_features=plan_features,
                    layer_index=i,
                    ignore_query=query_drop,
                )
            else:
                det_features['agent_corner'] = [[]]
                det_features['traj_embed'] = [[]]
                det_features['distance_positions'] = [[]]
                det_features['agent_traj'] = [[]]

            det_output = self._forward_task_layer(
                self.det_head,
                det_features,
                layer_index=i,
                feature_maps=feature_maps,
                metas=metas,
                stage=1,
            )

            det_features.update(det_output)

            map_output = self._forward_task_layer(
                self.map_head,
                map_features,
                layer_index=i,
                feature_maps=feature_maps,
                metas=metas,
                stage=1,
            )
            map_features.update(map_output)

            det_selected_keypoints = self.det_head.obtain_key_objects(
                det_output, num_objects=self.num_key_object
            )
            map_selected_keypoints = self.map_head.obtain_key_objects(
                map_output, num_objects=self.num_key_map, use_anchor=True
            )
            unified_selected_keypoints = {
                'det_selected_keypoints': det_selected_keypoints.clone().detach(),
                'map_selected_keypoints': map_selected_keypoints.clone().detach(),
            }
            all_selected_keypoints.append(unified_selected_keypoints)

            if self.dual_plan_head is not None:
                dual_plan_output = self._forward_task_layer(
                    self.dual_plan_head,
                    dual_plan_features,
                    layer_index=i,
                    feature_maps=feature_maps,
                    temporal_img_features=temporal_img_features,
                    metas=metas,
                    stage=1,
                    batch_size=batch_size,
                    cmd_embedding=dual_plan_features['cmd_embedding'],
                    ego_status_embedding=dual_plan_features['ego_status_embedding'],
                    given_keypoints=all_selected_keypoints[-1],
                )
                dual_plan_features.update(dual_plan_output)

                self.fusion_model.forward_merge(plan_features, dual_plan_features, i)
            else:
                dual_plan_features = {}

            key_input_instance_features = dual_plan_features.get('instance_feature', None)
            key_input_anchor_features = dual_plan_features.get('anchor_embed', None)
            plan_output = self._forward_task_layer(
                self.motion_plan_head,
                plan_features,
                layer_index=i,
                feature_maps=feature_maps,
                temporal_img_features=temporal_img_features,
                metas=metas,
                stage=1,
                batch_size=batch_size,
                cmd_embedding=plan_features['cmd_embedding'],
                ego_status_embedding=plan_features['ego_status_embedding'],
                addtion_instance_features=key_input_instance_features,
                addtion_anchor_embedding=key_input_anchor_features,
            )
            plan_features.update(plan_output)

        if self.longitudinal_head is not None:
            use_gt_anchor = self.longitudinal_head.use_gt_anchor
            if use_gt_anchor and self.training:
                num_modal = plan_features['prediction'][-1].shape[1]
                drivepath_anchor = metas['spatial_interval_2_traj'].to(
                    dtype=plan_features['prediction'][-1].dtype
                )
                drivepath_anchor = rearrange(drivepath_anchor, 'b t d -> b (t d)')
                drivepath_anchor = drivepath_anchor.unsqueeze(1)
                drivepath_anchor = drivepath_anchor.repeat(1, num_modal, 1)
            else:
                drivepath_anchor = plan_features['prediction'][-1].clone().detach()

            reference_feature = plan_features['instance_feature'].detach()
            reference_feature = rearrange(
                reference_feature, 'b (n m) d -> b n m d', n=self.motion_plan_head.sampler.num_modal
            )

            reference_feature = reference_feature.mean(dim=2)

            longitudinal_features = self.longitudinal_head.get_feature(
                prediction=drivepath_anchor,
                reference_feature=reference_feature,
                metas=metas,
            )
            drive_path_anchor = longitudinal_features['anchor'].clone().detach()

            if self.longitudinal_head.decode_perception_query:
                (
                    det_features,
                    longitudinal_percepted_trajs,
                    longitudinal_percepted_anchors,
                ) = self.longitudinal_head.obtain_perception_object_features(
                    det_features, num_objects=self.longitudinal_head.plan_num_det, data=metas
                )

            img_tokens = self.longitudinal_head.obtain_image_tokens(
                img_features=feature_maps,
                metas=metas,
            )
            longitudinal_features['img_tokens'] = img_tokens

            for i in range(self.num_decoder):
                longitudinal_output = self.longitudinal_head(
                    det_features=det_features,
                    map_features=map_features,
                    plan_features=longitudinal_features,
                    layer_index=i,
                    metas=metas,
                    ignore_query=query_drop,
                )
                if longitudinal_output is not None:
                    longitudinal_features.update(longitudinal_output)

            longitudinal_final_output = self.longitudinal_head.update_output(
                longitudinal_features['prediction'],
                longitudinal_features['scores'],
                longitudinal_features['drivepath'],
                longitudinal_features['predicted_distance'],
                prefix='longitudinal',
            )
        else:
            drive_path_anchor = None

        det_final_output = self.det_head.update_instance_bank_and_output(
            feature_maps=feature_maps,
            prediction=det_features['prediction'],
            classification=det_features['classification'],
            quality=det_features['quality'],
            cls=det_features['cls'],
            instance_feature=det_features['instance_feature'],
            anchor=det_features['anchor'],
            anchor_embed=det_features['anchor_embed'],
            key_points=det_features['key_points'],
            metas=metas,
            dn_metas=None,
            traj=det_features['traj'],
            traj_cls=det_features['traj_cls'],
            agents_drivepath=det_features['agents_drivepath'],
            traffic_state=det_features['traffic_state'],
        )
        map_final_output = self.map_head.update_instance_bank_and_output(
            feature_maps=feature_maps,
            prediction=map_features['prediction'],
            classification=map_features['classification'],
            quality=map_features['quality'],
            cls=map_features['cls'],
            instance_feature=map_features['instance_feature'],
            anchor=map_features['anchor'],
            anchor_embed=map_features['anchor_embed'],
            key_points=None,
            metas=metas,
            dn_metas=None,
            traj=map_features['traj'],
            traj_cls=map_features['traj_cls'],
            agents_drivepath=map_features['agents_drivepath'],
            traffic_state=map_features['traffic_state'],
        )
        plan_final_output = self.motion_plan_head.update_instance_bank_and_output(
            feature_maps=feature_maps,
            prediction=plan_features['prediction'],
            classification=plan_features['classification'],
            quality=plan_features['quality'],
            cls=plan_features['cls'],
            key_points=plan_features['key_points'],
            pred_ego_status=plan_features['pred_ego_status'],
            collision_matrixs=plan_features['collision_matrixs'],
            instance_feature=plan_features['instance_feature'],
            anchor=plan_features['anchor'],
            anchor_embed=plan_features['anchor_embed'],
            metas=metas,
            dn_metas=None,
        )

        if self.dual_plan_head is not None:
            dual_plan_final_output = self.dual_plan_head.update_instance_bank_and_output(
                feature_maps=feature_maps,
                prediction=dual_plan_features['prediction'],
                classification=dual_plan_features['classification'],
                quality=dual_plan_features['quality'],
                cls=dual_plan_features['cls'],
                key_points=dual_plan_features['key_points'],
                pred_ego_status=dual_plan_features['pred_ego_status'],
                collision_matrixs=dual_plan_features['collision_matrixs'],
                instance_feature=dual_plan_features['instance_feature'],
                anchor=dual_plan_features['anchor'],
                anchor_embed=dual_plan_features['anchor_embed'],
                metas=metas,
                dn_metas=None,
            )

            meraged_output = self.fusion_model.obtain_plan_output()

        if self.longitudinal_head is not None:
            self.longitudinal_head.update_instance_bank(
                longitudinal_features['instance_feature'],
                longitudinal_features['anchor'],
                longitudinal_features['score'],
                metas=metas,
            )

        if self.longitudinal_head is not None:

            for keys in longitudinal_final_output.keys():
                if keys in plan_final_output:
                    print(
                        f"WARNING: longitudinal_final_output has same keys as plan_final_output: {keys}. This may cause confusion."
                    )
                else:
                    plan_final_output[keys] = longitudinal_final_output[keys]

        if self.dual_plan_head is not None:

            for keys in plan_features.keys():
                if keys in dual_plan_final_output:
                    new_keys = 'dual_' + keys
                    plan_final_output[new_keys] = dual_plan_final_output[keys]

            for keys in meraged_output.keys():
                if keys in plan_final_output:
                    new_keys = 'merage_' + keys
                    plan_final_output[new_keys] = meraged_output[keys]

        self.det_head.reset_output()
        self.map_head.reset_output()
        self.motion_plan_head.reset_output()
        if self.longitudinal_head is not None:
            self.longitudinal_head.reset_output()
        if self.dual_plan_head is not None:
            self.dual_plan_head.reset_output()
            self.fusion_model.reset_output()

        plan_final_output['selected_keypoints'] = all_selected_keypoints
        plan_final_output['drive_path_anchor'] = drive_path_anchor

        plan_final_output['agent_corner'] = det_features['agent_corner']
        plan_final_output['agent_traj_embed'] = det_features['traj_embed']
        plan_final_output['agent_distance_positions'] = det_features['distance_positions']
        plan_final_output['agent_traj'] = det_features['agent_traj']

        if self.longitudinal_head is not None:
            plan_final_output['longitudinal_agent_corner'] = det_features['longitudinal_agent_corner']
            plan_final_output['longitudinal_agent_traj_embed'] = det_features['longitudinal_traj_embed']
            plan_final_output['longitudinal_agent_distance_positions'] = det_features[
                'longitudinal_distance_positions'
            ]
            plan_final_output['longitudinal_agent_traj'] = det_features['longitudinal_agent_traj']
            plan_final_output['longitudinal_agent_confidence'] = det_features['longitudinal_agent_confidence']
            plan_final_output['longitudinal_agent_cls'] = det_features['longitudinal_agent_cls']
            plan_final_output['light_state'] = det_features['longitudinal_light_state']

            det_final_output['longitudinal_percepted_trajs'] = longitudinal_percepted_trajs
            det_final_output['longitudinal_percepted_anchors'] = longitudinal_percepted_anchors

        det_final_output['percepted_trajs'] = percepted_trajs
        det_final_output['percepted_anchors'] = percepted_anchors

        return det_final_output, map_final_output, plan_final_output, None

    def loss(self, model_outs, data):
        det_output, map_output, planning_output, motion_output = model_outs
        losses = dict()

        train_det = self.task_config.get("train_det", True)
        train_map = self.task_config.get("train_map", True)
        train_spatial = self.task_config.get("train_spatial", True)
        train_lon = self.task_config.get("train_lon", True)

        if self.task_config['with_det'] and train_det:
            loss_det = self.det_head.loss(det_output, data)
            losses.update(loss_det)

        if self.task_config['with_map'] and train_map:
            loss_map = self.map_head.loss(map_output, data)
            losses.update(loss_map)

        if self.task_config['with_motion_plan'] and (train_spatial or train_lon):
            loss_plan = self.motion_plan_head.loss(
                planning_output,
                data,
            )
            if not train_spatial or not train_lon:
                filtered = {}
                for k, v in loss_plan.items():
                    is_longitudinal = k.startswith("longitudinal_")
                    if is_longitudinal and train_lon:
                        filtered[k] = v
                    elif (not is_longitudinal) and train_spatial:
                        filtered[k] = v
                loss_plan = filtered
            losses.update(loss_plan)

        return losses

    def post_process(self, model_outs, data):
        det_output, map_output, planning_output, motion_output = model_outs
        if self.task_config['with_det']:
            det_result = self.det_head.post_process(det_output)
            batch_size = len(det_result)

        if self.task_config['with_map']:
            map_result = self.map_head.post_process(map_output)
            batch_size = len(map_result)

        if self.task_config['with_motion_plan']:

            planning_result, motion_result = self.motion_plan_head.post_process(
                planning_output, keep_mm=True, data=data
            )

        results = [dict() for _ in range(batch_size)]
        for i in range(batch_size):
            if self.task_config['with_det']:
                results[i].update(det_result[i])
            if self.task_config['with_map']:
                results[i].update(map_result[i])
            if self.task_config['with_motion_plan']:
                if motion_result is not None:
                    results[i].update(motion_result[i])
                results[i].update(planning_result[i])

        return results


@HEADS.register_module()
class QueryFusion(BaseModule):
    """Fuse task queries while preserving each task-specific decoder state."""

    def __init__(
        self,
        graph_model: dict,
        distance_attn_model: dict,
        plan_mearge_model: dict,
        agent_anchor_encoder: dict,
        map_anchor_encoder: dict,
        norm_layer: dict,
        operation_order: List[str],
        merage_operation_order: List[str],
        num_query=1036,
        embed_dims=256,
        num_det=900,
        num_map=100,
        plan_num_det=50,
        plan_num_map=10,
        motion_queue=None,
        decode_perception_query=False,
        augment_bbox_pro=0.0,
        gnn_update_plan=False,
    ):
        super(QueryFusion, self).__init__()

        self.num_map = num_map
        self.num_det = num_det

        self.plan_num_det = plan_num_det
        self.plan_num_map = plan_num_map

        if motion_queue is not None:
            self.motion_queue = build_from_cfg(motion_queue, PLUGIN_LAYERS)
        else:
            self.motion_queue = None

        self.decode_perception_query = decode_perception_query
        self.gnn_update_plan = gnn_update_plan
        self.op_config_map = {
            "temp_cross_gnn": [distance_attn_model, ATTENTION],
            "cross_gnn": [distance_attn_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "agent_cross_map": [graph_model, ATTENTION],
            "agent_cross_map_norm": [norm_layer, NORM_LAYERS],
            "temp_cross_norm": [norm_layer, NORM_LAYERS],
            "cross_norm": [norm_layer, NORM_LAYERS],
            "gnn_norm": [norm_layer, NORM_LAYERS],
            "map_cross_gnn": [graph_model, ATTENTION],
            "agent_cross_gnn": [graph_model, ATTENTION],
            "dual_plan_merage": [plan_mearge_model, PLUGIN_LAYERS],
            "decode_cross_gnn": [graph_model, ATTENTION],
            "decode_cross_norm": [norm_layer, NORM_LAYERS],
        }
        self.operation_order_layer = operation_order.copy()
        if operation_order is None:
            operation_order = [
                "cross_gnn",
                "norm",
                "gnn",
                "norm",
            ] * 6
            print("WARNING: operation_order is None in fusion head, using default operation_order")
        elif isinstance(operation_order[0], List):
            operation_order = [item for sublist in operation_order for item in sublist]
        self.operation_order = operation_order

        if merage_operation_order is not None:
            self.merage_operation_order_layer = merage_operation_order.copy()
            self.merage_operation_order = [
                item for sublist in merage_operation_order for item in sublist
            ]
        else:
            self.merage_operation_order_layer = None
            self.merage_operation_order = None

        pos = torch.arange(num_query, dtype=torch.float32)
        position_encoding = get_1d_sincos_pos_embed_from_grid(embed_dims, pos)
        position_encoding = torch.from_numpy(position_encoding).float()

        self.position_embd = nn.Embedding(num_query, embed_dims)
        self.position_embd.weight = nn.Parameter(position_encoding, requires_grad=False)

        if len(self.operation_order) > 0:
            self.class_embed = nn.Embedding(2, embed_dims)
            self.class_embed.weight.data.normal_(mean=0.0, std=0.02)

        self.layers = nn.ModuleList(
            [self.build(*self.op_config_map.get(op, [None, None])) for op in self.operation_order]
        )
        if self.merage_operation_order is not None:
            self.merage_layers = nn.ModuleList(
                [
                    self.build(*self.op_config_map.get(op, [None, None]))
                    for op in self.merage_operation_order
                ]
            )
        self.num_query = num_query
        self.augment_bbox_pro = augment_bbox_pro

        if self.decode_perception_query:
            self.traj_encoder = nn.Sequential(
                nn.Linear(448, embed_dims),
                nn.ReLU(),
            )

        self.key_points = []
        self.prediction = []
        self.classification = []
        self.quality = []
        self.collision_matrixs = []
        self.pred_ego_status = []

    def build(self, cfg, registry):
        if cfg is None:
            return None
        return build_from_cfg(cfg, registry)

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def reset_output(self):
        self.key_points = []
        self.prediction = []
        self.classification = []
        self.quality = []
        self.collision_matrixs = []
        self.pred_ego_status = []

    def forward_merge(
        self,
        plan_features,
        dual_plan_features,
        layer_index,
        prediction=None,
        classification=None,
        pred_ego_status=None,
        quality=None,
    ):

        """Fuse the main and auxiliary planning branches at one decoder layer."""
        if self.merage_operation_order_layer is None:
            output = {}
            return output

        if prediction is None:
            prediction = self.prediction
        if classification is None:
            classification = self.classification
        if quality is None:
            quality = self.quality
        if pred_ego_status is None:
            pred_ego_status = self.pred_ego_status

        start_index = 0
        for j in range(layer_index):
            start_index += len(self.merage_operation_order_layer[j])
        operation_order = self.merage_operation_order_layer[layer_index]
        for index, op in enumerate(operation_order):
            i = start_index + index
            if self.merage_layers[i] is None:
                continue
            elif op == "dual_plan_merage":
                instance_feature = (
                    plan_features['instance_feature'] + dual_plan_features['instance_feature']
                )

                anchor = None
                anchor_embed = None
                anchor, cls, qt, plan_ego_status = self.merage_layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=None,
                    return_cls=True,
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                pred_ego_status.append(plan_ego_status)

        self.prediction = prediction
        self.classification = classification
        self.quality = quality
        self.pred_ego_status = pred_ego_status

        output = {
            "classification": classification,
            "prediction": prediction,
            "quality": quality,
            "pred_ego_status": pred_ego_status,
            "instance_feature": instance_feature,
            "anchor_embed": anchor_embed,
            "cls": cls,
        }

        return output

    def obtain_plan_output(self):

        output = {}
        output.update(
            {
                "classification": self.classification,
                "prediction": self.prediction,
                "quality": self.quality,
                "pred_ego_status": self.pred_ego_status,
            }
        )
        return output

    def forward(self, det_features, map_features, plan_features, layer_index, ignore_query=False):

        start_index = 0
        for j in range(layer_index):
            start_index += len(self.operation_order_layer[j])
        operation_order = self.operation_order_layer[layer_index]
        for index, op in enumerate(operation_order):
            i = start_index + index
            if self.layers[i] is None:
                continue
            elif op == "gnn":

                instance_feature = self.gnn(
                    i,
                    det_features,
                    map_features,
                    plan_features,
                )

            elif op == "cross_gnn":
                plan_instance_feature = self.cross_gnn(
                    i,
                    det_features,
                    map_features,
                    plan_features,
                    ignore_query=ignore_query,
                )

                plan_features['instance_feature'] = plan_instance_feature
            elif op == "decode_cross_gnn":
                plan_instance_feature = self.cross_gnn_decode_query(
                    i,
                    det_features,
                    map_features,
                    plan_features,
                    ignore_query=ignore_query,
                )

                plan_features['instance_feature'] = plan_instance_feature

            elif op == "agent_cross_gnn":
                plan_instance_feature = self.perception_cross_gnn(
                    i,
                    layer_index,
                    det_features,
                    plan_features,
                    ignore_query=ignore_query,
                )

                plan_features['instance_feature'] = plan_instance_feature
            elif op == "map_cross_gnn":
                plan_instance_feature = self.perception_cross_gnn(
                    i,
                    layer_index,
                    map_features,
                    plan_features,
                    ignore_query=ignore_query,
                )

                plan_features['instance_feature'] = plan_instance_feature

            elif op == "temp_cross_gnn":
                plan_instance_feature = self.temp_cross_gnn(
                    i,
                    det_features,
                    map_features,
                    plan_features,
                    ignore_query=ignore_query,
                )

                plan_features['instance_feature'] = plan_instance_feature
            elif op == "temp_cross_norm":
                plan_features['instance_feature'] = self.layers[i](
                    plan_features['instance_feature']
                )
            elif op == "cross_norm":
                plan_features['instance_feature'] = self.layers[i](
                    plan_features['instance_feature']
                )
            elif op == "decode_cross_norm":
                plan_features['instance_feature'] = self.layers[i](
                    plan_features['instance_feature']
                )
            elif op == "gnn_norm":
                instance_feature = self.layers[i](instance_feature)

                det_features['instance_feature'] = instance_feature[:, : self.num_det]
                map_features['instance_feature'] = instance_feature[
                    :, self.num_det : self.num_det + self.num_map
                ]

                if self.gnn_update_plan:
                    plan_features['instance_feature'] = instance_feature[
                        :, self.num_det + self.num_map :
                    ]

            elif op == "agent_cross_map":
                agent_features = self.agent_cross_map(
                    i,
                    det_features,
                    map_features,
                )
            elif op == "agent_cross_map_norm":
                det_features['instance_feature'] = self.layers[i](agent_features)

            else:
                raise ValueError(f"Unsupported operation: {op}")

        return det_features, map_features, plan_features

    def perception_cross_gnn(
        self, index, layer_index, perception_instance, plan_instance, ignore_query=False
    ):

        temporal_instance_features = perception_instance['queue_instance_feature']
        temporal_anchor_embed = perception_instance['queue_anchor_embed']
        num_key_objects = temporal_instance_features.shape[1]

        current_instance_feature = perception_instance["instance_feature"]
        current_anchor_embed = perception_instance["anchor_embed"]
        current_classification = perception_instance["classification"][-1].sigmoid()

        current_confidence = current_classification.max(dim=-1).values
        _, (current_instance_feature_selected, current_anchor_embed_selected) = topk(
            current_confidence, num_key_objects, current_instance_feature, current_anchor_embed
        )

        if layer_index > 2:
            temporal_instance_features = temporal_instance_features.permute(0, 2, 1, 3)
            current_instance_feature_selected = current_instance_feature_selected.unsqueeze(1)
            perception_instance_feature = torch.cat(
                [temporal_instance_features, current_instance_feature_selected], dim=1
            )
            perception_instance_feature = rearrange(
                perception_instance_feature, 'b t n d -> b (t n) d'
            )

            temporal_anchor_embed = temporal_anchor_embed.permute(0, 2, 1, 3)
            current_anchor_embed_selected = current_anchor_embed_selected.unsqueeze(1)
            perception_anchor_embed = torch.cat(
                [temporal_anchor_embed, current_anchor_embed_selected], dim=1
            )

            perception_instance_position = rearrange(
                perception_anchor_embed, 'b t n d -> b (t n) d'
            )

        else:

            perception_instance_feature = current_instance_feature_selected
            perception_instance_position = current_anchor_embed_selected

        plan_instance_feature = plan_instance['instance_feature']
        plan_instance_position = plan_instance['anchor_embed']

        plan_instance_feature = self.layers[index](
            plan_instance_feature,
            key=perception_instance_feature,
            value=perception_instance_position,
            query_pos=plan_instance_position,
            key_pos=perception_instance_position,
        )

        return plan_instance_feature

    def temp_cross_gnn(self, index, det_instance, map_instance, plan_instance, ignore_query=False):

        agent_instance_feature = det_instance["temp_instance_feature"]
        agent_anchor_embed = det_instance["temp_anchor_embed"]

        if agent_instance_feature is not None and ignore_query is False:
            det_confidence = det_instance["temp_confidence"]
            det_anchors = det_instance["temp_anchor"]
            _, (
                agent_instance_feature_selected,
                agent_anchor_embed_selected,
                det_anchors_selected,
            ) = topk(
                det_confidence,
                self.plan_num_det,
                agent_instance_feature,
                agent_anchor_embed,
                det_anchors,
            )

            map_instance_feature = map_instance["temp_instance_feature"]
            if map_instance_feature is not None:
                map_anchor_embed = map_instance["temp_anchor_embed"]
                map_confidence = map_instance["temp_confidence"]
                map_anchors = map_instance["temp_anchor"]
                _, (
                    map_instance_feature_selected,
                    map_anchor_embed_selected,
                    det_map_selected,
                ) = topk(
                    map_confidence,
                    self.plan_num_map,
                    map_instance_feature,
                    map_anchor_embed,
                    map_anchors,
                )

                distance_det, distance_map = self.get_distance(
                    det_anchors_selected, det_map_selected
                )
                perception_distance = torch.cat([distance_det, distance_map], dim=-1)

                bs = agent_instance_feature_selected.shape[0]
                class_embed = self.class_embed.weight.unsqueeze(0).expand(bs, -1, -1)

                agent_instance_feature_selected = agent_instance_feature_selected + class_embed[
                    :, 0, :
                ].unsqueeze(1)
                map_instance_feature_selected = map_instance_feature_selected + class_embed[
                    :, 1, :
                ].unsqueeze(1)

                perception_instance_feature = torch.cat(
                    [agent_instance_feature_selected, map_instance_feature_selected], dim=1
                )
                perception_instance_position = torch.cat(
                    [agent_anchor_embed_selected, map_anchor_embed_selected], dim=1
                )

            else:
                distance_det, _ = self.get_distance(det_anchors_selected, None)
                perception_distance = distance_det

                perception_instance_feature = agent_instance_feature_selected
                perception_instance_position = agent_anchor_embed_selected

            plan_instance_feature = plan_instance['instance_feature']
            plan_instance_position = plan_instance['anchor_embed']

            plan_instance_feature = self.layers[index](
                plan_instance_feature,
                key=perception_instance_feature,
                value=perception_instance_position,
                query_pos=plan_instance_position,
                key_pos=perception_instance_position,
                distances=perception_distance,
            )
        else:
            plan_instance_feature = plan_instance['instance_feature']
            plan_instance_position = plan_instance['anchor_embed']
            perception_distance = torch.zeros(
                plan_instance_feature.shape[0],
                plan_instance_feature.shape[1],
                device=plan_instance_feature.device,
            )
            plan_instance_feature = self.layers[index](
                plan_instance_feature,
                key=plan_instance_feature,
                value=plan_instance_feature,
                query_pos=plan_instance_position,
                key_pos=plan_instance_position,
                distances=perception_distance,
            )
        return plan_instance_feature

    def encode_distance(self, distances, dim, device):
        """Encode (B, N) distances with logarithmic radial bases emphasizing nearby objects."""

        log_dist = torch.log1p(distances)

        centers = torch.linspace(0, 3.0, dim, device=device)
        width = 0.3

        rbf = torch.exp(-((log_dist.unsqueeze(-1) - centers) ** 2) / (2 * width ** 2))
        return rbf

    def get_distance(self, det_anchors, map_anchors=None):
        det_positions = det_anchors[:, :, :2]
        distance_det = torch.norm(det_positions, dim=-1)
        if map_anchors is not None:
            map_positions = rearrange(map_anchors, 'b n (d k) -> b n d k', d=20, k=2)
            map_positions = map_positions.mean(dim=-2)
            distance_map = torch.norm(map_positions, dim=-1)
        else:
            distance_map = None
        return distance_det, distance_map

    def cross_gnn(self, index, det_instance, map_instance, plan_instance, ignore_query=False):

        plan_instance_feature = plan_instance['instance_feature']
        plan_instance_position = plan_instance['anchor_embed']

        if ignore_query is False:
            agent_instance_feature = det_instance["instance_feature"]
            agent_anchor_embed = det_instance["anchor_embed"]
            det_classification = det_instance["classification"][-1].sigmoid()
            det_anchors = det_instance["prediction"][-1]
            det_confidence = det_classification.max(dim=-1).values
            _, (
                agent_instance_feature_selected,
                agent_anchor_embed_selected,
                det_anchors_selected,
            ) = topk(
                det_confidence,
                self.plan_num_det,
                agent_instance_feature,
                agent_anchor_embed,
                det_anchors,
            )

            map_instance_feature = map_instance["instance_feature"]
            map_anchor_embed = map_instance["anchor_embed"]
            map_classification = map_instance["classification"][-1].sigmoid()
            map_anchors = map_instance["prediction"][-1]

            map_confidence = map_classification.max(dim=-1).values
            _, (map_instance_feature_selected, map_anchor_embed_selected, det_map_selected) = topk(
                map_confidence,
                self.plan_num_map,
                map_instance_feature,
                map_anchor_embed,
                map_anchors,
            )

            distance_det, distance_map = self.get_distance(det_anchors_selected, det_map_selected)
            perception_distance = torch.cat([distance_det, distance_map], dim=-1)

            bs = agent_instance_feature_selected.shape[0]

            class_embed = self.class_embed.weight.unsqueeze(0).expand(bs, -1, -1)

            agent_instance_feature_selected = agent_instance_feature_selected + class_embed[
                :, 0, :
            ].unsqueeze(1)
            map_instance_feature_selected = map_instance_feature_selected + class_embed[
                :, 1, :
            ].unsqueeze(1)

            perception_instance_feature = torch.cat(
                [agent_instance_feature_selected, map_instance_feature_selected], dim=1
            )
            perception_instance_position = torch.cat(
                [agent_anchor_embed_selected, map_anchor_embed_selected], dim=1
            )

        else:
            perception_instance_feature = plan_instance_feature
            perception_instance_position = plan_instance_position
            perception_distance = None

        plan_instance_feature = self.layers[index](
            plan_instance_feature,
            key=perception_instance_feature,
            value=perception_instance_position,
            query_pos=plan_instance_position,
            key_pos=perception_instance_position,
            distances=perception_distance,
        )

        return plan_instance_feature

    def gnn(self, index, det_instance, map_instance, plan_instance):

        conconated_instance_feature = torch.cat(
            [
                det_instance['instance_feature'],
                map_instance['instance_feature'],
                plan_instance['instance_feature'],
            ],
            dim=1,
        )
        conconated_instance_position = torch.cat(
            [
                det_instance['anchor_embed'],
                map_instance['anchor_embed'],
                plan_instance['anchor_embed'],
            ],
            dim=1,
        )

        position_ids = torch.arange(conconated_instance_feature.shape[1]).unsqueeze(0).cuda()
        conconated_instance_feature = conconated_instance_feature + self.position_embd(position_ids)

        instance_feature = self.layers[index](
            conconated_instance_feature,
            key=conconated_instance_feature,
            value=conconated_instance_feature,
            query_pos=conconated_instance_position,
            key_pos=conconated_instance_position,
        )

        return instance_feature

    def decode_box(self, box):
        yaw = torch.atan2(box[..., SIN_YAW], box[..., COS_YAW])
        box = torch.cat(
            [
                box[..., [X, Y, Z]],
                box[..., [W, L, H]].exp(),
                yaw[..., None],
                box[..., VX:],
            ],
            dim=-1,
        )
        return box

    def obtain_perception_object_features(
        self, perception_instance, history_horizon=3, time_horizon=3, num_objects=10, data=None
    ):

        """Select perception queries and encode their geometry for planning attention."""
        if not self.decode_perception_query:
            bs = perception_instance["instance_feature"].shape[0]
            device = perception_instance["instance_feature"].device
            data['augment_bbox'] = torch.zeros(size=(bs,), dtype=torch.bool, device=device)
            return perception_instance, None, None

        perception_instance_feature = perception_instance["instance_feature"]

        perception_classification = perception_instance["classification"][-1].sigmoid()
        perception_anchors = perception_instance["prediction"][-1]
        perception_confidence = perception_classification.max(dim=-1).values
        perception_trajs = perception_instance['traj'][-1]
        _, (
            percetion_instance_feature_selected,
            perception_trajs_selected,
            perception_anchors_selected,
        ) = topk(
            perception_confidence,
            num_objects,
            perception_instance_feature,
            perception_trajs,
            perception_anchors,
        )
        perception_anchors_selected = self.decode_box(perception_anchors_selected)[:, :, :-1]
        bs = perception_instance_feature.shape[0]

        augment_bbox = random.random() < self.augment_bbox_pro
        if augment_bbox and self.training:

            valid_flag = data['valid_flag']
            data['augment_bbox'] = valid_flag
            augmented_trajs = []
            augmented_bboxes = []
            fake_bbox = data['fake_bboxes_3d']
            fake_trajs = data['fake_agent_fut_trajs']
            fake_trajs = rearrange(fake_trajs, 'b n d -> b (n d)')

            for i in range(bs):
                trajs, bboxes = augment_perception_with_fake_bbox(
                    perception_trajs_selected[i],
                    perception_anchors_selected[i],
                    fake_trajs[i],
                    fake_bbox[i],
                    valid_flag[i],
                    given_index=-1,
                )
                augmented_trajs.append(trajs)
                augmented_bboxes.append(bboxes)

            perception_trajs_selected = torch.stack(augmented_trajs, dim=0)
            perception_anchors_selected = torch.stack(augmented_bboxes, dim=0)

        else:

            data['augment_bbox'] = torch.zeros(
                size=(bs,), dtype=torch.bool, device=perception_trajs_selected.device
            )

        fut_bbox_batch = rearrange(perception_trajs_selected, 'b n (t c) -> b n t c', t=6, c=2)

        fut_bbox_batch = torch.cat(
            (
                torch.zeros(bs, num_objects, 1, 2).to(perception_trajs_selected.device),
                fut_bbox_batch,
            ),
            dim=2,
        )

        half_fut_bbox_length = perception_anchors_selected[:, :, 3] / 2
        half_fut_bbox_width = perception_anchors_selected[:, :, 4] / 2
        bbox_init_center = perception_anchors_selected[:, :, :2]
        bbox_init_yaw = perception_anchors_selected[:, :, 6:7]
        bbox_init_state = torch.cat((bbox_init_center, bbox_init_yaw), dim=-1)

        fut_bbox_batch = rearrange(fut_bbox_batch, 'b n t c -> (b n) t c').detach()
        half_fut_bbox_length = rearrange(half_fut_bbox_length, 'b n -> (b n)').detach()
        half_fut_bbox_width = rearrange(half_fut_bbox_width, 'b n -> (b n)').detach()
        bbox_init_state = rearrange(bbox_init_state, 'b n c -> (b n) c').detach()

        agent_corner, agent_traj = compute_corners_torch(
            fut_bbox_batch.clone(),
            half_fut_bbox_length.clone(),
            half_fut_bbox_width.clone(),
            bbox_init_state.clone(),
            return_trajs=True,
        )

        agent_corner = rearrange(agent_corner, '(b n) t m c -> b n t m c', b=bs, n=num_objects)
        agent_traj = rearrange(agent_traj, '(b n) t d -> b n t d', b=bs, n=num_objects)

        if 'agent_corner' not in perception_instance:
            perception_instance['agent_corner'] = []
            perception_instance['traj_embed'] = []
            perception_instance['distance_positions'] = []
            perception_instance['agent_traj'] = []
        perception_instance['agent_traj'].append(agent_traj)
        perception_instance['agent_corner'].append(agent_corner)
        fut_t = agent_corner.shape[2]
        num_corner = agent_corner.shape[3]

        agent_corner = rearrange(agent_corner, 'b n t m c -> b (n t m) c', b=bs, n=num_objects)

        agent_corner_embed = gen_sineembed_for_position(agent_corner, hidden_dim=16)
        agent_corner_embed = rearrange(
            agent_corner_embed,
            'b (n t m) c -> b n (t m c)',
            b=bs,
            n=num_objects,
            t=fut_t,
            m=num_corner,
        )

        traj_embed = self.traj_encoder(agent_corner_embed)
        perception_instance['traj_embed'].append(traj_embed)

        distance_positions = perception_anchors_selected[:, :, :2].detach()
        distance_positions = torch.norm(distance_positions, dim=-1)
        perception_instance['distance_positions'].append(distance_positions)

        return perception_instance, perception_trajs_selected, perception_anchors_selected

    def agent_cross_map(self, index, det_instance, map_instance, ignore_query=False):

        agent_instance_feature = det_instance["instance_feature"]
        agent_anchor_embed = det_instance["anchor_embed"]

        map_instance_feature = map_instance["instance_feature"]
        map_anchor_embed = map_instance["anchor_embed"]
        map_classification = map_instance["classification"][-1].sigmoid()
        map_anchors = map_instance["prediction"][-1]

        map_confidence = map_classification.max(dim=-1).values
        _, (map_instance_feature_selected, map_anchor_embed_selected, det_map_selected) = topk(
            map_confidence, self.plan_num_map, map_instance_feature, map_anchor_embed, map_anchors
        )

        agent_instance_feature = self.layers[index](
            agent_instance_feature,
            key=map_instance_feature_selected,
            value=map_instance_feature_selected,
            query_pos=agent_anchor_embed,
            key_pos=map_anchor_embed_selected,
        )

        return agent_instance_feature

    def cross_gnn_decode_query(
        self, index, det_instance, map_instance, plan_instance, ignore_query=False
    ):

        plan_instance_feature = plan_instance['instance_feature']
        plan_instance_position = plan_instance['anchor_embed']

        det_traj_embed = det_instance['traj_embed'][-1]
        det_traj_distance = det_instance['distance_positions'][-1]

        plan_instance_feature = self.layers[index](
            plan_instance_feature + plan_instance_position,
            key=det_traj_embed,
            value=det_traj_embed,
            distances=det_traj_distance,
        )

        return plan_instance_feature
