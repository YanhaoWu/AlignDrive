from typing import List, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
from einops.einops import rearrange

from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import build_from_cfg
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.models import HEADS
from mmdet.core import reduce_mean
from mmdet.models import build_loss


from ..blocks import linear_relu_ln


from projects_sp.mmdet3d_plugin.models.attention import gen_sineembed_for_position
from projects_sp.mmdet3d_plugin.models.instance_bank import Img_instance_queue

from projects_sp.mmdet3d_plugin.models.utils import project_stepwise_waypoints_to_drivepath

from projects_sp.mmdet3d_plugin.models.longitudinal_planning.score_module.compute_b2d_score_wyh import (
    score_prediction,
)


__all__ = ["LongitudinalPlanningModule"]


@HEADS.register_module()
class LongitudinalPlanningModule(BaseModule):
    """Decode aligned planning queries, compute training losses, and select trajectories."""

    def __init__(
        self,
        instance_bank: dict,
        anchor_encoder: dict,
        graph_model: dict,
        img_feature_extrctor: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        spdeformable_model: dict,
        refine_layer: dict,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        anchor_encoder_low_frency: dict = None,
        objectdeformable_model: dict = None,
        temp_graph_model: dict = None,
        decoder: dict = None,
        sampler: dict = None,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        gt_id_key: str = "instance_id",
        with_instance_id: bool = True,
        task_prefix: str = 'det',
        reg_weights: List = None,
        operation_order=None,
        cls_threshold_to_reg: float = -1,
        dn_loss_weight: float = 5.0,
        decouple_attn: bool = True,
        init_cfg: dict = None,
        plan_loss_cls=None,
        plan_loss_reg=None,
        plan_loss_status=None,
        plan_loss_score=None,
        collision_loss=None,
        longitudinal_head_scale=1.0,
        keypoint_loss=None,
        use_ego_status=False,
        temporal_img=-1,
        other_output_pretext=None,
        final_used_head='base',
        gt_modal='distance',
        control_setting=None,
        longitudinal_train_use_grand=None,
        navigation_module=None,
        conditional_loss_weighting=None,
        **kwargs,
    ):
        super(LongitudinalPlanningModule, self).__init__(init_cfg)
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.gt_id_key = gt_id_key
        self.with_instance_id = with_instance_id
        self.task_prefix = task_prefix
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.dn_loss_weight = dn_loss_weight
        self.decouple_attn = decouple_attn
        self.plan_loss_cls = build_loss(plan_loss_cls)
        self.plan_loss_reg = build_loss(plan_loss_reg)
        if plan_loss_status is not None:
            self.plan_loss_status = build_loss(plan_loss_status)
        if plan_loss_score is not None:
            self.plan_loss_score = build_loss(plan_loss_score)
        if collision_loss is not None:
            self.collision_loss = build_loss(collision_loss)

        if keypoint_loss is not None:
            self.keypoint_loss = build_loss(keypoint_loss)
        else:
            self.keypoint_loss = None
        self.gt_modal = gt_modal
        self.longitudinal_head_scale = longitudinal_head_scale
        self.num_sample_waypoints = kwargs.get('num_sample_waypoints', 6)
        self.num_cmd = kwargs.get('num_cmd', 3)
        self.use_GAN = kwargs.get('use_GAN', False)
        self.use_tp = kwargs.get('use_tp', False)
        self.use_command = kwargs.get('use_command', True)
        self.collision_check = kwargs.get('collision_check', False)
        self.num_temporal_plan = kwargs.get('num_temporal_plan', 1)
        self.force_check_timeinterval = kwargs.get('force_check_timeinterval', False)
        self.gloalpoint_extend = kwargs.get('gloalpoint_extend', False)
        self.other_output_pretext = other_output_pretext
        self.final_used_head = final_used_head
        self.conditional_loss_weighting = conditional_loss_weighting or {}

        self.control_setting = control_setting
        if self.control_setting is None:
            self.control_setting = dict(
                spatial="spatial_interval_2_traj",
                collision_resscore=False,
                light_state_enhancement=3,
                temporal="temporal_interval_2_traj",
                perferred_anchor=-1,
                valid_waypoints=6,
            )
            print("[WARNING] control_setting is not set")
        self.control_spatial_grand = self.control_setting['spatial']
        self.control_temporal_grand = self.control_setting['temporal']
        self.light_state_enhancement = self.control_setting.get('light_state_enhancement', 3)
        self.collision_resscore = self.control_setting.get('collision_resscore', True)
        self.post_process_score = self.control_setting.get('post_process_score', False)
        self.iter_count = 0

        self.longitudinal_train_use_grand = longitudinal_train_use_grand
        if longitudinal_train_use_grand is None:
            self.longitudinal_train_use_grand = dict(
                temporal='temporal_interval_2_traj',
                spatial='spatial_interval_2_traj',
                augmentation='fake_defensive_ego_fut_trajs_5hz',
            )

        self.tp_res = kwargs.get('tp_res', False)
        self.command_res = kwargs.get('command_res', False)
        self.auxiliary_loss = kwargs.get('auxiliary_loss', None)
        if self.auxiliary_loss is None:
            warnings.warn(
                "auxiliary_loss is not set, default to False. "
                "This may cause dynamic loss not working as expected."
            )
            self.auxiliary_loss = {
                'dynamic_loss': False,
                'dynamic_loss_longitudinal': False,
                'dynamic_horzion': 2,
                'dynamic_dead_range': 0.5,
                'dynamic_loss_scale': 0.02,
                'lateral_loss': False,
                'lateral_loss_longitudinal': False,
                'lateral_loss_scale': 0.2,
                'dynamic_guide_augment': False,
                'dynamic_all_trajs': False,
                'dynamic_loss_other_trajs_scale': 0.1,
                'start_iter': 2000,
            }

        self.temporal_img = temporal_img
        self.temporal_gap = kwargs['temporal_gap'] if 'temporal_gap' in kwargs else 1
        self.use_perception_query = kwargs.get('use_perception_query', False)
        self.num_det = kwargs.get('num_det', 32)
        self.num_map = kwargs.get('num_map', 32)
        self.anchor_mearge = kwargs.get('anchor_mearge', False)
        self.anchor_supervise = kwargs.get('anchor_supervise', False)
        self.augment_bbox = kwargs.get('augment_bbox', False)
        self.augment_bbox_longitudinal = kwargs.get('augment_bbox_longitudinal', False)
        self.plan_control = kwargs.get('plan_control', False)
        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights
        if operation_order is None:
            operation_order_layer = [
                "temp_gnn",
                "gnn",
                "norm",
                "perception_gnn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder

            operation_order_layer = operation_order_layer[3:]
            operation_order_layer = [operation_order_layer]
        elif isinstance(operation_order[0], (list, tuple)):
            operation_order_layer = [list(layer) for layer in operation_order]
        else:
            operation_order_layer = []
            current_layer = []
            for op in operation_order:
                current_layer.append(op)
                if op == "refine":
                    operation_order_layer.append(current_layer)
                    current_layer = []
            if current_layer:
                operation_order_layer.append(current_layer)
        self.operation_order_layer = operation_order_layer
        self.operation_order = [op for layer in operation_order_layer for op in layer]

        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.instance_bank = build(instance_bank, PLUGIN_LAYERS)
        self.anchor_encoder = build(anchor_encoder, POSITIONAL_ENCODING)
        if anchor_encoder_low_frency is not None:
            self.anchor_encoder_low_frency = build(anchor_encoder_low_frency, POSITIONAL_ENCODING)
            self.low_frency_grand = ['temporal_interval_5_traj', 'spatial_interval_5_traj']
        else:
            self.anchor_encoder_low_frency = None

        self.sampler = build(sampler, BBOX_SAMPLERS)
        self.decoder = build(decoder, BBOX_CODERS)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "temp_gnn_both": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "agent_cross_gnn": [graph_model, ATTENTION],
            "map_cross_gnn": [graph_model, ATTENTION],
            "keypoint_cross_gnn": [graph_model, ATTENTION],
            "img_cross_gnn": [graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "deformable": [deformable_model, ATTENTION],
            "spdeformable": [spdeformable_model, ATTENTION],
            "obdeformable": [objectdeformable_model, ATTENTION],
            "refine": [refine_layer, PLUGIN_LAYERS],
            "global_extract": [img_feature_extrctor, PLUGIN_LAYERS],
            "global_gnn": [graph_model, ATTENTION],
            "path_gnn": [graph_model, ATTENTION],
            "path_norm": [norm_layer, NORM_LAYERS],
            "add_navigation": [navigation_module, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [build(*self.op_config_map.get(op, [None, None])) for op in self.operation_order]
        )
        self.embed_dims = self.instance_bank.embed_dims
        if self.decouple_attn:
            self.fc_before = nn.Linear(self.embed_dims, self.embed_dims * 2, bias=False)
            self.fc_after = nn.Linear(self.embed_dims * 2, self.embed_dims, bias=False)
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        if self.use_tp:
            input_dim = 12 if self.gloalpoint_extend else 2
            self.tp_encoder = nn.Sequential(*linear_relu_ln(self.embed_dims, 1, 2, input_dim))
        if self.use_command:
            if self.num_cmd == 6:
                print("Warning, no codebook position for nosie commands")
            self.cmd_codebook = nn.Embedding(self.num_cmd, self.embed_dims)

        self.use_ego_status = use_ego_status
        if self.use_ego_status:
            self.egp_status_encoder = nn.Sequential(
                *linear_relu_ln(self.embed_dims, 1, 1, 10),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
            print("-------------USING EGO STATUS-------------")

        if self.temporal_img > 0:
            self.temporal_queue = Img_instance_queue(
                queue_length=self.temporal_img,
                recurrent_temporal=False,
                temporal_gap=self.temporal_gap,
            )
        else:
            self.temporal_queue = None

        self.key_points = []
        self.prediction = []
        self.classification = []
        self.quality = []
        self.collision_matrixs = []
        self.pred_ego_status = []

        self.is_stuck = False

    def _sample_scenario_names(self, data):
        metas = data.get('img_metas', [])
        if hasattr(metas, 'data'):
            metas = metas.data
        if isinstance(metas, dict):
            metas = [metas]
        return [str(meta.get('scenario_name', '')) for meta in metas]

    def _conditional_weights(self, data, command, kind):
        names = self._sample_scenario_names(data)
        config = self.conditional_loss_weighting.get(kind, {})
        scenarios = set(config.get('scenarios', []))
        multiplier = float(config.get('multiplier', 1.0))
        command_ids = set(config.get('command_ids', []))
        command_ids_tensor = command.argmax(dim=-1) if command is not None else None
        device = command.device if command is not None else next(self.parameters()).device

        batch_size = command.shape[0] if command is not None else len(names)
        weights = torch.ones(batch_size, device=device, dtype=torch.float32)
        if multiplier == 1.0:
            return weights
        if len(names) != batch_size or any(not name for name in names):
            raise ValueError('Conditional loss weighting requires scenario_name for every sample')
        for index in range(batch_size):
            scenario_match = (
                not scenarios
                or names[index].replace('_1', '') in scenarios
                or names[index] in scenarios
            )
            command_match = command_ids_tensor is not None and (
                not command_ids or int(command_ids_tensor[index]) in command_ids
            )
            if scenario_match and command_match:
                weights[index] = multiplier
        return weights

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

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
        debug=False,
    ):

        """Run the standalone planning decoder; UnifiedHead uses the staged interface."""
        output = self.forward_split(
            feature_maps=feature_maps,
            metas=metas,
        )
        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_reg(
        self,
        cls_scores,
        reg_preds,
        pred_ego_status,
        data=None,
        pretext='base',
        use_vocab=True,
        model_outs=None,
    ):

        """Compute spatial trajectory classification, regression, and optional auxiliary losses."""
        output = {}
        num_modal = self.sampler.num_modal
        num_grand = self.sampler.num_grand
        grand_name = self.sampler.grand_name
        sp_grand_name = [name for name in grand_name if 'DS' not in name]
        num_drivestyle = num_grand - len(sp_grand_name)
        all_winner_modal = []
        for decoder_idx, (cls, reg, status) in enumerate(
            zip(cls_scores, reg_preds, pred_ego_status)
        ):
            bs = reg.shape[0]

            winner_modal, winner_ds, gt_traj_list, gt_mask_list = self.sampler.sample(
                data, reg, decoder_idx, use_vocab=use_vocab, augment_bbox=self.augment_bbox
            )
            all_winner_modal.append(winner_modal)
            reg = rearrange(
                reg,
                'b (n m) (t d) -> b n m t d',
                n=num_modal,
                m=num_grand,
                t=self.num_sample_waypoints,
                d=2,
            )
            selected_modal_waypoints = reg[torch.arange(reg.shape[0]), winner_modal]
            loss_grand = {}
            for i, name in enumerate(sp_grand_name):
                weights = gt_mask_list[i].unsqueeze(-1)
                selected_waypoints = selected_modal_waypoints[:, i]
                selected_gt = gt_traj_list[i]
                gt_command = data['gt_ego_fut_cmd']
                spatial_weight = (
                    self._conditional_weights(data, gt_command, 'spatial')
                    if 'spatial' in name
                    else None
                )
                longitudinal_weight = (
                    self._conditional_weights(data, gt_command, 'longitudinal')
                    if 'temporal' in name
                    else None
                )
                reg_loss = self.plan_loss_reg(
                    selected_waypoints,
                    selected_gt,
                    weight=weights,
                    name=name,
                    command=gt_command,
                    iter_count=self.iter_count,
                    sample_weight=spatial_weight,
                    longitudinal_weight=longitudinal_weight,
                )
                loss_grand[name] = reg_loss

                if self.auxiliary_loss['dynamic_loss'] and 'temporal' in name:

                    agent_corer = model_outs['agent_corner'][decoder_idx]
                    agent_traj = model_outs['agent_traj'][decoder_idx]
                    if len(agent_corer) > 0:

                        collision_weight = self._conditional_weights(data, gt_command, 'collision')
                        dynamic_loss = self.collision_loss(
                            selected_waypoints,
                            agent_corer,
                            agent_traj,
                            interactive_agents=None,
                            sample_weight=collision_weight,
                        )
                    else:
                        dynamic_loss = torch.zeros_like(reg_loss)

                    loss_grand[name + '_collision'] = dynamic_loss

            if num_drivestyle > 0:

                style_cls = cls['syt_cls'][torch.arange(bs), winner_modal]
                drive_style_cls_loss = self.plan_loss_cls(style_cls, winner_ds)
                loss_grand['drive_style'] = drive_style_cls_loss

                style_reg_5hz = selected_modal_waypoints[:, :3, :, :]
                style_reg_2hz = selected_modal_waypoints[:, 3:6, :, :]

                selected_style_5hz = style_reg_5hz[torch.arange(bs), winner_ds]
                selected_style_2hz = style_reg_2hz[torch.arange(bs), winner_ds]

                gt_style_traj_5hz = data['temporal_interval_2_traj']
                gt_style_traj_2hz = data['temporal_interval_5_traj']
                gt_style_mask_5hz = data['temporal_interval_2_mask']
                gt_style_mask_2hz = data['temporal_interval_5_mask']

                loss_grand['style_reg_5hz'] = self.plan_loss_reg(
                    selected_style_5hz, gt_style_traj_5hz, weight=gt_style_mask_5hz.unsqueeze(-1)
                )
                loss_grand['style_reg_2hz'] = self.plan_loss_reg(
                    selected_style_2hz, gt_style_traj_2hz, weight=gt_style_mask_2hz.unsqueeze(-1)
                )

                if self.iter_count <= self.auxiliary_loss['driving_style_start']:
                    if self.iter_count == self.auxiliary_loss['driving_style_start']:
                        print("Starting to use driving style loss")
                    loss_grand['drive_style'] = loss_grand['drive_style'] * 0.0001
                    loss_grand['style_reg_5hz'] = loss_grand['style_reg_5hz'] * 0.0001
                    loss_grand['style_reg_2hz'] = loss_grand['style_reg_2hz'] * 0.0001

            modal_class = cls['cls']
            modal_class = modal_class[:, :, 0]
            cls_weights = (
                torch.cat(gt_mask_list[: len(sp_grand_name)], dim=1).reshape(bs, -1).any(-1)
            )
            cls_loss = self.plan_loss_cls(modal_class, winner_modal, weight=cls_weights)

            gt_ego_status = data['ego_status'].unsqueeze(1).expand(-1, num_modal, -1)
            if status is not None:
                status_loss = self.plan_loss_status(status, gt_ego_status)
                if status_loss >= 1:
                    status_loss = status_loss * 0.01

            for keys in loss_grand.keys():
                loss_name = f"{pretext}_planning_loss_reg_{keys}_{decoder_idx}"
                output.update({loss_name: loss_grand[keys]})

            if num_drivestyle > 0:
                output.update(
                    {
                        f"{pretext}_planning_loss_cls_drive_style_{decoder_idx}": loss_grand[
                            'drive_style'
                        ],
                        f"{pretext}_planning_loss_reg_style_5hz_{decoder_idx}": loss_grand[
                            'style_reg_5hz'
                        ],
                        f"{pretext}_planning_loss_reg_style_2hz_{decoder_idx}": loss_grand[
                            'style_reg_2hz'
                        ],
                    }
                )

            output.update(
                {
                    f"{pretext}_planning_loss_cls_{decoder_idx}": cls_loss,
                }
            )

            if status is not None:
                output.update(
                    {
                        f"{pretext}_planning_loss_status_{decoder_idx}": status_loss,
                    }
                )

        return output, all_winner_modal

    @force_fp32(apply_to=("model_outs"))
    def loss_speed_aware_trajectory(
        self,
        scores,
        drive_path,
        reg_preds,
        pred_ego_status=None,
        predicted_distance=None,
        data=None,
        pretext='base',
        use_vocab=True,
        winner_modal=None,
        model_outs=None,
    ):

        """Supervise longitudinal displacements conditioned on the selected spatial path."""
        output = {}
        for decoder_idx, (score, reg, status) in enumerate(zip(scores, reg_preds, pred_ego_status)):
            bs = score.shape[0]
            num_grand = score.shape[2]
            if 'gt_ego_fut_masks' not in data:
                mask = torch.ones((bs, self.num_sample_waypoints), device=reg.device)
            else:
                mask = data['gt_ego_fut_masks']
                mask = (
                    mask.unsqueeze(1)
                    .unsqueeze(1)
                    .expand(-1, self.sampler.num_modal, score.shape[2], -1)
                )

            bs = reg.shape[0]
            reg = rearrange(
                reg,
                'b (n m) (t d) -> b n m t d',
                n=self.sampler.num_modal,
                m=num_grand,
                t=self.num_sample_waypoints,
                d=2,
            )
            selected_reg = reg[torch.arange(bs), winner_modal[decoder_idx]]
            selected_drive_path = rearrange(
                drive_path, 'b (n m) t d -> b n m t d', n=self.sampler.num_modal, m=num_grand
            )
            selected_drive_path = selected_drive_path[torch.arange(bs), winner_modal[decoder_idx]]

            augment_bbox = data['augment_bbox']
            gt_traj = data[self.longitudinal_train_use_grand['temporal']]
            gt_traj_mask = self.longitudinal_train_use_grand['temporal'].replace('_traj', '_mask')
            gt_traj_mask = data[gt_traj_mask]

            if self.augment_bbox_longitudinal:

                augmened_gt_traj = torch.where(
                    augment_bbox.unsqueeze(-1).unsqueeze(-1),
                    data[self.longitudinal_train_use_grand['augmentation']],
                    data[self.longitudinal_train_use_grand['temporal']],
                )
            else:
                augmened_gt_traj = data[self.longitudinal_train_use_grand['temporal']]

            if self.longitudinal_train_use_grand['temporal'] == 'temporal_interval_5_traj':
                gt_horizon = gt_traj.shape[1] * 0.5
            else:
                gt_horizon = gt_traj.shape[1] * 0.2

            distance_idx = self.sampler.select_distance_granularity(
                data, gt_horizon=gt_horizon, gt_trajs=augmened_gt_traj
            )
            gt_scores = torch.tensor(distance_idx, device=reg.device, dtype=torch.long)

            score = score[torch.arange(bs), winner_modal[decoder_idx]].reshape(bs, num_grand)
            score_mask = gt_traj_mask.any(-1)
            loss_score = self.plan_loss_cls(score, gt_scores, weight=score_mask)

            if self.gt_modal == 'distance' and predicted_distance[decoder_idx] is not None:
                selected_distance = predicted_distance[decoder_idx]
                selected_distance = rearrange(
                    selected_distance,
                    'b (n m) t d -> b n m t d',
                    n=self.sampler.num_modal,
                    m=num_grand,
                )
                selected_distance = selected_distance[torch.arange(bs), winner_modal[decoder_idx]]

                winner_distance = selected_distance[torch.arange(bs), gt_scores]

                augmened_gt_traj_distance = torch.norm(augmened_gt_traj, dim=-1, p=2)

                loss_reg = self.plan_loss_reg(
                    winner_distance,
                    augmened_gt_traj_distance.unsqueeze(-1),
                    weight=gt_traj_mask.unsqueeze(-1),
                )

                winner_reg = selected_reg[torch.arange(bs), gt_scores]

            else:

                winner_reg = selected_reg[torch.arange(bs), gt_scores]
                loss_reg = self.plan_loss_reg(
                    winner_reg, augmened_gt_traj, weight=gt_traj_mask.unsqueeze(-1)
                )

            longitudinal_head_scale = 1
            if self.iter_count >= self.auxiliary_loss['start_iter']:
                if self.iter_count == self.auxiliary_loss['start_iter']:
                    print("-------------START USING AUXILIARY LOSS-------------")
                    print("-------------Larger Longitudinal Loss-------------")
                    longitudinal_head_scale = self.longitudinal_head_scale

                valid_mask = torch.ones_like(loss_reg, dtype=torch.bool)
                if self.auxiliary_loss['dynamic_loss_longitudinal']:
                    agent_corer = model_outs['longitudinal_agent_corner'][-1]
                    agent_traj = model_outs['longitudinal_agent_traj'][-1]

                    if self.auxiliary_loss['dynamic_all_trajs']:
                        selected_reg = rearrange(selected_reg, 'b n t d -> (b n) t d')
                        abs_traj = torch.cumsum(selected_reg, dim=1)
                        agent_corer = agent_corer.unsqueeze(1).expand(-1, num_grand, -1, -1, -1, -1)
                        agent_traj = agent_traj.unsqueeze(1).expand(-1, num_grand, -1, -1, -1)
                        agent_corer = rearrange(agent_corer, 'b n m a t d -> (b n) m a t d')
                        agent_traj = rearrange(agent_traj, 'b n m t d -> (b n) m t d')
                        winner_reg = selected_reg
                    else:
                        abs_traj = torch.cumsum(winner_reg, dim=1)

                    dynamic_loss = torch.tensor(0.0, dtype=torch.float32, device=winner_reg.device)
                    if len(agent_corer) > 0:

                        longitudinal_displaceement = abs_traj[:, -1, 1]
                        threshold = self.auxiliary_loss['dynamic_dead_range']
                        valid_mask = (longitudinal_displaceement > threshold).detach()
                        time_horizon = self.auxiliary_loss['dynamic_horzion']
                        if valid_mask.any():
                            valid_reg = winner_reg[valid_mask][:, :time_horizon, :]
                            valid_agent_corer = agent_corer[valid_mask][:, :, : time_horizon + 1, :]
                            valid_agent_traj = agent_traj[valid_mask][:, :, : time_horizon + 1, :]

                            agent_corner_sum = torch.sum(
                                torch.abs(valid_agent_corer), dim=(-3, -2, -1)
                            )
                            agent_class = model_outs['longitudinal_agent_cls'][-1][valid_mask]
                            traffic_light_idx = [6, 7, 8]
                            valid_agent_mask = torch.ones_like(agent_corner_sum, dtype=torch.bool)

                            for idx in traffic_light_idx:
                                valid_agent_mask = valid_agent_mask & (agent_class != idx)

                            valid_agent_mask = valid_agent_mask & (agent_corner_sum > 0.1)
                            collision_weight = self._conditional_weights(
                                data, data['gt_ego_fut_cmd'], 'collision'
                            )
                            collision_weight = collision_weight[valid_mask]
                            dynamic_loss = self.collision_loss(
                                valid_reg,
                                valid_agent_corer,
                                valid_agent_traj,
                                valid_agent_mask,
                                interactive_agents=None,
                                sample_weight=collision_weight,
                            )
                            dynamic_loss = dynamic_loss * self.auxiliary_loss['dynamic_loss_scale']
                            dynamic_loss = dynamic_loss * longitudinal_head_scale

                if self.auxiliary_loss['lateral_loss_longitudinal']:

                    winner_drive_path = selected_drive_path[torch.arange(bs), gt_scores].detach()
                    projected_reg = project_stepwise_waypoints_to_drivepath(
                        winner_reg, winner_drive_path
                    )
                    projected_reg = projected_reg[:, 1:]

                    abs_winner_reg = torch.cumsum(winner_reg, dim=1)
                    abs_projected_reg = projected_reg

                    loss_lateral = torch.tensor(0.0, dtype=torch.float32, device=winner_reg.device)

                    if valid_mask.any():

                        valid_reg = abs_winner_reg[valid_mask]
                        valid_projected_reg = abs_projected_reg[valid_mask]
                        loss_lateral = valid_reg[:, :, 0] - valid_projected_reg[:, :, 0]
                        loss_lateral = torch.mean(torch.abs(loss_lateral))
                        loss_lateral = loss_lateral * self.auxiliary_loss['lateral_loss_scale']
                        loss_lateral *= longitudinal_head_scale

            loss_score *= longitudinal_head_scale
            loss_reg *= longitudinal_head_scale

            loss_dict = {}
            loss_dict[f"{pretext}_planning_loss_reg_{decoder_idx}"] = loss_reg
            loss_dict[f"{pretext}_planning_loss_score_{decoder_idx}"] = loss_score
            if (
                self.auxiliary_loss['dynamic_loss_longitudinal']
                and self.iter_count > self.auxiliary_loss['start_iter']
            ):
                loss_dict[f"{pretext}_planning_loss_dynamic_{decoder_idx}"] = dynamic_loss
            if (
                self.auxiliary_loss['lateral_loss_longitudinal']
                and self.iter_count > self.auxiliary_loss['start_iter']
            ):
                loss_dict[f"{pretext}_planning_loss_lateral_{decoder_idx}"] = loss_lateral
            output.update(loss_dict)

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss(self, model_outs, data, feature_maps=None, gan_idx=0):

        """Collect spatial and auxiliary-branch losses for all decoder layers."""
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        pred_ego_status = model_outs["pred_ego_status"]

        output, winner_modal = self.loss_reg(
            cls_scores,
            reg_preds,
            pred_ego_status,
            data=data,
            use_vocab=self.anchor_supervise,
            model_outs=model_outs,
        )
        if self.other_output_pretext is not None:
            for output_pretext in self.other_output_pretext:
                if output_pretext == 'longitudinal':
                    scores = model_outs[f"{output_pretext}_scores"]
                    reg_preds = model_outs[f"{output_pretext}_prediction"]
                    drive_path = model_outs[f"{output_pretext}_drivepath"]
                    predicted_distance = model_outs[f"{output_pretext}_predicted_distance"]
                    pred_ego_status = [None for _ in range(len(scores))]

                    longitudinal_loss = self.loss_speed_aware_trajectory(
                        scores,
                        drive_path,
                        reg_preds,
                        pred_ego_status,
                        predicted_distance,
                        data=data,
                        pretext=output_pretext,
                        use_vocab=self.anchor_supervise,
                        winner_modal=winner_modal,
                        model_outs=model_outs,
                    )
                    output.update(longitudinal_loss)

                else:
                    cls_scores = model_outs[f"{output_pretext}_classification"]
                    reg_preds = model_outs[f"{output_pretext}_prediction"]
                    pred_ego_status = model_outs[f"{output_pretext}_pred_ego_status"]

                    output.update(
                        self.loss_reg(
                            cls_scores,
                            reg_preds,
                            pred_ego_status,
                            data=data,
                            pretext=output_pretext,
                            use_vocab=self.anchor_supervise,
                        )
                    )

        self.iter_count += 1
        return output

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(end_dim=1)[dn_valid_mask]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(end_dim=1)[dn_valid_mask][
            ..., : len(self.reg_weights)
        ]
        dn_pos_mask = dn_cls_target >= 0
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)),
            1.0,
        )
        return (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        )

    def process_waypoints(self, output):

        """Integrate offsets into absolute positions, preserving batched and unbatched layouts."""
        if len(output.shape) == 3:
            output = np.cumsum(output, axis=1)
        else:
            output = np.cumsum(output, axis=0)

        return output

    @force_fp32(apply_to=("model_outs"))
    def find_winner(
        self,
        model_outs,
        classification_key,
        prediction_key,
        mod='waypoints',
        keep_mm=True,
        given_modal=None,
        data=None,
    ):
        """Select and decode the configured planning branch without changing serialized output keys."""
        bs = model_outs[classification_key].shape[0]
        output = {}
        score_name = 'mm_score'
        mm_output_name = 'mm_' + mod

        if given_modal is not None:
            pred_traj = model_outs[prediction_key]
            pred_traj = rearrange(pred_traj, 'b (n m) t d -> b n m t d', n=self.sampler.num_modal)
            reg = pred_traj[torch.arange(bs), given_modal]

            spatial_waypoints = model_outs['longitudinal_drivepath']
            spatial_waypoints = rearrange(
                spatial_waypoints, 'b (n m) t d -> b n m t d', n=self.sampler.num_modal
            )
            spatial_waypoints = spatial_waypoints[torch.arange(bs), given_modal]
            spatial_waypoints = spatial_waypoints[:, 0]

            grand_score = model_outs[classification_key][torch.arange(bs), given_modal].squeeze(-1)
            winner_grand = torch.argmax(grand_score, dim=-1)
            temporal_waypoints = reg[torch.arange(bs), winner_grand]
            perferred_anchor = self.control_setting['perferred_anchor']
            if self.post_process_score:

                (
                    temporal_waypoints,
                    winner_grand_rule,
                    rule_scores,
                    sorted_indices,
                    fastest_threshold_trajs,
                ) = self.obtain_scored_waypoints(
                    reg,
                    model_outs,
                    reference_score=grand_score,
                    perfer_anchor=perferred_anchor,
                    data=data,
                    drive_path=spatial_waypoints,
                )
            else:
                rule_scores = grand_score
                winner_grand_rule = winner_grand

                fastest_threshold_trajs = temporal_waypoints

            if self.plan_control == 'open_loop':
                output[mod] = temporal_waypoints
            else:
                meraged_waypoints = project_stepwise_waypoints_to_drivepath(
                    temporal_waypoints, spatial_waypoints
                )
                meraged_waypoints = meraged_waypoints[:, 1:] - meraged_waypoints[:, :-1]

                output[mod] = meraged_waypoints

            output[mod + '_drivepath'] = spatial_waypoints

            output[score_name] = grand_score
            output['winner_grand'] = winner_grand
            output['winner_grand_rule'] = winner_grand_rule
            output['rule_scores'] = rule_scores
            output['fastest_threshold_trajs'] = fastest_threshold_trajs
            output[mm_output_name] = reg

        else:

            pred_traj = model_outs[prediction_key]
            pred_traj = rearrange(
                pred_traj,
                'b (n m) t d -> b n m t d',
                n=self.sampler.num_modal,
                m=self.sampler.num_grand,
            )
            temporal_waypoints_index = self.sampler.grand_name.index(self.control_temporal_grand)

            grand_score = model_outs[classification_key]
            perferred_anchor = self.control_setting['perferred_anchor']
            winner_modal = torch.argmax(model_outs[classification_key], dim=-1)
            reg = pred_traj[torch.arange(bs), winner_modal]
            temporal_waypoints = reg[:, temporal_waypoints_index]
            spatial_waypoints = reg[:, self.sampler.grand_name.index(self.control_spatial_grand)]

            rule_scores = grand_score
            winner_grand_rule = winner_modal

            fastest_threshold_trajs = temporal_waypoints

            if self.plan_control == 'open_loop':
                output[mod] = reg[:, temporal_waypoints_index]
            else:
                meraged_waypoints = project_stepwise_waypoints_to_drivepath(
                    temporal_waypoints, spatial_waypoints
                )
                meraged_waypoints = meraged_waypoints[:, 1:] - meraged_waypoints[:, :-1]

                output[mod] = meraged_waypoints

            spatial_waypoints = reg[:, self.sampler.grand_name.index(self.control_spatial_grand)]
            output[mod + '_drivepath'] = spatial_waypoints

            for i in range(len(self.sampler.grand_name)):
                grand_name = self.sampler.grand_name[i]
                output[grand_name] = reg[:, i]

            output[mm_output_name] = model_outs[prediction_key]

            scores = model_outs[classification_key]
            scores = scores.unsqueeze(-1)
            scores = scores.expand(-1, -1, self.sampler.num_grand)
            scores = scores.reshape(bs, -1)

            output["winner_modal"] = winner_modal
            output['winner_grand'] = winner_modal
            output[score_name] = scores
            output['winner_grand_rule'] = winner_grand_rule
            output['rule_scores'] = rule_scores
            output['fastest_threshold_trajs'] = fastest_threshold_trajs

        decoded_output = {}
        decoded_output['winner_modal'] = (
            output['winner_modal'] if 'winner_modal' in output else None
        )
        decoded_output['winner_grand'] = (
            output['winner_grand'] if 'winner_grand' in output else None
        )
        decoded_output['winner_grand_rule'] = (
            output['winner_grand_rule'] if 'winner_grand_rule' in output else None
        )
        decoded_output['rule_scores'] = output['rule_scores'] if 'rule_scores' in output else None
        decoded_output['fastest_threshold_trajs'] = (
            output['fastest_threshold_trajs'] if 'fastest_threshold_trajs' in output else None
        )
        decoded_output['sorted_indices'] = (
            output['sorted_indices'] if 'sorted_indices' in output else None
        )

        if output is not None:
            for key in output.keys():
                if output[key] is not None:
                    output[key] = output[key].detach().cpu().numpy()

            decoded_waypoints = output[mod]
            decoded_waypoints = self.process_waypoints(decoded_waypoints)
            decoded_output.update({mod: decoded_waypoints})

            drivepath_key = mod + '_drivepath'
            if drivepath_key in output:
                drivepath = output[drivepath_key]
                drivepath = self.process_waypoints(drivepath)
                decoded_output.update({drivepath_key: drivepath})
            if 'fastest_threshold_trajs' in output:
                fastest_threshold_trajs = output['fastest_threshold_trajs']
                fastest_threshold_trajs = self.process_waypoints(fastest_threshold_trajs)
                decoded_output.update({'fastest_threshold_trajs': fastest_threshold_trajs})

            if keep_mm and mm_output_name in output:
                decoded_mm_waypoints = output[mm_output_name]
                mm_waypoints = []
                for i in range(decoded_mm_waypoints.shape[1]):
                    processed_waypoints = self.process_waypoints(decoded_mm_waypoints[:, i])
                    mm_waypoints.append(processed_waypoints)
                decoded_output[mm_output_name] = np.stack(mm_waypoints, axis=1)
                decoded_output[score_name] = output[score_name]
        return decoded_output

    @force_fp32(apply_to=("model_outs"))
    def process_outputs(
        self,
        model_outs,
        multi_stage_keys,
        output_idx=-1,
        given_modal=None,
        keep_mm=False,
        pretext=None,
        data=None,
    ):
        """Normalize one branch output and decode trajectories with an optional key prefix."""
        prediction_key = multi_stage_keys[0]
        classification_key = multi_stage_keys[1]

        for keys in model_outs.keys():
            if keys in multi_stage_keys:
                if output_idx > len(model_outs[keys]) - 1:
                    return {}
                else:
                    model_outs[keys] = model_outs[keys][output_idx]
            else:
                model_outs[keys] = model_outs[keys]

        if isinstance(model_outs[classification_key], dict):

            for key in model_outs[classification_key].keys():
                model_outs[key] = model_outs[classification_key][key]

            if 'cls' in model_outs[classification_key].keys():
                model_outs[classification_key] = model_outs[classification_key]['cls']

        if len(model_outs[classification_key].shape) == 3:
            model_outs[classification_key] = model_outs[classification_key].squeeze(-1)

        if len(model_outs[prediction_key].shape) == 3:
            model_outs[prediction_key] = model_outs[prediction_key].reshape(
                model_outs[prediction_key].shape[0], model_outs[prediction_key].shape[1], -1, 2
            )

        decoded_output = self.find_winner(
            model_outs=model_outs,
            classification_key=classification_key,
            prediction_key=prediction_key,
            mod='waypoints',
            given_modal=given_modal,
            data=data,
        )

        temporal_grand = (
            self.longitudinal_train_use_grand['temporal']
            if given_modal is not None else self.control_temporal_grand
        )
        temporal_stride = int(temporal_grand.split('_')[2])
        decoded_output['waypoints_timestep'] = np.full(
            model_outs[prediction_key].shape[0], temporal_stride / 10.0,
            dtype=np.float64,
        )
        decoded_output['waypoints_grand'] = np.full(
            model_outs[prediction_key].shape[0], temporal_grand
        )

        if 'light_state' in model_outs:

            light_state = model_outs['light_state'][-1].detach().cpu().numpy()
            decoded_output['traffic_light_state'] = light_state

        if pretext is not None:
            new_output = {}
            for key in decoded_output.keys():
                new_key = f"{pretext}_{key}"
                new_output[new_key] = decoded_output[key]
            decoded_output = new_output
        return decoded_output

    @force_fp32(apply_to=("model_outs"))
    def post_process(self, model_outs, output_idx=-1, keep_mm=False, data=None):
        """Decode planning outputs into independent per-sample dictionaries."""
        bs = model_outs['prediction'][0].shape[0]
        decoded_output = self.process_outputs(
            model_outs=model_outs,
            multi_stage_keys=['prediction', 'classification'],
            output_idx=output_idx,
            keep_mm=keep_mm,
            pretext='base',
            data=data,
        )

        if self.other_output_pretext is not None:
            for output_pretext in self.other_output_pretext:

                if output_pretext == 'longitudinal':
                    decoded_output.update(
                        self.process_outputs(
                            model_outs=model_outs,
                            multi_stage_keys=[
                                f"{output_pretext}_prediction",
                                f"{output_pretext}_scores",
                            ],
                            output_idx=output_idx,
                            keep_mm=keep_mm,
                            pretext=output_pretext,
                            given_modal=decoded_output['base_winner_modal'],
                            data=data,
                        )
                    )

                else:
                    decoded_output.update(
                        self.process_outputs(
                            model_outs=model_outs,
                            multi_stage_keys=[
                                f"{output_pretext}_prediction",
                                f"{output_pretext}_classification",
                            ],
                            output_idx=output_idx,
                            keep_mm=keep_mm,
                            pretext=output_pretext,
                        )
                    )

        new_decoded_output = {}
        for key in decoded_output.keys():
            if self.final_used_head in key:

                new_key = key.replace(self.final_used_head + '_', '')
                new_decoded_output[new_key] = decoded_output[key]
            new_decoded_output[key] = decoded_output[key]

        decoded_output = new_decoded_output
        decoded_output['planning_head'] = np.full(bs, self.final_used_head)

        if 'winner_modal' not in decoded_output.keys():
            decoded_output['winner_modal'] = decoded_output['base_winner_modal']
        elif decoded_output['winner_modal'] is None:
            decoded_output['winner_modal'] = decoded_output['base_winner_modal']

        decoded_output_list = [
            {
                key: None if value is None else value[sample_index]
                for key, value in decoded_output.items()
            }
            for sample_index in range(bs)
        ]
        return decoded_output_list, None

    def get_feature(self, feature_maps, metas):
        """Build the query and anchor state consumed by the staged decoder."""
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]
        (
            temporal_img_features,
            instance_feature,
            anchor,
            anchor_embed,
            temp_instance_feature,
            temp_anchor,
            temp_anchor_embed,
            temp_confidence,
            cmd,
            cmd_embedding,
            ego_status_embedding,
            time_interval,
        ) = self.obtain_queries(feature_maps=feature_maps, batch_size=batch_size, metas=metas)
        output = {
            "instance_feature": instance_feature,
            "anchor": anchor,
            "anchor_embed": anchor_embed,
            "temp_instance_feature": temp_instance_feature,
            "temp_anchor": temp_anchor,
            "temp_anchor_embed": temp_anchor_embed,
            "temp_confidence": temp_confidence,
            "time_interval": time_interval,
            "temporal_img_features": temporal_img_features,
            "cmd": cmd,
            "cmd_embedding": cmd_embedding,
            "ego_status_embedding": ego_status_embedding,
        }
        return output

    def forward_split(self, feature_maps, metas):

        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        (
            temporal_img_features,
            instance_feature,
            anchor,
            anchor_embed,
            temp_instance_feature,
            temp_anchor,
            temp_anchor_embed,
            cmd,
            cmd_embedding,
            ego_status_embedding,
            time_interval,
        ) = self.obtain_queries(feature_maps=feature_maps, batch_size=batch_size, metas=metas)

        attn_mask = None
        dn_metas = None
        key_points = []
        prediction = []
        classification = []
        quality = []

        for index in range(self.num_decoder):

            (
                prediction,
                classification,
                quality,
                cls,
                instance_feature,
                anchor,
                anchor_embed,
                key_points,
            ) = self.forward_one_layer(
                layer_index=index,
                instance_feature=instance_feature,
                temp_instance_feature=temp_instance_feature,
                anchor=anchor,
                anchor_embed=anchor_embed,
                temp_anchor_embed=temp_anchor_embed,
                feature_maps=feature_maps,
                temporal_img_features=temporal_img_features,
                metas=metas,
                prediction=prediction,
                classification=classification,
                quality=quality,
                time_interval=time_interval,
                dn_metas=dn_metas,
                attn_mask=attn_mask,
                cmd_embedding=cmd_embedding,
                key_points=key_points,
                batch_size=batch_size,
            )

        output = self.update_instance_bank_and_output(
            feature_maps=feature_maps,
            prediction=prediction,
            classification=classification,
            quality=quality,
            cls=cls,
            instance_feature=instance_feature,
            anchor=anchor,
            anchor_embed=anchor_embed,
            metas=metas,
            dn_metas=dn_metas,
            key_points=key_points,
        )

        return output, None

    def obtain_queries(self, feature_maps, batch_size, metas):

        """Fetch temporal queries, encode anchors, and add navigation and ego-status embeddings."""
        ego_status_embedding = None

        if self.use_ego_status:
            ego_status = metas['ego_status']

            ego_status_embedding = self.egp_status_encoder(ego_status)

        if self.temporal_queue is not None:
            timestamp = metas['timestamp']
            self.temporal_queue.check(feature_maps, timestamp)
            temporal_img_features, temporal_meta = self.temporal_queue.get_feature(
                current_feature=feature_maps, current_meta=metas, cat_features=False
            )
            self.temporal_queue.cache(feature_maps, timestamp, meta=metas)
        else:
            temporal_img_features = None
            temporal_meta = None

        self._temporal_metas = temporal_meta

        cmd = metas['gt_ego_fut_cmd'].argmax(dim=-1)
        if self.use_command:
            cmd_embedding = self.cmd_codebook(cmd).unsqueeze(1)
        else:
            cmd_embedding = None

        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            temp_confidence,
            time_interval,
        ) = self.instance_bank.get(
            batch_size, metas, dn_metas=None, force_check_timeinterval=self.force_check_timeinterval
        )

        anchor_embed = self.encode_anchor(anchor)
        if temp_anchor is not None:
            temp_anchor_embed = self.encode_anchor(temp_anchor)
        else:
            temp_anchor_embed = None

        return (
            temporal_img_features,
            instance_feature,
            anchor,
            anchor_embed,
            temp_instance_feature,
            temp_anchor,
            temp_anchor_embed,
            temp_confidence,
            cmd,
            cmd_embedding,
            ego_status_embedding,
            time_interval,
        )

    def encode_anchor(self, anchor):

        if self.anchor_encoder_low_frency is None:

            return self.anchor_encoder(anchor)

        input_size = len(anchor.shape)
        if input_size == 3:

            anchor = anchor.unsqueeze(2)

        anchor = rearrange(
            anchor,
            'b (m n) a (t d) -> b m n a t d',
            m=self.sampler.num_modal,
            n=self.sampler.num_grand,
            a=anchor.shape[2],
            d=2,
        )

        low_frency_index = torch.tensor(
            [self.sampler.grand_name.index(keys) for keys in self.low_frency_grand],
            device=anchor.device,
        )

        grand_range = torch.arange(len(self.sampler.grand_name), device=anchor.device)
        mask = torch.ones_like(grand_range, dtype=torch.bool)
        mask[low_frency_index] = False
        hign_frency_index = grand_range[mask]

        low_frency_anchor = anchor[:, :, low_frency_index, :, :6]
        low_frency_anchor = rearrange(low_frency_anchor, 'b m n a t d -> b m n a (t d)')
        low_frency_anchor_embed = self.anchor_encoder_low_frency(low_frency_anchor)

        high_frency_anchor = anchor[:, :, hign_frency_index]
        high_frency_anchor = rearrange(high_frency_anchor, 'b m n a t d -> b m n a (t d)')
        high_frency_anchor_embed = self.anchor_encoder(high_frency_anchor)

        embed_dim = high_frency_anchor_embed.size(-1)
        combined_embed = torch.zeros(
            anchor.size(0),
            anchor.size(1),
            anchor.size(2),
            anchor.size(3),
            embed_dim,
            device=anchor.device,
        )
        combined_embed[:, :, low_frency_index, :] = low_frency_anchor_embed
        combined_embed[:, :, hign_frency_index, :] = high_frency_anchor_embed

        combined_embed = rearrange(combined_embed, 'b m n a d -> b (m n) a d')
        combined_embed = combined_embed.squeeze(dim=2)
        return combined_embed

    def forward_one_layer(
        self,
        layer_index,
        instance_feature,
        temp_instance_feature=None,
        anchor=None,
        anchor_embed=None,
        temp_anchor_embed=None,
        feature_maps=None,
        temporal_img_features=None,
        metas=None,
        prediction=None,
        classification=None,
        pred_ego_status=None,
        collision_matrixs=None,
        quality=None,
        time_interval=None,
        given_keypoints=None,
        dn_metas=None,
        attn_mask=None,
        cmd_embedding=None,
        ego_status_embedding=None,
        key_points=None,
        batch_size=None,
        stage=None,
        addtion_instance_features=None,
        addtion_anchor_embedding=None,
    ):

        """Run a decoder layer in temporal-fusion or spatial-refinement mode."""
        operation_order = self.operation_order_layer[layer_index]

        if "deformable" not in operation_order:
            if "obdeformable" in operation_order:
                deformable_index = operation_order.index("obdeformable")
            else:
                deformable_index = operation_order.index("ffn")
        else:
            deformable_index = operation_order.index("deformable")
        if stage == 0:
            operation_order = operation_order[:deformable_index]
        elif stage == 1:
            operation_order = operation_order[deformable_index:]
        else:
            raise NotImplementedError(f"{stage} is not supported.")

        if key_points is None:
            key_points = self.key_points
        if prediction is None:
            prediction = self.prediction
        if classification is None:
            classification = self.classification
        if quality is None:
            quality = self.quality
        if collision_matrixs is None:
            collision_matrixs = self.collision_matrixs
        if pred_ego_status is None:
            pred_ego_status = self.pred_ego_status
        cls = None

        start_index = 0
        for j in range(layer_index):
            start_index += len(self.operation_order_layer[j])
        start_index += deformable_index * stage
        for index, op in enumerate(operation_order):
            i = start_index + index
            if self.layers[i] is None:
                continue
            if op == "temp_gnn_both":
                temp_instance_feature = (
                    temp_instance_feature.squeeze(dim=2)
                    if temp_instance_feature is not None
                    else None
                )
                temp_anchor_embed = (
                    temp_anchor_embed.squeeze(dim=2) if temp_anchor_embed is not None else None
                )
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask if temp_instance_feature is None else None,
                )
            elif op == "temp_gnn":
                num_instance = instance_feature.shape[1]
                instance_feature = instance_feature.reshape(
                    batch_size * num_instance, 1, self.embed_dims
                )
                anchor_embed = anchor_embed.reshape(batch_size * num_instance, 1, self.embed_dims)
                if temp_instance_feature is not None:
                    temp_instance_feature = temp_instance_feature.reshape(
                        batch_size * num_instance, self.num_temporal_plan, self.embed_dims
                    )
                    temp_anchor_embed = temp_anchor_embed.reshape(
                        batch_size * num_instance, self.num_temporal_plan, self.embed_dims
                    )
                if attn_mask is not None:
                    attn_mask = attn_mask.reshape(
                        batch_size * num_instance, batch_size * num_instance
                    )
                    print("UNIMPLEMENTED for attn_mask")

                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask if temp_instance_feature is None else None,
                )

                instance_feature = instance_feature.reshape(
                    batch_size, num_instance, self.embed_dims
                )
                anchor_embed = anchor_embed.reshape(batch_size, num_instance, self.embed_dims)
                if temp_instance_feature is not None:
                    temp_anchor_embed = temp_anchor_embed.reshape(
                        batch_size, num_instance, self.num_temporal_plan, self.embed_dims
                    )
                    temp_instance_feature = temp_instance_feature.reshape(
                        batch_size, num_instance, self.num_temporal_plan, self.embed_dims
                    )

            elif op == 'global_extract':
                global_img_features = self.layers[i](
                    feature_maps,
                    metas,
                )
            elif op == 'global_gnn':

                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    global_img_features,
                    global_img_features,
                )
            elif op == "add_navigation":
                if self.use_tp:

                    tp = metas['tp_line']
                    tp = tp.to(anchor_embed.dtype)
                    bs = tp.shape[0]
                    if len(tp.shape) == 2:
                        tp = tp.unsqueeze(1)
                    else:
                        tp = tp.reshape(bs, -1).unsqueeze(1)
                    tp_embedding = self.tp_encoder(tp)
                    if cmd_embedding is not None:
                        cmd_embedding = cmd_embedding + tp_embedding
                    else:
                        cmd_embedding = tp_embedding

                    self.tp_embedding = tp_embedding

                instance_feature = self.layers[i](
                    instance_feature,
                    cmd_embedding,
                )

            elif op == "gnn":
                if self.use_tp:

                    tp = metas['tp_line']
                    tp = tp.to(anchor_embed.dtype)
                    bs = tp.shape[0]
                    if len(tp.shape) == 2:
                        tp = tp.unsqueeze(1)
                    else:
                        tp = tp.reshape(bs, -1).unsqueeze(1)
                    tp_embedding = self.tp_encoder(tp)
                    if cmd_embedding is not None:
                        cmd_embedding = cmd_embedding + tp_embedding
                    else:
                        cmd_embedding = tp_embedding

                    self.tp_embedding = tp_embedding

                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    key=cmd_embedding,
                    value=cmd_embedding,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm":
                instance_feature = self.layers[i](instance_feature)
            elif op == "ffn":

                meraged_instance_feature = instance_feature

                if self.command_res:
                    meraged_instance_feature = instance_feature + cmd_embedding
                instance_feature = self.layers[i](meraged_instance_feature)

            elif op == "spdeformable":
                instance_feature, key_point, similarity = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    current_metas=metas,
                    temporal_feature_map=temporal_img_features,
                    temporal_metas=getattr(self, '_temporal_metas', None),
                )
                key_points.append(key_point)

            elif op == "deformable":

                anchor_cumsum = rearrange(anchor, "b m (t d) -> b m t d", d=2)
                anchor_cumsum = torch.cumsum(anchor_cumsum, dim=3)
                anchor_cumsum = rearrange(anchor_cumsum, "b m t d -> b m (t d)")
                instance_feature, key_point = self.layers[i](
                    instance_feature,
                    anchor_cumsum,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
                key_points.append(key_point)

            elif op == "obdeformable":
                keypoints_features, key_point = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                    given_keypoints=given_keypoints,
                )
                key_points.append(given_keypoints)
            elif op == "keypoint_cross_gnn":
                instance_feature = self.keypoint_cross_gnn(
                    instance_feature,
                    keypoints_features,
                    key_points,
                    anchor_embed,
                    feature_maps,
                    batch_size,
                    i,
                )

            elif op == "path_gnn":

                num_temporal_query = instance_feature.shape[1] // 2

                temporary_waypoints_instance_features = instance_feature[:, :num_temporal_query]
                spatial_waypoints_instance_feature = instance_feature[:, num_temporal_query:]

                temporary_waypoints_anchor_embed = anchor_embed[:, :num_temporal_query]
                spatial_waypoints_anchor_embed = anchor_embed[:, num_temporal_query:]

                temporal_waypoints_instance_features = self.graph_model(
                    i,
                    temporary_waypoints_instance_features,
                    key=spatial_waypoints_instance_feature,
                    query_pos=temporary_waypoints_anchor_embed,
                    key_pos=spatial_waypoints_anchor_embed,
                )
            elif op == "path_norm":
                temporal_waypoints_instance_features = self.layers[i](
                    temporal_waypoints_instance_features
                )

                instance_feature[:, :num_temporal_query] = temporal_waypoints_instance_features

            elif op == "refine":

                if addtion_instance_features is not None:
                    meraged_instance_feature = torch.cat(
                        [instance_feature, addtion_instance_features.clone().detach()], dim=-1
                    )
                else:
                    meraged_instance_feature = instance_feature

                if addtion_anchor_embedding is not None:
                    meraged_anchor_embedding = torch.cat(
                        [anchor_embed, addtion_anchor_embedding.clone().detach()], dim=-1
                    )
                else:
                    meraged_anchor_embedding = anchor_embed

                if self.use_ego_status:
                    meraged_instance_feature = rearrange(
                        meraged_instance_feature,
                        "b (n m) d -> b n m d",
                        n=self.sampler.num_modal,
                        m=self.sampler.num_grand,
                    )
                    meraged_instance_feature[:, :, 0:2, :] = meraged_instance_feature[
                        :, :, 0:2, :
                    ] + ego_status_embedding.unsqueeze(1).unsqueeze(2)
                    meraged_instance_feature = rearrange(
                        meraged_instance_feature, "b n m d -> b (n m) d"
                    )

                if self.tp_res:
                    meraged_instance_feature = meraged_instance_feature + self.tp_embedding

                anchor, cls, qt, plan_ego_status = self.layers[i](
                    meraged_instance_feature,
                    anchor,
                    meraged_anchor_embedding,
                    time_interval=time_interval,
                    return_cls=True,
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                pred_ego_status.append(plan_ego_status)

                collision_matrixs.append(None)

                anchor_embed = self.encode_anchor(anchor)
                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[:, : self.instance_bank.num_temp_instances]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        self.key_points = key_points
        self.prediction = prediction
        self.classification = classification
        self.quality = quality
        self.pred_ego_status = pred_ego_status
        self.collision_matrixs = collision_matrixs
        if stage == 0:
            return instance_feature
        else:
            output = {
                "classification": classification,
                "prediction": prediction,
                "anchor": anchor,
                "quality": quality,
                "pred_ego_status": pred_ego_status,
                "instance_feature": instance_feature,
                "anchor_embed": anchor_embed,
                "key_points": key_points,
                "cls": cls,
                "collision_matrixs": collision_matrixs,
            }
            return output

    def reset_output(self):
        """Clear per-forward decoder buffers without altering cached instance-bank state."""
        self.key_points = []
        self.prediction = []
        self.classification = []
        self.quality = []
        self.collision_matrixs = []
        self.pred_ego_status = []

    def keypoint_cross_gnn(
        self,
        instance_feature,
        keypoints_features,
        key_points,
        anchor_embed,
        feature_maps,
        batch_size,
        i,
    ):

        keypoints_used = key_points[-1]
        if isinstance(keypoints_used, dict):
            mearged_keypoints = []
            for key in keypoints_used.keys():
                mearged_keypoints.append(keypoints_used[key].reshape(batch_size, -1, 3))
            keypoints_used = torch.cat(mearged_keypoints, dim=1)
        else:
            keypoints_used = keypoints_used.reshape(batch_size, -1, 3)
        keypoints_posembeding = gen_sineembed_for_position(keypoints_used)
        instance_feature = self.graph_model(
            i,
            instance_feature,
            key=keypoints_features,
            query_pos=anchor_embed,
            key_pos=keypoints_posembeding,
        )
        return instance_feature

    def update_instance_bank_and_output(
        self,
        feature_maps,
        prediction,
        classification,
        quality,
        cls,
        instance_feature,
        anchor,
        anchor_embed,
        metas,
        dn_metas,
        key_points,
        pred_ego_status,
        collision_matrixs,
    ):
        output = {}

        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
                "instance_feature": instance_feature,
                "anchor_embed": anchor_embed,
                "key_points": key_points,
                "collision_matrixs": collision_matrixs,
                "pred_ego_status": pred_ego_status,
            }
        )

        if isinstance(cls, dict):
            confidence = cls['cls']
        else:
            confidence = cls

        self.instance_bank.cache(instance_feature, anchor, confidence, metas, feature_maps)
        if self.with_instance_id:
            instance_id = self.instance_bank.get_instance_id(
                cls, anchor, self.decoder.score_threshold
            )
            output["instance_id"] = instance_id

        self.prediction = prediction
        self.classification = classification
        self.quality = quality
        self.collision_matrixs = collision_matrixs
        self.pred_ego_status = pred_ego_status
        return output

    def reset_indices(self, sorted_indices, reference_score, perfer_anchor):
        bs = sorted_indices.shape[0]
        for i in range(bs):

            sorted_score = reference_score[i][sorted_indices[i]]
            suggested_anchor = sorted_indices[i][0].clone()
            perfer_anchor_score = reference_score[i][perfer_anchor]

            if perfer_anchor_score >= sorted_score[1] and sorted_indices[i][0] != 0:

                original_indices = sorted_indices[i].clone()

                sorted_indices[i][0] = perfer_anchor

                remaining_indices = original_indices[original_indices != perfer_anchor]
                if suggested_anchor in remaining_indices:
                    remaining_indices = remaining_indices[remaining_indices != suggested_anchor]
                    sorted_indices[i][1] = suggested_anchor
                    sorted_indices[i][2:] = remaining_indices
                else:

                    sorted_indices[i][1:] = remaining_indices

        return sorted_indices

    def obtain_scored_waypoints(
        self,
        candidate_trajs,
        model_outs,
        reference_score,
        rule_threshold=0.6,
        perfer_anchor=4,
        data=None,
        drive_path=None,
    ):

        """Score candidate trajectories and apply the configured safety-aware selection."""
        agents_corners = model_outs['longitudinal_agent_corner'][-1]

        agents_confidence = model_outs['longitudinal_agent_confidence'][-1]

        novalid_corners = torch.zeros_like(agents_corners) - 99
        agents_corners[agents_confidence < 0.3] = novalid_corners[agents_confidence < 0.3]

        agents_cls = model_outs['longitudinal_agent_cls'][-1]

        agents_corners[agents_cls == 4] = novalid_corners[agents_cls == 4]
        agents_corners[agents_cls == 5] = novalid_corners[agents_cls == 5]

        agents_corners[agents_cls == 6] = novalid_corners[agents_cls == 6]
        agents_corners[agents_cls == 7] = novalid_corners[agents_cls == 7]
        agents_corners[agents_cls == 8] = novalid_corners[agents_cls == 8]

        agents_corners[agents_cls == 10] = novalid_corners[agents_cls == 10]
        agents_corners[agents_cls == 11] = novalid_corners[agents_cls == 11]

        if data is not None:
            current_speed = data['ego_speed']
        else:
            current_speed = None

        if 'light_state' in model_outs:
            light_state = model_outs['light_state'][-1].item()
        else:
            light_state = 0

        if drive_path is not None:
            drive_path = torch.cumsum(drive_path, dim=1)
            is_not_turing = abs(drive_path[:, 2, 0]) < 1
        else:
            is_not_turing = False

        if light_state == 1 and not self.is_stuck:
            reference_score[is_not_turing, 0:2] += self.light_state_enhancement

        if self.collision_resscore:
            rule_score_hard, rule_score_soft, current_collision = score_prediction(
                candidate_trajs,
                agents_corners,
                current_speed=current_speed,
                light_state=light_state,
            )
            if self.is_stuck:
                rule_score = rule_score_soft
                print("[INFO] Using soft scores")
            else:
                rule_score = rule_score_hard
        else:
            rule_score_hard = torch.ones_like(reference_score)
            rule_score_soft = torch.ones_like(reference_score)
            current_collision = torch.zeros_like(reference_score[:, 0], dtype=torch.bool)
            rule_score = rule_score_hard

        (
            final_waypoints,
            winner_grand,
            fastest_threshold_trajs,
            refernce_suggestion,
            valid_output,
        ) = self.select_final_waypoints(
            candidate_trajs,
            reference_score,
            rule_score,
            rule_threshold,
            perfer_anchor=perfer_anchor,
            current_collision=current_collision,
            is_stuck=self.is_stuck,
            reset_indices_fn=self.reset_indices,
        )
        scores = rule_score

        if not valid_output[0].item():
            (
                final_waypoints,
                winner_grand,
                fastest_threshold_trajs,
                refernce_suggestion,
                valid_output,
            ) = self.select_final_waypoints(
                candidate_trajs,
                reference_score,
                rule_score_soft,
                rule_threshold,
                perfer_anchor=perfer_anchor,
                current_collision=current_collision,
                is_stuck=self.is_stuck,
                reset_indices_fn=self.reset_indices,
            )
            scores = rule_score_soft
            print("Using Soft Score")

        return final_waypoints, winner_grand, scores, refernce_suggestion, fastest_threshold_trajs

    def select_final_waypoints(
        self,
        candidate_trajs,
        reference_score,
        rule_score,
        rule_threshold,
        perfer_anchor=-1,
        current_collision=None,
        is_stuck=False,
        reset_indices_fn=None,
    ):
        """Select a final trajectory using reference and rule-based scores.

        Args:
            candidate_trajs (Tensor): Candidate trajectories, shape [B, N, T, 2].
            reference_score (Tensor): Reference scores, shape [B, N].
            rule_score (Tensor): Rule-based scores, shape [B, N].
            rule_threshold (float): Minimum acceptable rule-based score.
            perfer_anchor (int): Preferred mode index; -1 disables preference.
            current_collision (Tensor or None): Current collision flags, shape [B].
            is_stuck (bool): Whether the ego vehicle is stuck.
            reset_indices_fn (callable or None): Optional candidate reordering
                function accepting sorted_indices, reference_score, perfer_anchor.

        Returns:
            tuple: Selected waypoints [B, T, 2], winning indices [B], and
            fastest trajectories satisfying the threshold [B, T, 2].
        """

        bs, num_modal = candidate_trajs.shape[0], candidate_trajs.shape[1]

        sorted_indices = torch.argsort(reference_score, dim=-1, descending=True)
        refernce_suggestion = sorted_indices.clone()

        if perfer_anchor != -1 and reset_indices_fn is not None:
            sorted_indices = reset_indices_fn(sorted_indices, reference_score, perfer_anchor)

        final_waypoints = [None] * bs
        winner_grand = [None] * bs
        valid_output = torch.ones(bs, device=candidate_trajs.device, dtype=torch.bool)

        for i in range(bs):
            sorted_score = rule_score[i][sorted_indices[i]]
            sorted_trajs = candidate_trajs[i][sorted_indices[i]]

            for j in range(sorted_score.shape[0]):
                if sorted_score[j] >= rule_threshold:
                    final_waypoints[i] = sorted_trajs[j]
                    winner_grand[i] = sorted_indices[i][j]
                    break

            if final_waypoints[i] is None:

                valid_output[i] = False
                sorted_indices[i] = torch.argsort(rule_score[i], dim=-1, descending=True)
                sorted_score = rule_score[i][sorted_indices[i]]
                sorted_trajs = candidate_trajs[i][sorted_indices[i]]

                if sorted_score[0] > 0:
                    final_waypoints[i] = sorted_trajs[0]
                    winner_grand[i] = sorted_indices[i][0]
                elif is_stuck and current_collision is not None and current_collision[i]:
                    sorted_indices[i] = torch.argsort(reference_score[i], dim=-1, descending=True)
                    sorted_trajs = candidate_trajs[i][sorted_indices[i]]
                    final_waypoints[i] = sorted_trajs[0]
                    winner_grand[i] = sorted_indices[i][0]
                    print("Current_Collision")
                else:
                    final_waypoints[i] = candidate_trajs[i][0]
                    winner_grand[i] = torch.tensor(
                        0, device=candidate_trajs.device, dtype=sorted_indices.dtype
                    )

        fastest_threshold_trajs = torch.zeros_like(candidate_trajs[:, 0])
        for i in range(bs):
            for j in range(num_modal):
                inverse_j = num_modal - j - 1
                if rule_score[i][inverse_j] > rule_threshold:
                    fastest_threshold_trajs[i] = candidate_trajs[i][inverse_j]
                    break

        final_waypoints = torch.stack(final_waypoints, dim=0)
        winner_grand = torch.stack(winner_grand, dim=0)

        return (
            final_waypoints,
            winner_grand,
            fastest_threshold_trajs,
            refernce_suggestion,
            valid_output,
        )
