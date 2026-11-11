from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import math
from einops import rearrange

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
from mmdet.models import HEADS, LOSSES
from mmdet.core import reduce_mean

from ..blocks import DeformableFeatureAggregation as DFG
from ..instance_bank import topk

__all__ = ["Sparse4DHead"]


@HEADS.register_module()
class Sparse4DHead(BaseModule):
    def __init__(
        self,
        instance_bank: dict,
        anchor_encoder: dict,
        graph_model: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        temp_graph_model: dict = None,
        loss_cls: dict = None,
        loss_reg: dict = None,
        motion_loss_reg: dict = None,
        motion_loss_cls: dict = None,
        motion_sampler: dict = None,
        trafficstate_sampler: dict = None,
        decoder: dict = None,
        sampler: dict = None,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        gt_id_key: str = "instance_id",
        with_instance_id: bool = True,
        task_prefix: str = 'det',
        reg_weights: List = None,
        class_weight: List = None,
        operation_order: Optional[List[str]] = None,
        cls_threshold_to_reg: float = -1,
        dn_loss_weight: float = 5.0,
        decouple_attn: bool = True,
        init_cfg: dict = None,
        traj_head: dict = None,
        **kwargs,
    ):
        super(Sparse4DHead, self).__init__(init_cfg)
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
        self.class_weight = class_weight
        self.iter_count = 0 
        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights


        # operation_order_layer must be List[List[str]] (one sublist per decoder).
        # Flat List[str] configs (e.g. stage1_align) are split on "refine".
        if operation_order is None:
            layer_template = [
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
            ]
            self.operation_order_layer = [
                layer_template.copy() for _ in range(num_decoder)
            ]
            # delete the 'gnn' and 'norm' layers in the first transformer blocks
            self.operation_order_layer[0] = self.operation_order_layer[0][3:]
        elif isinstance(operation_order[0], (list, tuple)):
            self.operation_order_layer = [list(layer) for layer in operation_order]
        else:
            # Flat list of op names → chunk by refine boundaries
            layers, cur = [], []
            for op in operation_order:
                cur.append(op)
                if op == "refine":
                    layers.append(cur)
                    cur = []
            if cur:
                layers.append(cur)
            self.operation_order_layer = layers
        self.operation_order = [
            op for layer in self.operation_order_layer for op in layer
        ]

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.instance_bank = build(instance_bank, PLUGIN_LAYERS)
        self.anchor_encoder = build(anchor_encoder, POSITIONAL_ENCODING)
        self.sampler = build(sampler, BBOX_SAMPLERS)
        self.decoder = build(decoder, BBOX_CODERS)
        self.loss_cls = build(loss_cls, LOSSES)
        self.loss_reg = build(loss_reg, LOSSES)
        if motion_loss_reg is not None:
            self.motion_loss_reg = build(motion_loss_reg, LOSSES)
            self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS)
        else:
            self.motion_loss_reg = None
            self.motion_sampler = None
        
        if trafficstate_sampler is not None:
            self.trafficstate_sampler = build(trafficstate_sampler, BBOX_SAMPLERS)
        else:
            self.trafficstate_sampler = None


        if motion_loss_cls is not None:
            self.motion_loss_cls = build(motion_loss_cls, LOSSES)



        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "deformable": [deformable_model, ATTENTION],
            "refine": [refine_layer, PLUGIN_LAYERS],
            "motion": [traj_head, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = self.instance_bank.embed_dims
        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        self.prediction = []
        self.classification = []
        self.quality = []
        self.traj = []
        self.key_points = []
        self.traj_cls = []
        self.agents_drivepath = []
        self.traffic_state = []


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
    ):
        output = self.forward_split(
            feature_maps=feature_maps,
            metas=metas,
        )
        return output



    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data, motion_loss_cache):
        cls_scores = model_outs["traj_cls"]
        cls_predicted = True
        if len(cls_scores) == 0:
            cls_scores = model_outs["classification"]   # 借用检测的结果
            cls_predicted = False

        reg_preds = model_outs["traj"]
        output = {}
        for decoder_idx, (cls, reg) in enumerate(
            zip(cls_scores, reg_preds)
        ):
            if len(reg.shape) < 5:          # B, N, M, T, C
                reg = reg.unsqueeze(-3) 
            (
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
                num_pos
            ) = self.motion_sampler.sample(
                reg,  # 创造一个多模态维度
                data["gt_agent_fut_trajs"],
                data["gt_agent_fut_masks"],
                motion_loss_cache,
            )
            num_pos = max(reduce_mean(num_pos), 1.0)

            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            if cls_predicted:
                cls = cls.squeeze(-1)
                cls = cls.flatten(end_dim=1)
                cls_loss = self.motion_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(
                reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos
            )

            if cls_predicted:
                if decoder_idx == 0:
                    cls_loss = cls_loss * 1e-6 # no information for the first decoder to predict
                    reg_loss = reg_loss * 1e-6 # no information for the first decoder to predict


                output.update(
                    {
                        f"motion_loss_cls_{decoder_idx}": cls_loss,
                        f"motion_loss_reg_{decoder_idx}": reg_loss,
                    }
                )
            else:
                if decoder_idx == 0:
                    reg_loss = reg_loss * 1e-6
                    
                output.update(
                    {
                        f"motion_loss_reg_{decoder_idx}": reg_loss,
                    }
                )

        return output



    @force_fp32(apply_to=("model_outs"))
    def loss_dp(self, model_outs, data, motion_loss_cache):
        # 监督模型预测的DrivePath
        # cls_scores = model_outs["traj_cls"]
        cls_predicted = False
        cls_scores = model_outs["classification"]   #@ 借用检测的结果，占位
        # if len(cls_scores) == 0:
        #     cls_scores = model_outs["classification"]   # 借用检测的结果
        #     cls_predicted = False

        reg_preds = model_outs["agents_drivepath"]
        output = {}
        for decoder_idx, (cls, reg) in enumerate(
            zip(cls_scores, reg_preds)
        ):
            if len(reg.shape) < 5:          # B, N, M, T, C
                reg = reg.unsqueeze(-3) 
            (
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
                num_pos
            ) = self.motion_sampler.sample(
                reg,  # 创造一个多模态维度
                data["agents_spatial_interval_5_traj"],
                data["agents_spatial_interval_5_mask"],
                motion_loss_cache,
            )
            num_pos = max(reduce_mean(num_pos), 1.0)

            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            if cls_predicted:
                cls = cls.squeeze(-1)
                cls = cls.flatten(end_dim=1)
                cls_loss = self.motion_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(
                reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos
            )

            if cls_predicted:
                output.update(
                    {
                        f"motion_dp_loss_cls_{decoder_idx}": cls_loss,
                        f"motion_dp_loss_reg_{decoder_idx}": reg_loss,
                    }
                )
            else:
                output.update(
                    {
                        f"motion_dp_loss_reg_{decoder_idx}": reg_loss,
                    }
                )

        return output







    @force_fp32(apply_to=("model_outs"))
    def loss_traffic_state(self, model_outs, data, motion_loss_cache):
        # 监督模型预测的TrafficStates, 也就是该交通灯是否影响自车
        cls_predicted = False
        traffic_state_pred = model_outs['traffic_state']     # [B,900,1]
        traffic_state_gt = data['traffic_light_state']  # [B,900, 1]

        output = {}
        for decoder_idx in range(len(traffic_state_pred)):
            state_pred = traffic_state_pred[decoder_idx]  # [B,900,1]
            (
                state_traget, 
                valid_mask,
                num_pos
             ) = self.trafficstate_sampler.sample(
                state_pred,
                traffic_state_gt,
                motion_loss_cache,
            )
            state_pred = state_pred.squeeze(-1)

            if torch.sum(valid_mask) == 0:
                state_loss = self.loss_cls(
                    state_pred.flatten(end_dim=1),
                    state_traget.flatten(end_dim=1),
                    avg_factor=max(reduce_mean(num_pos), 1.0),
                )
                state_loss = state_loss * 1e-6 
                # state_loss = state_pred.sum() * 0.0
            else:
                state_loss = self.loss_cls(
                    state_pred[valid_mask],
                    state_traget[valid_mask],
                    avg_factor=max(reduce_mean(num_pos), 1.0),
                )

            output.update(
                {
                    f"traffic_light_state_{decoder_idx}": state_loss,
                }
            )



            # if cls_predicted:
            #     output.update(
            #         {
            #             f"motion_dp_loss_cls_{decoder_idx}": cls_loss,
            #             f"motion_dp_loss_reg_{decoder_idx}": reg_loss,
            #         }
            #     )
            # else:
                # output.update(
                #     {
                #         f"motion_dp_loss_reg_{decoder_idx}": reg_loss,
                #     }
                # )

        return output










    @force_fp32(apply_to=("model_outs"))
    def loss(self, model_outs, data, feature_maps=None):
        # ===================== prediction losses ======================
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        output = {}
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):
            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls,
                reg,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            reg_target_full = reg_target.clone()
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask_valid = mask.clone()

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = None
            if self.class_weight is not None:
                class_weight = cls.new_tensor(self.class_weight)
                cls_weight = class_weight[cls_target.clamp(min=0, max=len(class_weight) - 1)]
                cls_weight = torch.where(
                    cls_target >= len(class_weight),
                    cls_weight.new_ones(cls_weight.shape),
                    cls_weight,
                )
            cls_loss = self.loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            
            if mask.shape[1] == 900: # 很不鲁棒，用于判断是否是BBox的预测，还是Map的预测

                mask = mask.reshape(-1)
                bs = reg.shape[0]
                reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
                reg_target = reg_target.flatten(end_dim=1)[mask]
                reg = reg.flatten(end_dim=1)[mask]
                reg_weights = reg_weights.flatten(end_dim=1)[mask]
                reg_target = torch.where(
                    reg_target.isnan(), reg.new_tensor(0.0), reg_target
                )
                cls_target = cls_target[mask]
                if qt is not None:
                    qt = qt.flatten(end_dim=1)[mask]
            
            else:
                mask = mask.reshape(-1)
                bs = reg.shape[0]
                reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
                reg_target = reg_target.flatten(end_dim=1)
                reg = reg.flatten(end_dim=1)
                reg_weights = reg_weights.flatten(end_dim=1)
                reg_target = torch.where(
                    reg_target.isnan(), reg.new_tensor(0.0), reg_target
                )
                cls_target = cls_target
                if qt is not None:
                    qt = qt.flatten(end_dim=1)

            gt_command = data['gt_ego_fut_cmd']

            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                prefix=f"{self.task_prefix}_",
                suffix=f"_{decoder_idx}",
                quality=qt,
                cls_target=cls_target,
                command=gt_command,
                bs = bs,
                mask = mask,
                iter_count = self.iter_count,
            )

            output[f"{self.task_prefix}_loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)

        if self.motion_loss_reg is not None:
            motion_loss_cache = dict(
                indices=self.sampler.indices, 
            )
            motion_loss = self.loss_motion(model_outs, data, motion_loss_cache)
            output.update(motion_loss)

        if 'agents_drivepath' in model_outs:
            if len(model_outs['agents_drivepath']) > 0:
                dp_loss_cache = dict(
                    indices=self.sampler.indices, 
                )
                dp_loss = self.loss_dp(model_outs, data, motion_loss_cache)
                output.update(dp_loss)

        if 'traffic_state' in model_outs:
            if len(model_outs['traffic_state']) > 0:
                # 只对选择出来的交通灯进行监督
                indices_cache = dict(
                    indices=self.sampler.indices, 
                )
                traffic_state_loss = self.loss_traffic_state(model_outs, data, indices_cache)
                output.update(traffic_state_loss)
            




        self.iter_count += 1


        if "dn_prediction" not in model_outs:
            return output

        # ===================== denoising losses ======================
        dn_cls_scores = model_outs["dn_classification"]
        dn_reg_preds = model_outs["dn_prediction"]

        (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        ) = self.prepare_for_dn_loss(model_outs)
        for decoder_idx, (cls, reg) in enumerate(
            zip(dn_cls_scores, dn_reg_preds)
        ):
            if (
                "temp_dn_valid_mask" in model_outs
                and decoder_idx == self.num_single_frame_decoder
            ):
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")

            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
            reg_loss = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                    ..., : len(self.reg_weights)
                ],
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
                prefix=f"{self.task_prefix}_",
                suffix=f"_dn_{decoder_idx}",
            )
            output[f"{self.task_prefix}_loss_cls_dn_{decoder_idx}"] = cls_loss
            output.update(reg_loss)


        return output

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(
            end_dim=1
        )[dn_valid_mask]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(
            end_dim=1
        )[dn_valid_mask][..., : len(self.reg_weights)]
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

    @force_fp32(apply_to=("model_outs"))
    def post_process(self, model_outs, output_idx=-1):
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("instance_id"),
            model_outs.get("quality"),
            output_idx=output_idx,
            traj_pred=model_outs.get("traj"),
            traj_cls=model_outs.get("traj_cls"),
            # traj_pred_
        )






    # --------prepaer_for_unified_model-------------# 

    def forward_split(self, feature_maps, metas):
        # 1. get query
        # 2. prepare anchors
        # 3. forward layers
        # 4. update instance bank and output
        # 5. return output
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        # obtain_querys / InstanceBank.get return 8-tuple (post-497b118 queue API);
        # forward_split previously unpacked 5 and crashed.
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            temp_confidence,
            queue_instance_feature,
            queue_anchor,
            time_interval,
        ) = self.obtain_querys(batch_size=batch_size, metas=metas)
        (
            attn_mask,
            dn_metas,
            temp_dn_reg_target,
            anchor_embed,
            temp_anchor_embed,
            queue_anchor_embed,
        ) = self.prepare_anchors(
            instance_feature=instance_feature,
            anchor=anchor,
            temp_instance_feature=temp_instance_feature,
            temp_anchor=temp_anchor,
            queue_anchor=queue_anchor,
            time_interval=time_interval,
            metas=metas,
        )
        if dn_metas is not None:
            print("WARNING--------------------, dn training is not Finsined")

        # Must clear ALL output buffers each iter. Previously only
        # prediction/classification/quality were reset; key_points (and
        # traj*) kept appending across iters → linear CUDA memory growth
        # (~12MB/iter at bs=4) and OOM on L20 after ~2k iters.
        self.reset_output()

        prediction = self.prediction
        classification = self.classification
        quality = self.quality
        cls = None

        for index in range(self.num_decoder):
            # forward one layer
            prediction, classification, quality, cls, instance_feature, anchor, anchor_embed = self.forward_one_layer(
                layer_index=index,
                instance_feature=instance_feature,
                temp_instance_feature=temp_instance_feature,
                temp_anchor_embed=temp_anchor_embed,
                anchor=anchor,
                anchor_embed=anchor_embed,
                feature_maps=feature_maps,
                metas=metas,
                prediction=prediction,
                classification=classification,
                quality=quality,
                time_interval=time_interval,
                dn_metas=dn_metas,
                attn_mask=attn_mask,
            )

            self.prediction = prediction
            self.classification = classification
            self.quality = quality

                


        # update instance bank
        output = self.update_instance_bank_and_output(
            feature_maps=feature_maps,
            prediction=prediction,
            classification=classification,
            quality=quality,
            cls=cls,
            instance_feature=instance_feature,
            anchor=anchor,
            anchor_embed=anchor_embed,
            key_points=None,
            metas=metas,
            dn_metas=dn_metas,
            traj=None,
            traj_cls=None,
            agents_drivepath=None,
            traffic_state=None,
        )

        return output





    def reset_output(self):
        self.prediction = []
        self.classification = []
        self.quality = []
        self.traj = []
        self.key_points = []
        self.traj_cls = []
        self.agents_drivepath = []
        self.traffic_state = []


    def get_feature(self, feature_maps, metas):
        # 1. get query
        # 2. prepare anchors
        # 3. forward layers
        # 4. update instance bank and output
        # 5. return output
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        instance_feature, anchor, temp_instance_feature, temp_anchor, temp_confidence, queue_instance_feature, queue_anchor, time_interval = self.obtain_querys(batch_size=batch_size, metas=metas)




        attn_mask, dn_metas, temp_dn_reg_target, anchor_embed, temp_anchor_embed, queue_anchor_embed = self.prepare_anchors(
            instance_feature=instance_feature, anchor=anchor, temp_instance_feature=temp_instance_feature, temp_anchor=temp_anchor, queue_anchor=queue_anchor, time_interval=time_interval, metas=metas
        )
        if dn_metas is not None:
            print("WARNING--------------------, dn training is not Finsined")
        

        output = {
            "instance_feature": instance_feature,
            "anchor": anchor,
            "anchor_embed": anchor_embed,
            "temp_instance_feature": temp_instance_feature,
            "temp_anchor": temp_anchor,
            "temp_anchor_embed": temp_anchor_embed,
            "temp_confidence": temp_confidence,
            "queue_instance_feature": queue_instance_feature,
            "queue_anchor": queue_anchor,
            "queue_anchor_embed": queue_anchor_embed,
            "time_interval": time_interval,
        }

        return output







    def obtain_querys(self, batch_size, metas):

        # 获取instance_feautures
        if (
            self.sampler.dn_metas is not None
            and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
            self.sampler.dn_metas = None
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            temp_confidence,
            queue_instance_feature,
            queue_anchor,
            time_interval,
        ) = self.instance_bank.get(
            batch_size, metas, dn_metas=self.sampler.dn_metas
        )

        # if temp_instance_feature is not None:
        #     assert temp_instance_feature.shape[1] == temp_confidence.shape[1]

        return instance_feature, anchor, temp_instance_feature, temp_anchor, temp_confidence, queue_instance_feature, queue_anchor, time_interval


    def prepare_anchors(self, instance_feature, anchor, temp_instance_feature, temp_anchor, queue_anchor, time_interval, metas):
        attn_mask = None
        dn_metas = None
        temp_dn_reg_target = None
        if self.training and hasattr(self.sampler, "get_dn_anchors"):
            if self.gt_id_key in metas["img_metas"][0]:
                gt_instance_id = [
                    torch.from_numpy(x[self.gt_id_key]).cuda()
                    for x in metas["img_metas"]
                ]
            else:
                gt_instance_id = None
            dn_metas = self.sampler.get_dn_anchors(
                metas[self.gt_cls_key],
                metas[self.gt_reg_key],
                gt_instance_id,
            )
        if dn_metas is not None:
            (
                dn_anchor,
                dn_reg_target,
                dn_cls_target,
                dn_attn_mask,
                valid_mask,
                dn_id_target,
            ) = dn_metas
            num_dn_anchor = dn_anchor.shape[1]
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_anchor, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            instance_feature = torch.cat(
                [
                    instance_feature,
                    instance_feature.new_zeros(
                        batch_size, num_dn_anchor, instance_feature.shape[-1]
                    ),
                ],
                dim=1,
            )
            num_instance = instance_feature.shape[1]
            num_free_instance = num_instance - num_dn_anchor
            attn_mask = anchor.new_ones(
                (num_instance, num_instance), dtype=torch.bool
            )
            attn_mask[:num_free_instance, :num_free_instance] = False
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask

        anchor_embed = self.anchor_encoder(anchor)
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None
        if queue_anchor is not None: # B, N, T, C
            num_temporal = queue_anchor.shape[2]
            bs = queue_anchor.shape[0]
            queue_anchor = rearrange(queue_anchor, 'b n t c -> b (n t) c')
            queue_anchor_embed = self.anchor_encoder(queue_anchor)
            queue_anchor_embed = rearrange(queue_anchor_embed, 'b (n t) c -> b n t c', b=bs, t=num_temporal)
        else:
            queue_anchor_embed = None


        return attn_mask, dn_metas, temp_dn_reg_target, anchor_embed, temp_anchor_embed, queue_anchor_embed



    def forward_one_layer(
        self,
        layer_index,
        instance_feature,
        temp_instance_feature=None,
        temp_anchor_embed=None,
        anchor=None,
        anchor_embed=None,
        feature_maps=None,
        metas=None,
        prediction = None,
        classification = None,
        quality = None,
        key_points=None,
        traj = None,
        traj_cls = None,
        agents_drivepath = None,
        traffic_state=None,
        time_interval=None,
        dn_metas=None,
        attn_mask=None,
        stage=None,
    ):
        if prediction is None:
            prediction = self.prediction
        if classification is None:
            classification = self.classification
        if quality is None:
            quality = self.quality
        if traj is None:
            traj = self.traj
        if traj_cls is None:
            traj_cls = self.traj_cls
        if agents_drivepath is None:
            agents_drivepath = self.agents_drivepath
        if traffic_state is None:
            traffic_state = self.traffic_state
        if key_points is None:
            key_points = self.key_points

        cls = None

        # =================== forward the layers ====================
        operation_order = self.operation_order_layer[layer_index]
        # 计算当前的i在全局的下标
        deformable_index = operation_order.index("deformable")
        if stage == 0:
            operation_order = operation_order[:deformable_index]
        elif stage == 1:
            operation_order = operation_order[deformable_index:]
        elif stage is None:
            pass  # full layer for AlignDriveHead / 497b118-style
        else:
            raise NotImplementedError(f"{stage} is not supported.")
        start_index = 0
        for j in range(layer_index):
            start_index += len(self.operation_order_layer[j])

        if stage == 1:
            start_index += deformable_index

        for index, op in enumerate(operation_order):
            i = start_index + index
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask
                    if temp_instance_feature is None
                    else None,
                )
            elif op == "gnn":
                instance_feature = self.graph_model(        # mutil-head attention
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                instance_feature, key_point = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
                key_points.append(key_point)
            elif op == "refine":
                anchor, cls, qt, motion, motion_cls, drivepath, traffcstate  = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=True,
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                if motion is not None:
                    traj.append(motion) # B, N, T, C    # 2,900,6,2
                if motion_cls is not None:
                    traj_cls.append(motion_cls) # B, N, T, C
                if drivepath is not None:
                    agents_drivepath.append(drivepath)
                if traffcstate is not None:
                    traffic_state.append(traffcstate)

                if len(prediction) == self.num_single_frame_decoder:
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor, cls
                    )
                    if (
                        dn_metas is not None
                        and self.sampler.num_temp_dn_groups > 0
                        and dn_id_target is not None
                    ):
                        (
                            instance_feature,
                            anchor,
                            temp_dn_reg_target,
                            temp_dn_cls_target,
                            temp_valid_mask,
                            dn_id_target,
                        ) = self.sampler.update_dn(
                            instance_feature,
                            anchor,
                            dn_reg_target,
                            dn_cls_target,
                            valid_mask,
                            dn_id_target,
                            self.instance_bank.num_anchor,
                            self.instance_bank.mask,
                        )
                anchor_embed = self.anchor_encoder(anchor)
                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
            # elif op == "motion":
            #     det_anchors = prediction[-1]
            #     det_classification = classification[-1]
            #     motion, traj_class = self.layers[i](det_classification, det_anchors,anchor_embed, instance_feature)
            #     traj.append(motion)
            #     traj_cls.append(traj_class)

            else:
                raise NotImplementedError(f"{op} is not supported.")

        self.prediction = prediction
        self.classification = classification
        self.quality = quality
        self.traj = traj
        self.traj_cls = traj_cls
        self.agents_drivepath = agents_drivepath
        self.traffic_state = traffic_state
        self.key_points = key_points


        if stage == 0:
            return instance_feature
        elif stage is None:
            return (
                prediction,
                classification,
                quality,
                cls,
                instance_feature,
                anchor,
                anchor_embed,
            )
        else:
            output = {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
                "traj": traj,
                "traj_cls": traj_cls,
                "agents_drivepath": agents_drivepath,
                "traffic_state": traffic_state,
                "instance_feature": instance_feature,
                "anchor_embed": anchor_embed,
                "key_points": key_points,
                "anchor": anchor,
                "cls": cls,
            }
            return output


    def find_front_anchors(self, anchors, num_objects, confidence):
        """
        过滤anchors并调整数量, 保留前方的anchor(x坐标大于0的点超过50%), 
        并根据confidence调整到num_objects指定的数量
        
        参数:
            anchors: torch.Tensor, 形状为 (B, N, num_pts, 2)
            num_objects: int, 每个batch最终需要保留的anchor数量
            confidence: torch.Tensor, 形状为 (B, N), 每个anchor的置信度
            
        返回:
            front_anchors: torch.Tensor, 形状为 (B, num_objects, num_pts, 2)
            front_indices: List[List[int]], 每个batch中保留的anchor索引
        """
        B, N, num_pts, _ = anchors.shape
        
        # 1. 判断前方anchor (x坐标大于0的点超过50%)
        x_coords = anchors[..., 1]  # 应该是第0维，不是第1维
        is_front = x_coords > 0
        num_front_points = torch.sum(is_front, dim=-1)
        keep_mask = num_front_points > (num_pts * 0.5)
        
        # 2. 对每个batch处理
        front_anchors_list = []
        front_indices = []
        
        for b in range(B):
            # 获取当前batch的前方anchor索引
            batch_keep_indices = torch.where(keep_mask[b])[0]
            
            # 如果没有满足条件的anchor，选择confidence最高的num_objects个
            if len(batch_keep_indices) == 0:
                _, topk_indices = torch.topk(confidence[b], k=num_objects)
                batch_keep_indices = topk_indices
            # 如果数量多于num_objects，按confidence保留前num_objects个
            elif len(batch_keep_indices) > num_objects:
                batch_conf = confidence[b][batch_keep_indices]
                _, topk_indices = torch.topk(batch_conf, k=num_objects)
                batch_keep_indices = batch_keep_indices[topk_indices]
            # 如果数量不足num_objects，用confidence最高的补充
            elif len(batch_keep_indices) < num_objects:
                # 先保留所有符合条件的
                selected = batch_keep_indices.tolist()
                remaining = num_objects - len(selected)
                
                # 获取不在当前选择中的其他anchor
                all_indices = torch.arange(N, device=anchors.device)
                other_indices = all_indices[~torch.isin(all_indices, batch_keep_indices)]
                
                # 按confidence从其他anchor中选择补充
                if len(other_indices) > 0:
                    other_conf = confidence[b][other_indices]
                    _, topk_indices = torch.topk(other_conf, k=min(remaining, len(other_indices)))
                    selected.extend(other_indices[topk_indices].tolist())
                
                batch_keep_indices = torch.tensor(selected, device=anchors.device)
            
            # 确保数量正确
            batch_keep_indices = batch_keep_indices[:num_objects]
            
            # 保存结果
            front_indices.append(batch_keep_indices.tolist())
            front_anchors_list.append(anchors[b, batch_keep_indices])
        
        # 确保每个batch都有num_objects个anchor
        front_anchors = torch.stack(front_anchors_list)
        
        return front_anchors, front_indices

    def obtain_key_objects(self, det_output, num_objects, use_anchor=False):
        # =========== det/map feature/anchor ===========
        instance_feature = det_output["instance_feature"]
        anchor_embed = det_output["anchor_embed"]
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values

        if use_anchor:
            bs = det_anchors.shape[0]
            num_anchors = det_anchors.shape[1]
            key_points = det_output["prediction"][-1].reshape(bs, num_anchors, -1, 2)
            num_coords = 2
            keypoints_selected, front_indices = self.find_front_anchors(key_points, num_objects, det_confidence)
        else:
            key_points = det_output["key_points"][-1]
            num_coords = 3

            _, (instance_feature_selected, keypoints_selected) = topk(
                det_confidence, num_objects, instance_feature, key_points
            )
            keypoints_selected = keypoints_selected.reshape(
                keypoints_selected.shape[0], num_objects, -1, num_coords
            )

        if num_coords == 2:
            # 拼接一个固定高度
            height_selected = keypoints_selected.new_zeros(
                keypoints_selected.shape[0], keypoints_selected.shape[1], keypoints_selected.shape[2], 1
            )
            height_selected = height_selected - 1.84 # 雷达高度
            keypoints_selected = torch.cat(
                [
                    keypoints_selected,
                    height_selected,
                ],
                dim=-1,
            )

        return keypoints_selected





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
        key_points=None,
        traj=None,
        traj_cls=None,
        agents_drivepath=None,
        traffic_state=None,
    ):
        output = {}
        # split predictions of learnable instances and noisy instances
        if dn_metas is not None:
            num_free_instance = 100 # 乱给的, 不给的话下面全是未定义提醒
            dn_classification = [
                x[:, num_free_instance:] for x in classification
            ]
            classification = [x[:, :num_free_instance] for x in classification]
            dn_prediction = [x[:, num_free_instance:] for x in prediction]
            prediction = [x[:, :num_free_instance] for x in prediction]
            quality = [
                x[:, :num_free_instance] if x is not None else None
                for x in quality
            ]
            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask,
                        "dn_id_target": dn_id_target,
                    }
                )
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask
            dn_instance_feature = instance_feature[:, num_free_instance:]
            dn_anchor = anchor[:, num_free_instance:]
            instance_feature = instance_feature[:, :num_free_instance]
            anchor_embed = anchor_embed[:, :num_free_instance]
            anchor = anchor[:, :num_free_instance]
            cls = cls[:, :num_free_instance]

            # cache dn_metas for temporal denoising
            self.sampler.cache_dn(
                dn_instance_feature,
                dn_anchor,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )
        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
                "instance_feature": instance_feature,
                "anchor_embed": anchor_embed,
            }
        )
        if traj is not None:
            output["traj"] = traj
        if traj_cls is not None:
            output["traj_cls"] = traj_cls
        if agents_drivepath is not None:
            output["agents_drivepath"] = agents_drivepath
        if traffic_state is not None:
            output["traffic_state"] = traffic_state
        if key_points is not None:
            output["key_points"] = key_points

        # cache current instances for temporal modeling

        self.instance_bank.cache(
            instance_feature, anchor, cls, metas, feature_maps
        )
        if self.with_instance_id:
            instance_id = self.instance_bank.get_instance_id(
                cls, anchor, self.decoder.score_threshold
            )
            output["instance_id"] = instance_id
        else:
            instance_id = None


        self.prediction = prediction
        self.classification = classification
        self.quality = quality
        if hasattr(self, "traj"):
            self.traj = traj
        if hasattr(self, "traj_cls"):
            self.traj_cls = traj_cls
        if hasattr(self, "agents_drivepath"):
            self.agents_drivepath = agents_drivepath
        if hasattr(self, "traffic_state"):
            self.traffic_state = traffic_state
        return output
