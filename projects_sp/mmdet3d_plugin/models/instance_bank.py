"""Instance banks for temporal detection/planning caches.

InstanceBank.get() returns a fixed 8-tuple:
(instance_feature, anchor, cached_feature, cached_anchor, confidence,
 temporal_instance_feature, temporal_anchor, time_interval).
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
import pickle
from projects_sp.mmdet3d_plugin.datasets.utils import obtain_traj_given_speed

from .attention import get_1d_sincos_pos_embed_from_grid, gen_sineembed_for_position
__all__ = ["InstanceBank", "InstanceBank_HiP_AD"]


def _stack_plan_anchors(all_anchors):
    if not all_anchors:
        raise ValueError("plan anchor list must not be empty")
    reference_shape = all_anchors[0].shape
    if len(reference_shape) != 4:
        raise ValueError(f"plan anchors must have shape (mode, cluster, time, xy), got {reference_shape}")
    if any(anchor.shape != reference_shape for anchor in all_anchors):
        raise ValueError("all plan anchor granularities must have the same shape")
    stacked = np.stack(all_anchors, axis=1)
    stacked = np.transpose(stacked, (0, 2, 1, 3, 4))
    return stacked.reshape(reference_shape[0], reference_shape[1] * len(all_anchors), reference_shape[2], reference_shape[3])


def topk(confidence, k, *inputs, return_indices=False):
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (
        indices + torch.arange(bs, device=indices.device)[:, None] * N
    ).reshape(-1)
    outputs = []
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    if return_indices:
        return confidence, outputs, indices
    else:
        return confidence, outputs


@PLUGIN_LAYERS.register_module()
class InstanceBank(nn.Module):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        spatial_anchor=None,
        anchor_handler=None,
        anchor_mearge=False,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
        max_time_interval=2,
        project_temporal_anchor=True,
        queue_length=1, # 保存的temporal info的长度。之前的版本和默认是保存1帧
        num_queue_instance=30,  # 默认保存30个instance, 这部分instance用于Plan感知，而不是预测下一阵的objects
        temp_gap=1, # 获取temporal info的间隔
        train_at_2hz=False,  # 497b118: 2Hz train uses older queue head as temporal cache
        queue_mode="plan",
    ):
        super(InstanceBank, self).__init__()
        assert queue_mode in ("temporal", "plan"), (
            f"queue_mode must be 'temporal' or 'plan', got {queue_mode!r}"
        )
        self.queue_mode = queue_mode
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval
        self.project_temporal_anchor = project_temporal_anchor
        self.queue_length = queue_length
        self.temp_gap = temp_gap
        self.num_queue_instance = num_queue_instance
        self.train_at_2hz = train_at_2hz
        self.instance_feature_queue = []
        self.anchor_queue = []
    
        if self.queue_mode == "plan" and self.queue_length > 1:
            num_use_query = self.queue_length // self.temp_gap
            temporal_posi_embedding = get_1d_sincos_pos_embed_from_grid(
                self.embed_dims, np.arange(num_use_query)
            )
            self.temporal_posi_embedding = nn.Parameter(
                torch.zeros(num_use_query, self.embed_dims), requires_grad=False
            )
            self.temporal_posi_embedding.data.copy_(
                torch.from_numpy(temporal_posi_embedding).float()
            )

        if anchor_handler is not None:
            anchor_handler = build_from_cfg(anchor_handler, PLUGIN_LAYERS)
            assert hasattr(anchor_handler, "anchor_projection")
        self.anchor_handler = anchor_handler
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        if isinstance(spatial_anchor, str):
            spatial_anchor = np.load(spatial_anchor)
        elif isinstance(spatial_anchor, (list, tuple)):
            spatial_anchor = np.array(spatial_anchor)
        else:
            spatial_anchor = anchor.copy()
            # print("--------NO Spatial Anchor Provided, Using Anchor as Spatial Anchor--------")
        if anchor_mearge:
            anchor = np.concatenate([anchor, spatial_anchor], axis=0)
            print("--------Temporal Anchor and Spatial Anchor Mearge--------")


        if len(anchor.shape) == 3: # for map
            anchor = anchor.reshape(anchor.shape[0], -1)
        if len(anchor.shape) == 4: # for plan anchor 3, 10, 6, 2
            anchor = rearrange(anchor, 'b n t c -> (b n) (t c)')

        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        # print("Initing Anchors Using Spatial Anchors")
        self.reset()

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def reset(self):
        self.cached_feature = None
        self.cached_anchor = None
        self.metas = None
        self.mask = None
        self.confidence = None
        self.temp_confidence = None
        self.instance_id = None
        self.prev_id = 0
        # 为了保存多帧的历史信息
        self.instance_feature_queue = []
        self.anchor_queue = []
        self.instance_id_queue = []
        self.metas_queue = []
        self.confidence_queue = []

    def get(self, batch_size, metas=None, dn_metas=None, force_check_timeinterval=False):
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        if (
            self.cached_anchor is not None
            and batch_size == self.cached_anchor.shape[0]
        ):
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            self.mask = torch.abs(time_interval) <= self.max_time_interval
            if force_check_timeinterval:
                # 将不连续的时间的特征清零
                self.cached_feature[~self.mask] = torch.zeros_like(
                    self.cached_feature[0]
                )

            if (
                self.queue_mode == "plan"
                and self.queue_length > 1
            ):
                for i in range(len(self.instance_feature_queue)):
                    self.instance_feature_queue[i][~self.mask] = torch.zeros_like(
                        self.instance_feature_queue[i][0]
                    )
                    self.anchor_queue[i][~self.mask] = torch.zeros_like(
                        self.anchor_queue[i][0]
                    )

            #
            if self.anchor_handler is not None and self.project_temporal_anchor:
                T_temp2cur = self.cached_anchor.new_tensor(
                    np.stack(
                        [
                            x["T_global_inv"]
                            @ self.metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(metas["img_metas"])
                        ]
                    )
                )
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor,
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]

            if (
                self.anchor_handler is not None
                and dn_metas is not None
                and batch_size == dn_metas["dn_anchor"].shape[0]
            ):
                num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
                dn_anchor = self.anchor_handler.anchor_projection(
                    dn_metas["dn_anchor"].flatten(1, 2),
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]
                dn_metas["dn_anchor"] = dn_anchor.reshape(
                    batch_size, num_dn_group, num_dn, -1
                )
            time_interval = torch.where(
                torch.logical_and(time_interval != 0, self.mask),
                time_interval,
                time_interval.new_tensor(self.default_time_interval),
            )
        elif self.queue_mode == "temporal" and len(self.instance_feature_queue) > 0:
            if batch_size != self.instance_feature_queue[-1].shape[0]:
                self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )
        elif self.queue_mode == "temporal":
            if self.cached_feature is not None:
                if self.cached_feature.shape[0] != batch_size:
                    self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )
        else:
            self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )

        if self.queue_mode == "plan" and self.queue_length > 1:
            # 获取历史特征
            if len(self.instance_feature_queue) < self.queue_length:
                # 遵循先进先出原则，index越大的越新
                for i in range(len(self.instance_feature_queue), self.queue_length):
                    self.instance_feature_queue.append(
                        torch.zeros_like(instance_feature[:, : self.num_queue_instance])
                    )
                    self.anchor_queue.append(
                        torch.zeros_like(anchor[:, : self.num_queue_instance])
                    )
                    self.instance_id_queue.append(None)
            output_instance_feature = []
            output_anchor = []
            count = 0
            for i in range(0, len(self.instance_feature_queue), self.temp_gap):
                output_instance_feature.append(self.instance_feature_queue[i])
                output_instance_feature[count] = (
                    output_instance_feature[count]
                    + self.temporal_posi_embedding[count].unsqueeze(0).unsqueeze(0)
                )
                output_anchor.append(self.anchor_queue[i])
                count += 1

            temporal_instance_feature = torch.stack(output_instance_feature, dim=2)
            temporal_anchor = torch.stack(output_anchor, dim=2)
        else:
            temporal_instance_feature = None
            temporal_anchor = None


        return (
            instance_feature,
            anchor,
            self.cached_feature,    # B, 600, 256
            self.cached_anchor,     # B, 600, 11
            self.confidence,
            temporal_instance_feature,
            temporal_anchor,
            time_interval,
        )

    def update(self, instance_feature, anchor, confidence):
        if self.cached_feature is None:
            return instance_feature, anchor

        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]


        N = self.num_anchor - self.num_temp_instances
        if N <= 0:
            return instance_feature, anchor

        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor) = topk(
            confidence, N, instance_feature, anchor
        )
        selected_feature = torch.cat(
            [self.cached_feature, selected_feature], dim=1
        )
        selected_anchor = torch.cat(
            [self.cached_anchor, selected_anchor], dim=1
        )
        instance_feature = torch.where(
            self.mask[:, None, None], selected_feature, instance_feature
        )
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)
        self.confidence = torch.where(
            self.mask[:, None],
            self.confidence,
            self.confidence.new_tensor(0)
        )
        if self.instance_id is not None:
            self.instance_id = torch.where(
                self.mask[:, None],
                self.instance_id,
                self.instance_id.new_tensor(-1),
            )

        if num_dn > 0:
            instance_feature = torch.cat(
                [instance_feature, dn_instance_feature], dim=1
            )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
        return instance_feature, anchor

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
        instance_id=None,
    ):
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()    # B, num_mode, C
        anchor = anchor.detach()
        confidence = confidence.detach()

        self.metas = metas
        confidence = confidence.max(dim=-1).values.sigmoid()
        if self.confidence is not None:
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
        self.temp_confidence = confidence
        if self.num_temp_instances == instance_feature.shape[1]:    # 直接全部保留
            self.cached_feature = instance_feature
            self.cached_anchor = anchor
            self.confidence = confidence
        else:
            (
                self.confidence,
                (self.cached_feature, self.cached_anchor),
            ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)
        

        # 保存多帧instance信息
        if self.queue_length > 1:
            if self.queue_mode == "temporal":
                self.instance_feature_queue.append(self.cached_feature.clone())
                self.anchor_queue.append(self.cached_anchor.clone())
                self.metas_queue.append(self.metas)
                self.confidence_queue.append(self.confidence.clone())
                if len(self.instance_feature_queue) > self.queue_length:
                    self.instance_feature_queue.pop(0)
                    self.anchor_queue.pop(0)
                    self.metas_queue.pop(0)
                    self.confidence_queue.pop(0)
            else:
                if instance_id is not None:
                    confidence_save, (feature_save, anchor_save, instance_id_save) = topk(
                        confidence,
                        self.num_queue_instance,
                        instance_feature,
                        anchor,
                        instance_id,
                    )
                else:
                    confidence_save, (feature_save, anchor_save) = topk(
                        confidence, self.num_queue_instance, instance_feature, anchor
                    )
                    instance_id_save = None
                self.instance_feature_queue.append(feature_save)
                self.anchor_queue.append(anchor_save)
                self.instance_id_queue.append(instance_id_save)
                self.metas_queue.append(self.metas)
                self.confidence_queue.append(self.confidence.clone())
                if len(self.instance_feature_queue) > self.queue_length:
                    self.instance_feature_queue.pop(0)
                    self.anchor_queue.pop(0)
                    self.instance_id_queue.pop(0)
                    self.metas_queue.pop(0)
                    self.confidence_queue.pop(0)

        if self.train_at_2hz and self.queue_mode == "temporal":
            if len(self.instance_feature_queue) >= self.queue_length:
                self.cached_feature = self.instance_feature_queue[0]
                self.cached_anchor = self.anchor_queue[0]
                self.confidence = self.confidence_queue[0]
                self.metas = self.metas_queue[0]
            else:
                self.cached_feature = None
                self.cached_anchor = None
                self.confidence = None
                self.metas = None

    def get_instance_id(self, confidence, anchor=None, threshold=None):
        confidence = confidence.max(dim=-1).values.sigmoid()
        instance_id = confidence.new_full(confidence.shape, -1).long()

        
        if (
            self.instance_id is not None
            and self.instance_id.shape[0] == instance_id.shape[0]
        ):
            instance_id[:, : self.instance_id.shape[1]] = self.instance_id    #

        mask = instance_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)



        num_new_instance = mask.sum()
        new_ids = torch.arange(num_new_instance).to(instance_id) + self.prev_id
        instance_id[torch.where(mask)] = new_ids
        self.prev_id += num_new_instance
        self.update_instance_id(instance_id, confidence)    # 只保留了600个有效ID， 历史query和历史id是对应上的
        return instance_id

    def update_instance_id(self, instance_id=None, confidence=None):
        if self.temp_confidence is None:
            if confidence.dim() == 3:  # bs, num_anchor, num_cls
                temp_conf = confidence.max(dim=-1).values
            else:  # bs, num_anchor
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence
        instance_id = topk(temp_conf, self.num_temp_instances, instance_id)[1][
            0
        ]
        instance_id = instance_id.squeeze(dim=-1)
        self.instance_id = F.pad(
            instance_id,
            (0, self.num_anchor - self.num_temp_instances),
            value=-1,
        )



@PLUGIN_LAYERS.register_module()
class InstanceBank_HiP_AD(InstanceBank):
    def __init__(self, *args, **kwargs):
        kwargs["queue_mode"] = "temporal"
        super().__init__(*args, **kwargs)


@PLUGIN_LAYERS.register_module()
class InstanceBankPlan(nn.Module):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_handler=None,
        anchor_mearge=False,
        num_temp_instances=0,
        default_time_interval=0.2,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
        max_time_interval=2,
        project_temporal_anchor=True,
        queue_length=3,
        temporal_gap=0.5,
        used_grand=None,
    ):
        super(InstanceBankPlan, self).__init__()
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval
        self.project_temporal_anchor = project_temporal_anchor
        self.temporal_gap = int(temporal_gap)
        if self.temporal_gap <= 0:
            self.temporal_gap = 1
            print("temporal_gap should be greater than 0, set to 1")
        
        if anchor_handler is not None:
            anchor_handler = build_from_cfg(anchor_handler, PLUGIN_LAYERS)
            assert hasattr(anchor_handler, "anchor_projection")
        self.anchor_handler = anchor_handler
        self.anchor_grand = None
        if isinstance(anchor, str):
            if anchor.endswith('.npy'):
                anchor = np.load(anchor)
            else:
                with open(anchor, 'rb') as f:
                    anchor = pickle.load(f)
                if used_grand is None:
                    self.anchor_grand = list(anchor.keys())
                    all_anchors = []
                    for key in self.anchor_grand:
                        all_anchors.append(anchor[key])
                    anchor = _stack_plan_anchors(all_anchors)
                else:# 只使用指定的粒度
                    all_anchors = []
                    max_len = 0
                    for key in used_grand:
                        all_anchors.append(anchor[key])     # anchor[key]: N, M, T, 2
                        if max_len < anchor[key].shape[2]:
                            max_len = anchor[key].shape[2]
                        # 如果没有达到指定的长度，则补齐
                    for i in range(len(all_anchors)):
                        if all_anchors[i].shape[2] < max_len:
                            all_anchors[i] = np.pad(all_anchors[i], ((0, 0), (0, 0), (0, max_len - all_anchors[i].shape[2]), (0, 0)), mode='constant', constant_values=0)

                    anchor = _stack_plan_anchors(all_anchors)
                    self.anchor_grand = used_grand


        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)

        anchor = np.concatenate([np.zeros([anchor.shape[0], anchor.shape[1], 1, anchor.shape[3]]), anchor], axis=2)    # 添加一个全零的anchor，作为padding
        anchor = anchor[:,:, 1:] - anchor[:,:, :-1]

        if len(anchor.shape) == 3: # for map
            anchor = anchor.reshape(anchor.shape[0], -1)
        if len(anchor.shape) == 4: # for plan anchor 3, 10, 6, 2
            anchor = rearrange(anchor, 'b n t c -> (b n) (t c)')

        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )

        # 时间位置编码 
        self.queue_length = queue_length
        temporal_posi_embedding = get_1d_sincos_pos_embed_from_grid(self.embed_dims, np.arange(self.queue_length))
        self.temporal_posi_embedding = nn.Parameter(torch.zeros(self.queue_length, self.embed_dims), requires_grad=False)
        self.temporal_posi_embedding.data.copy_(torch.from_numpy(temporal_posi_embedding).float())



        # print("Initing Anchors Using Spatial Anchors")
        self.reset()

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def reset(self):
        # self.cached_feature = None
        # self.cached_anchor = None

        self.instance_feature_queue = []
        self.anchor_queue = []

        self.metas = None
        self.mask = None
        self.confidence = None
        self.temp_confidence = None
        self.instance_id = None
        self.prev_id = 0

    def get(self, batch_size, metas=None, dn_metas=None, force_check_timeinterval=False):
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))


        # 不是第一帧，并且缓存队列不为空
        force_reset = True
        if (len(self.anchor_queue) > 0):
            if (batch_size == self.anchor_queue[-1].shape[0]):  # train到val的切换
                history_time = self.metas["timestamp"]
                time_interval = metas["timestamp"] - history_time
                time_interval = time_interval.to(dtype=instance_feature.dtype)
                self.mask = torch.abs(time_interval) <= self.max_time_interval
                if force_check_timeinterval:
                    # 将不连续的时间的特征清零
                    for i in range(len(self.instance_feature_queue)):
                        self.instance_feature_queue[i][~self.mask] = torch.zeros_like(self.instance_feature_queue[i][0])  # 不连续的位置清零
                        self.anchor_queue[i][~self.mask] = torch.zeros_like(self.anchor_queue[i][0])  # 不连续的位置清零

                time_interval = torch.where(
                    torch.logical_and(time_interval != 0, self.mask),
                    time_interval,
                    time_interval.new_tensor(self.default_time_interval),
                )
                force_reset = False

        if force_reset:
            self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )

        # 获取历史特征
        if len(self.instance_feature_queue) < self.queue_length:
            # 遵循先进先出原则，index越大的越新
            for i in range(len(self.instance_feature_queue), self.queue_length):
                self.instance_feature_queue.append(torch.zeros_like(instance_feature))
                self.anchor_queue.append(torch.zeros_like(anchor))
        # 为每个时刻的历史特征添加时间位置编码
        output_instance_feature = []
        output_anchor = []
        for i in range(0, len(self.instance_feature_queue), self.temporal_gap):
            output_instance_feature.append(
                self.instance_feature_queue[i] + 
                self.temporal_posi_embedding[i].unsqueeze(0).unsqueeze(0)
            )
            output_anchor.append(self.anchor_queue[i])
            # 下述操作会导致缓存队列中的特征被反复添加时间位置编码
            # self.instance_feature_queue[i] = self.instance_feature_queue[i] + self.temporal_posi_embedding[i].unsqueeze(0).unsqueeze(0)  # [B, N, D]

        # 获取历史特征以及anchor
        temporal_instance_feature = torch.stack(output_instance_feature, dim=2)  # [B,N,T,D]
        temporal_anchor = torch.stack(output_anchor, dim=2)  # [B, N, T, D]







        return (
            instance_feature,
            anchor,
            temporal_instance_feature,
            temporal_anchor,
            None,
            time_interval,
        )

    def update(self, instance_feature, anchor, confidence):
        if self.cached_feature is None:
            return instance_feature, anchor

        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        N = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor) = topk(
            confidence, N, instance_feature, anchor
        )
        selected_feature = torch.cat(
            [self.cached_feature, selected_feature], dim=1
        )
        selected_anchor = torch.cat(
            [self.cached_anchor, selected_anchor], dim=1
        )
        instance_feature = torch.where(
            self.mask[:, None, None], selected_feature, instance_feature
        )
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)
        self.confidence = torch.where(
            self.mask[:, None],
            self.confidence,
            self.confidence.new_tensor(0)
        )
        if self.instance_id is not None:
            self.instance_id = torch.where(
                self.mask[:, None],
                self.instance_id,
                self.instance_id.new_tensor(-1),
            )

        if num_dn > 0:
            instance_feature = torch.cat(
                [instance_feature, dn_instance_feature], dim=1
            )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
        return instance_feature, anchor

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()    # B, num_mode, C
        anchor = anchor.detach()
        confidence = confidence.detach()

        self.metas = metas
        confidence = confidence.max(dim=-1).values.sigmoid()
        if self.confidence is not None:
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
        self.temp_confidence = confidence
        if self.num_temp_instances == instance_feature.shape[1]:    # 直接全部保留
            self.instance_feature_queue.append(instance_feature)
            self.anchor_queue.append(anchor)
            self.confidence = confidence
            if len(self.instance_feature_queue) > self.queue_length:
                self.instance_feature_queue.pop(0)
                self.anchor_queue.pop(0)

        else:
            (
                confidence,
                (cached_feature, cached_anchor),
            ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)
            self.confidence = confidence
            self.instance_feature_queue.append(cached_feature)
            self.anchor_queue.append(cached_anchor)
            if len(self.instance_feature_queue) > self.queue_length:
                self.instance_feature_queue.pop(0)
                self.anchor_queue.pop(0)






@PLUGIN_LAYERS.register_module()
class DrivePathBank(nn.Module):
    def __init__(
        self,
        plan_grand,
        num_modal,
        anchor_grand,
        embed_dims,
        timeinterval=5,
        mutil_step_query=1,
        speed_range=[0.2, 3],
        num_ds=2,
        drivepath_grand='spatial_interval_5_traj',
        prediction_mode='waypoints',
        **kwargs,
    ):
        super(DrivePathBank, self).__init__()
        self.embed_dims = embed_dims
        self.anchor_grand = anchor_grand
        self.plan_grand = plan_grand
        self.num_grand = len(plan_grand)
        self.drivepath_list = []
        self.num_modal = num_modal
        self.mutil_step_query = mutil_step_query
        self.drivepath_grand = drivepath_grand
        self.drivepath_list.append(plan_grand.index(self.drivepath_grand))  # 只使用一个粒度的drive path
        self.drivepath_list = torch.tensor(self.drivepath_list, dtype=torch.long)
        self.num_drivepath = len(self.drivepath_list) * self.num_modal
        self.num_ds = num_ds    # 速度风格的数量
        self.num_speed_ds = num_ds * self.num_modal # query的数量
        self.speed_scale = timeinterval * 0.1 # 0.1是采集频率，interval是间隔
        self.prediction_mode = prediction_mode
        self.queue_length = kwargs.get('queue_length', 1)  # 保存的temporal info的长度。之前的版本和默认是保存1帧
        # self.speed_range = torch.tensor([0.2, 1.5, 6]).cuda()
        if speed_range is None:
            self.speed_range = None
        else:
            # 
            average_speed = []
            for i in range(len(speed_range)):
                average_speed.append(np.array(speed_range[i]).mean())

            self.register_buffer('speed_range', torch.tensor(average_speed), persistent=False)
        self.instance_feature = nn.Parameter(       # 随机初始化
            torch.randn(self.num_speed_ds, self.mutil_step_query, embed_dims),   
            requires_grad=True,
        )
        queue_length = kwargs.get('queue_length', 1)  # 保存的temporal info的长度。之前的版本和默认是保存1帧
        self.queue_length = queue_length
        if self.queue_length > 1:
            temporal_posi_embedding = get_1d_sincos_pos_embed_from_grid(self.embed_dims, np.arange(self.queue_length))
            self.temporal_posi_embedding = nn.Parameter(torch.zeros(self.queue_length, self.embed_dims), requires_grad=False)
            self.temporal_posi_embedding.data.copy_(torch.from_numpy(temporal_posi_embedding).float())
        self.temporal_queue = []

        
    def init_weight(self):
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)
            
    def get(self, trajs, reference_feature, ego_speed):
        # 从 trajs 中掏出来drive_path
        # trajs: torch.Tensor, 形状为(B, M, 12)
        bs = trajs.shape[0]
        num_t = trajs.shape[-1] // 2     # 最后一个维度除以2就是时刻数
    
        trajs = rearrange(trajs, 'b (m g) c -> b m g c', m=self.num_modal, g=self.num_grand)   # (B, M, N, 12)
        drive_path = trajs[:, :, self.drivepath_list, :]  # (B, M, N, 12)
        drive_path = rearrange(drive_path, 'b m g c -> b (m g) c')  # (B, MN, 12)
        drive_path = rearrange(drive_path, 'b n (t k) -> (b n) t k', t=num_t, k=2)  # (B*N, 6, 2) 将Batch维度融合
        
        speed = ego_speed * self.speed_scale # (B, 1), 取出速度信息, 做单位转换, Wapoints的距离是 m / 0.5s 取决于interval
        speed = speed.repeat(1, self.num_drivepath, 1)  # (B, M*N), 扩展到与drive_path相同的维度
        speed = rearrange(speed, 'b m n -> (b m) n', m=self.num_drivepath, n=1)  # (B*N, 1)

        speed_anchors = []

        if self.speed_range is None:
            anchors = drive_path.clone()
            anchors = rearrange(anchors, '(b m) t k -> b m (t k)', b=bs, m=self.num_modal, t=num_t, k=2)
            for i in range(self.num_ds):
                speed_anchors.append(anchors)
        else:
            # 根据当前速度获取anchor 
            # speed_bins = speed.repeat(1, num_t)  # (B*N, 6), 扩展到与drive_path相同的时间步数
            # anchors = obtain_traj_given_speed(drive_path, speed_bins)  # (B*N, 6, 2)    
            # anchors = rearrange(anchors, '(b m) t k -> b m (t k)', b=bs, m=self.num_modal, t=num_t, k=2) # 注意，本身应该有一个维度为1，代表drivepath的数量，这里为了简化，直接丢掉了
            # speed_anchors.append(anchors)
            # B, M , N, 6 ,2 --> Batch_size, 模态数, 粒度(相当于只保留了空间粒度这里), 时间步数, 坐标维度 
            # 根据预定义速度范围生成anchor
            for speed_predefined in self.speed_range:
                speed_set = speed_predefined * self.speed_scale  # 将预定义速度转换为实际速度
                speed_set = speed_set.unsqueeze(0).repeat(bs, self.num_drivepath)
                speed_set = rearrange(speed_set, 'b (m n) -> (b m) n', m=self.num_drivepath, n=1)  # (B*N, 1)
                speed_set = speed_set.repeat(1, num_t)  # (B*N, 6), 扩展到与drive_path相同的时间步数

                # 预测模式为waypoints时，获取对应速度的anchor
                anchor_set = obtain_traj_given_speed(drive_path, speed_set.to(drive_path.dtype), num_horzion=num_t).to(drive_path.dtype)  # (B*N, 6, 2) # 注意，本身应该有一个维度为1，代表drivepath的数量，这里为了简化，直接丢掉了
                anchor_set = rearrange(anchor_set, '(b m) t k -> b m (t k)', b=bs, m=self.num_modal, t=num_t, k=2)
                speed_anchors.append(anchor_set)

        # B, M , N, 6 ,2 --> Batch_size, 模态数, 粒度(相当于只保留了空间粒度这里), 时间步数, 坐标维度 
        speed_anchors = torch.stack(speed_anchors, dim=2)   
        # (B, M ,N ,K, 12) -> Batch_size, 模态数, 粒度(相当于只保留了空间粒度这里), 不同的速度采样, 12
        full_anchors = rearrange(speed_anchors, 'b m n t -> b (m n) t')  # (B, N, 2)

        instance_features = torch.tile(self.instance_feature[None], (bs, 1, 1, 1) )
        # instance_features = instance_features.squeeze(-2)
        # 恢复形状
        drive_path = drive_path.unsqueeze(1)  # (B*N, 1, 6, 2)
        drive_path = drive_path.repeat(1, self.num_ds, 1, 1)  # 与speed anchor对齐
        drive_path = rearrange(drive_path, '(b n) m t k -> b (n m) t k',b=bs, t=num_t, k=2)  # (B*N, 6, 2) 将Batch维度融合

        reference_feature = reference_feature.unsqueeze(2)  # (B, N, 1, C)
        reference_feature = reference_feature.repeat(1, 1, self.num_ds, 1)  # (B, N, M, C)
        reference_feature = rearrange(reference_feature, 'b n m c -> b (n m) c')  # (B*N, C)

        return full_anchors, instance_features, reference_feature, drive_path


    # def cache(
    #     self,
    #     instance_feature,
    #     anchor,
    #     confidence,
    #     metas=None,
    # ):
    #     if self.num_temp_instances <= 0:
    #         return
    #     instance_feature = instance_feature.detach()    # B, num_mode, C
    #     anchor = anchor.detach()
    #     confidence = confidence.detach()

    #     self.metas = metas
    #     confidence = confidence.max(dim=-1).values.sigmoid()
    #     if self.confidence is not None:
    #         confidence[:, : self.num_temp_instances] = torch.maximum(
    #             self.confidence * self.confidence_decay,
    #             confidence[:, : self.num_temp_instances],
    #         )
    #     self.temp_confidence = confidence
    #     if self.num_temp_instances == instance_feature.shape[1]:    # 直接全部保留
    #         self.instance_feature_queue.append(instance_feature)
    #         self.anchor_queue.append(anchor)
    #         self.confidence = confidence
    #         if len(self.instance_feature_queue) > self.queue_length:
    #             self.instance_feature_queue.pop(0)
    #             self.anchor_queue.pop(0)

    #     else:
    #         (
    #             confidence,
    #             (cached_feature, cached_anchor),
    #         ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)
    #         self.confidence = confidence
    #         self.instance_feature_queue.append(cached_feature)
    #         self.anchor_queue.append(cached_anchor)
    #         if len(self.instance_feature_queue) > self.queue_length:
    #             self.instance_feature_queue.pop(0)
    #             self.anchor_queue.pop(0)



@PLUGIN_LAYERS.register_module()
class HistoryPlanBank(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_temp_instances=0,
        default_time_interval=0.2,
        confidence_decay=0.6,
        max_time_interval=2,
        project_temporal_anchor=True,
        queue_length=3,
        temporal_gap=0.5,
    ):
        super(HistoryPlanBank, self).__init__()
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval
        self.project_temporal_anchor = project_temporal_anchor
        if temporal_gap <= 1:
            temporal_gap = 1
            print("temporal_gap should be greater than 0, set to 1")
        self.temporal_gap = int(temporal_gap)
    
        # 时间位置编码 
        self.queue_length = queue_length
        temporal_posi_embedding = get_1d_sincos_pos_embed_from_grid(self.embed_dims, np.arange(self.queue_length))
        self.temporal_posi_embedding = nn.Parameter(torch.zeros(self.queue_length, self.embed_dims), requires_grad=False)
        self.temporal_posi_embedding.data.copy_(torch.from_numpy(temporal_posi_embedding).float())
        # print("Initing Anchors Using Spatial Anchors")
        self.reset()


    def reset(self):
        # self.cached_feature = None
        # self.cached_anchor = None

        self.instance_feature_queue = []
        self.anchor_queue = []

        self.metas = None
        self.mask = None
        self.confidence = None
        self.temp_confidence = None
        self.instance_id = None
        self.prev_id = 0

    def get(self, batch_size, instance_feature, anchor, metas=None, dn_metas=None, force_check_timeinterval=False):

        force_reset = True
        if (len(self.anchor_queue) > 0):
            if (batch_size == self.anchor_queue[-1].shape[0]):  # train到val的切换
                history_time = self.metas["timestamp"]
                time_interval = metas["timestamp"] - history_time
                time_interval = time_interval.to(dtype=instance_feature.dtype)
                self.mask = torch.abs(time_interval) <= self.max_time_interval
                if force_check_timeinterval:
                    # 将不连续的时间的特征清零
                    for i in range(len(self.instance_feature_queue)):
                        self.instance_feature_queue[i][~self.mask] = torch.zeros_like(self.instance_feature_queue[i][0])  # 不连续的位置清零
                        self.anchor_queue[i][~self.mask] = torch.zeros_like(self.anchor_queue[i][0])  # 不连续的位置清零

                time_interval = torch.where(
                    torch.logical_and(time_interval != 0, self.mask),
                    time_interval,
                    time_interval.new_tensor(self.default_time_interval),
                )
                force_reset = False

        if force_reset:
            self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )

        # 获取历史特征
        if len(self.instance_feature_queue) < self.queue_length:
            # 遵循先进先出原则，index越大的越新
            for i in range(len(self.instance_feature_queue), self.queue_length):
                self.instance_feature_queue.append(torch.zeros_like(instance_feature))
                self.anchor_queue.append(torch.zeros_like(anchor))

        # 为每个时刻的历史特征添加时间位置编码
        output_instance_feature = []
        output_anchor = []
        for i in range(0, len(self.instance_feature_queue), self.temporal_gap):
            output_instance_feature.append(
                self.instance_feature_queue[i] + 
                self.temporal_posi_embedding[i].unsqueeze(0).unsqueeze(0)
            )
            output_anchor.append(self.anchor_queue[i])
            # 下述操作会导致缓存队列中的特征被反复添加时间位置编码
            # self.instance_feature_queue[i] = self.instance_feature_queue[i] + self.temporal_posi_embedding[i].unsqueeze(0).unsqueeze(0)  # [B, N, D]

        # 获取历史特征以及anchor
        temporal_instance_feature = torch.stack(output_instance_feature, dim=2)  # [B,N,T,D]
        temporal_anchor = torch.stack(output_anchor, dim=2)  # [B, N, T, D]


        return (
            temporal_instance_feature,
            temporal_anchor,
            None,
            time_interval,
        )




    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
    ):
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()    # B, num_mode, C
        anchor = anchor.detach()
        confidence = confidence.detach()

        self.metas = metas
        confidence = confidence.max(dim=-1).values.sigmoid()
        if self.confidence is not None:
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
        self.temp_confidence = confidence
        if self.num_temp_instances == instance_feature.shape[1]:    # 直接全部保留
            self.instance_feature_queue.append(instance_feature)
            self.anchor_queue.append(anchor)
            self.confidence = confidence
            if len(self.instance_feature_queue) > self.queue_length:
                self.instance_feature_queue.pop(0)
                self.anchor_queue.pop(0)

        else:
            (
                confidence,
                (cached_feature, cached_anchor),
            ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)
            self.confidence = confidence
            self.instance_feature_queue.append(cached_feature)
            self.anchor_queue.append(cached_anchor)
            if len(self.instance_feature_queue) > self.queue_length:
                self.instance_feature_queue.pop(0)
                self.anchor_queue.pop(0)





class Img_instance_queue(nn.Module):
    # cache the history feature map
    def __init__(self, queue_length=1, n_embd=256, recurrent_temporal=False, add_tempora_posi=False, temporal_gap=1):
        super(Img_instance_queue, self).__init__()
        self.temporal_gap = temporal_gap
        self.recurrent_temporal = recurrent_temporal
        
        # 如只使用3帧历史帧，但是每帧之间的gap是间隔3
        self.queue_length = queue_length * temporal_gap
        self.num_use_temporal = queue_length
        
        self.temporal_feature_queue = []  # 形状: [T, B, N, D]
        self.timestamp_queue = []  # 存储 [T, B] 的时间戳
        self.meta_queue = []  # 存储 [T, B] 的元信息
        self.period = 0
        self.add_tempora_posi = add_tempora_posi
        if add_tempora_posi:
            temporal_posi_embedding = get_1d_sincos_pos_embed_from_grid(n_embd, np.arange(self.num_use_temporal))
            self.temporal_posi_embedding = nn.Parameter(torch.zeros(self.num_use_temporal, n_embd), requires_grad=False)
            self.temporal_posi_embedding.data.copy_(torch.from_numpy(temporal_posi_embedding).float())
        self.mask = None
        self.defualt_interval = 2   # 假设同一个场景内的帧间隔不会超过2s (应当考虑数据精度)
        self.batch_size = 0

    def reset(self):
        """ 重置缓存队列 """
        self.temporal_feature_queue = []
        self.timestamp_queue = []
        self.meta_queue = []
        self.period = 0
        self.mask = None
        self.batch_size = 0


    def check(self, features, timestamp):
        # features: [formated feature maps] list
        feature_maps = features[0]  # features[1], features[2]是formated之后的information

        B = feature_maps.shape[0]  # 获取 batch size
        if (self.batch_size !=0) and (B != self.batch_size):
            self.reset()
        if len(self.timestamp_queue) > 0:
            prev_timestamp = self.timestamp_queue[-1]  # 取出上一时间步的所有样本时间戳
            self.mask = torch.abs(timestamp - prev_timestamp) < self.defualt_interval  # [B]，判断 batch 内每个样本是否连续
        else:
            self.mask = torch.ones_like(timestamp, dtype=torch.bool)  # 第一个时间步默认连续
        # 处理历史特征缓存
        if len(self.temporal_feature_queue) > 0:
            for i in range(len(self.temporal_feature_queue)):
                self.temporal_feature_queue[i][~self.mask] = torch.zeros_like(self.temporal_feature_queue[i][0])  # 不连续的位置清零
        else:
            prev_features = torch.zeros_like(feature_maps)  # 第一个时间步没有历史特征，初始化零特征



    def cache(self, feature, timestamp, meta=None):
        """
        缓存历史特征
        :param feature: list [formated feature maps]
        :param timestamp: Tensor [B]  (每个样本的时间戳)
        """
        feature_map = feature[0]  # features[1], features[2]是formated之后的information
        B = feature_map.shape[0]  # 获取 batch size
        # 追加到队列
        # self.ego_feature_queue.append(prev_features)
        self.temporal_feature_queue.append(feature_map.detach())  # 不传递梯度
        self.timestamp_queue.append(timestamp)  # 存储时间戳
        if meta is not None:
            projection_mat = meta['projection_mat']
            image_wh = meta['image_wh']
            meta_dict = {'projection_mat': projection_mat, 'image_wh': image_wh}
            self.meta_queue.append(meta_dict)
        else:
            self.meta_queue.append(None)
        self.period += 1
        # 超过长度限制，则丢弃最老的时间步
        if len(self.temporal_feature_queue) > self.queue_length:
            self.temporal_feature_queue.pop(0)
            self.timestamp_queue.pop(0)
            self.meta_queue.pop(0)
        self.batch_size = B
        return True

    def get_feature(self, current_feature, current_meta, length=-1, cat_features=False):
        """
        # current_feature的形状需要用来做0初始化
        取出最近 `length` 轮的特征，并加上时间编码
        """
        current_feature_map = current_feature[0]
        feature_map_info = current_feature[1:]  # 各个feature Map的尺寸信息等

        if length == -1:
            # 按照self.temporal_gap跳着取
            features = self.temporal_feature_queue[:-(self.temporal_gap-1)][::-1][::self.temporal_gap][::-1]
            # 取出所有的时间戳
            # features = self.temporal_feature_queue  # 取历史帧（去掉当前帧）
            meta_info = self.meta_queue
        else:
            features = self.temporal_feature_queue[-length:]
            meta_info = self.meta_queue[-length:]

        current_projection_mat = current_meta['projection_mat']
        current_image_wh = current_meta['image_wh']
        current_meta = {'projection_mat': current_projection_mat, 'image_wh': current_image_wh}
        # 前面的帧数不够，用0填充
        if len(features) < self.num_use_temporal:
            features = [torch.zeros_like(current_feature_map) for _ in range(self.num_use_temporal - len(features))] + features
            # meta 如果不够，用当前帧的meta填充
            meta_info = [current_meta for _ in range(self.num_use_temporal - len(meta_info))] + meta_info

        if isinstance(features, list):
            for i in range(len(features)):  # 添加时序的位置编码
                if self.add_tempora_posi:
                    features[i] = features[i] + self.temporal_posi_embedding[i].unsqueeze(0).unsqueeze(0)  # Batch维度和N_token维度
            if cat_features:
                features = torch.cat(features, dim=1)  # 在 token 维度拼接，形状 [B, T*N, D]

        # warp feature map

        formated_features = []
        for i in range(len(features)):
            formated_features.append([features[i]] + feature_map_info)


        return formated_features, meta_info
