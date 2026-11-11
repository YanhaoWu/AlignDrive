import torch
import torch.nn as nn
import numpy as np

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn import xavier_init
from einops.einops import rearrange

from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
)

# from mmdet3d_plugin.core.box3d import *
from ..blocks import linear_relu_ln


@PLUGIN_LAYERS.register_module()
class SpecialSignAware(BaseModule):
    def __init__(
        self,
        mode='sum'
    ):
        super(SpecialSignAware, self).__init__()
        self.mode = mode


    def forward(
        self,
        query,
        det_feature,
    ):

        return query + det_feature




@PLUGIN_LAYERS.register_module()
class TemporalMixer(BaseModule):
    def __init__(
        self,
        embed_dims=256,
    ):
        super(TemporalMixer, self).__init__()
        self.embed_dims = embed_dims
        self.temporal_mlp = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
        )
        # self.score_mlp = nn.Sequential(
        #     nn.Linear(embed_dims, embed_dims),
        #     nn.ReLU(),
        # )


    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)

    def forward(
        self,
        plan_querys,
        temporal_plan_querys,
        temporal_plan_query_scores,
        longitudinal_embedding,
        metas,
    ):
        bs = plan_querys.shape[0]

        if temporal_plan_querys is None:
            temporal_plan_querys = plan_querys
            temporal_plan_query_scores = torch.ones(
                bs, temporal_plan_querys.shape[1], 1, device=plan_querys.device
            ).sigmoid()
            # 在Query维度进行归一化
        temporal_plan_querys = temporal_plan_querys / (
            temporal_plan_querys.norm(dim=1, keepdim=True) + 1e-6
        )

        # 根据得分对历史轨迹进行缩放
        if len(temporal_plan_query_scores.shape) == 2:
            temporal_plan_query_scores = temporal_plan_query_scores.unsqueeze(-1)
        temporal_plan_querys = self.temporal_mlp(temporal_plan_querys)
        temporal_plan_querys = temporal_plan_querys * temporal_plan_query_scores # 不需要sigmod,历史保存confidence时，已经用了sigmod

        # 加入ego_status信息
        if longitudinal_embedding is not None:
            if len(longitudinal_embedding.shape) == 2:
                longitudinal_embedding = longitudinal_embedding.unsqueeze(1)
            plan_querys = plan_querys + longitudinal_embedding

        return plan_querys, temporal_plan_querys
