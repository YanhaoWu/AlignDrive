import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
    ATTENTION,
)

from projects_sp.mmdet3d_plugin.core.box3d import *
from ..blocks import linear_relu_ln
from mmcv.utils import build_from_cfg
import torch.nn.functional as F
from einops import rearrange
from types import SimpleNamespace

@PLUGIN_LAYERS.register_module()
class MotionPlanningRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
    ):
        super(MotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.motion_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.motion_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, fut_ts * 2),
        )
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward( 
        self,
        motion_query,
        plan_query,
        ego_feature,
        ego_anchor_embed,
    ):
        bs, num_anchor = motion_query.shape[:2]
        motion_cls = self.motion_cls_branch(motion_query).squeeze(-1)
        motion_reg = self.motion_reg_branch(motion_query).reshape(bs, num_anchor, self.fut_mode, self.fut_ts, 2)
        plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)
        plan_reg = self.plan_reg_branch(plan_query).reshape(bs, 1, 3 * self.ego_fut_mode, self.ego_fut_ts, 2)
        planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed)
        return motion_cls, motion_reg, plan_cls, plan_reg, planning_status



# ----------------------------------------------------------------------------------e2e--------------------------------------------------------------------------------------------
def get_norm(norm_type, dim, **kwargs):
    if norm_type == "layer_norm":
        return LayerNorm(dim, **kwargs)
    elif norm_type == "rms_norm":
        return RMSNorm(dim, **kwargs)
    else:
        raise ValueError(f"norm type {norm_type} not supported")


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(
            input, self.weight.shape, self.weight, self.bias, 1e-5
        ).to(input.dtype)


class RMSNorm(torch.nn.Module):
    # from https://github.com/meta-llama/llama/blob/main/llama/model.py#L34
    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


@PLUGIN_LAYERS.register_module()
class BlockAR(nn.Module):
    def __init__(self, embed_dims, num_heads, batch_first=True, dropout=0.0, bias=True, flash_attention=True, norm_type="layer_norm"):
        super().__init__()
        norm_kwargs = {"bias": bias} if norm_type == "layer_norm" else {}
        # Normalization layers
        self.ln_temporal = get_norm(norm_type, embed_dims, **norm_kwargs)
        self.ln_mlp_1 = get_norm(norm_type, embed_dims, **norm_kwargs)

        assert flash_attention, "Only support flash attention for now"

        temporal_attn_cfg = (
            dict(type="CausalFlashAttention", n_embd=embed_dims, n_head=num_heads, bias=bias, dropout=dropout, causal=True)
        )
        # 使用 build_from_cfg 构建 temporal_attn
        self.temporal_attn = build_from_cfg(temporal_attn_cfg, ATTENTION)

        # MLP layers
        config = SimpleNamespace(
            n_embd=embed_dims,
            bias=bias,
            dropout=dropout
        )
        self.mlp_1 = build_from_cfg(
            dict(type="MLP", config=config), PLUGIN_LAYERS
        )
        self.torch_cp_enabled = False

    def forward_func(self, x, kvcache=None):
        B, T, C = x.shape  # x: (batch_size, timesteps, channels)
        temporal_out, cache_ele_t = self.temporal_attn(self.ln_temporal(x), kvcache)
        x = x + temporal_out
        x = x + self.mlp_1(self.ln_mlp_1(x))
        return x, cache_ele_t if kvcache is not None else None

    def forward(self, x, kvcache=None):
        return self.forward_func(x, kvcache)




@PLUGIN_LAYERS.register_module()
class BlockPAR(nn.Module):
    def __init__(self, config, norm_type="layer_norm"):
        super().__init__()
        norm_kwargs = {"bias": config.bias} if norm_type == "layer_norm" else {}

        # Normalization layers
        self.ln_x = get_norm(norm_type, config.n_embd, **norm_kwargs)
        self.ln_y = get_norm(norm_type, config.n_embd, **norm_kwargs)
        self.ln_mlp_1 = get_norm(norm_type, config.n_embd, **norm_kwargs)

        # # Temporal attention layer (causal)
        # self.temporal_attn = (
        #     CausalFlashAttention(config, causal=True)
        #     if config.flash_attention
        #     else CausalSelfAttention(config, causal=True, block_size=config.block_size)
        # )

        ca_attn_cfg = (
            dict(type="CrossAttention", dim=config.n_embd, num_heads=config.n_head, qkv_bias=config.bias ,causal=False)
        )
        # ca_attn_cfg = (
        #     dict(type="CausalFlashAttention", dim=config.n_embd, num_heads=config.n_head, qkv_bias=config.bias ,causal=False)
        # )


        # 使用 build_from_cfg 构建 temporal_attn
        self.cross_attn = build_from_cfg(ca_attn_cfg, ATTENTION)
        # MLP layers
        self.mlp_1 = build_from_cfg(
            dict(type="MLP", config=config), PLUGIN_LAYERS
        )

        sa_attn_cfg = (   
            dict(type="CausalFlashAttention", n_embd=config.n_embd, n_head=config.n_head, bias=config.bias, dropout=config.dropout, causal=False)
        )
        self.self_attn = build_from_cfg(sa_attn_cfg, ATTENTION)
        self.mlp_2 = build_from_cfg(
            dict(type="MLP", config=config), PLUGIN_LAYERS
        )


        self.torch_cp_enabled = False

    def forward_func(self, x, y, kvcache=None):
        B, T, C = x.shape  # x: (batch_size, timesteps, spatial_size, channels)
        # cross-attention
        attn_out, cache_ele_t = self.cross_attn(self.ln_x(x), self.ln_y(y), kvcache)
        x = x + attn_out
        x = x + self.mlp_1(self.ln_mlp_1(x))
        # self-attention
        attn_out, cache_ele_t = self.self_attn(self.ln_x(x), kvcache)
        x = x + attn_out
        x = x + self.mlp_2(self.ln_mlp_1(x))
        

        return x, cache_ele_t if kvcache is not None else None

    def forward(self, x, y, kvcache=None):

        return self.forward_func(x, y, kvcache)




@PLUGIN_LAYERS.register_module()
class TemporalFusion(nn.Module):
    def __init__(self, config, norm_type="layer_norm"):
        super().__init__()
        norm_kwargs = {"bias": config.bias} if norm_type == "layer_norm" else {}

        # Normalization layers
        self.ln_x = get_norm(norm_type, config.n_embd, **norm_kwargs)
        # self.ln_y = get_norm(norm_type, config.n_embd, **norm_kwargs)
        self.ln_mlp_1 = get_norm(norm_type, config.n_embd, **norm_kwargs)

        tm_attn_cfg = (
            dict(type="CausalFlashAttention", n_embd=config.n_embd, n_head=config.n_head, bias=config.bias, dropout=config.dropout, causal=False)
        )
        # 使用 build_from_cfg 构建 temporal_attn
        self.temporal_attn = build_from_cfg(tm_attn_cfg, ATTENTION)
        self.mlp_1 = build_from_cfg(
            dict(type="MLP", config=config), PLUGIN_LAYERS
        )
        sa_attn_cfg = (   
            dict(type="CausalFlashAttention", n_embd=config.n_embd, n_head=config.n_head, bias=config.bias, dropout=config.dropout, causal=False)
        )

        self.self_attn = build_from_cfg(sa_attn_cfg, ATTENTION)
        self.mlp_2 = build_from_cfg(
            dict(type="MLP", config=config), PLUGIN_LAYERS
        )


        self.torch_cp_enabled = False

    def forward_func(self, x, y, kvcache=None):
        # x是当前帧的特征，y是历史帧的特征

        B, T, N, C = x.shape  # x: (batch_size, spatial_size, channels)
        x = torch.cat([x, y], dim=1)    # 在时间维度拼接
        # temporal-attention
        x = rearrange(x, 'b t n c -> (b n) t c')
        attn_out, cache_ele_t = self.temporal_attn(self.ln_x(x), kvcache)
        x = x + attn_out
        x = x + self.mlp_1(self.ln_mlp_1(x))
        # self-attention
        x = rearrange(x, '(b n) t c -> b t n c', b=B, n=N)
        # 取回当前帧的特征
        x = x[:, 0]
        attn_out, cache_ele_t = self.self_attn(self.ln_x(x), kvcache)
        x = x + attn_out
        x = x + self.mlp_2(self.ln_mlp_1(x))
        
        # 恢复到原始维度
        x = rearrange(x, 'b (t n) c -> b t n c', t=T, b=B, n=N)

        return x, cache_ele_t if kvcache is not None else None

    def forward(self, x, y, kvcache=None):

        return self.forward_func(x, y, kvcache)





@PLUGIN_LAYERS.register_module()
class EgoTemporalFusion(nn.Module):
    # 在查询历史信息的时候，不使用 self-attention
    def __init__(self, config, norm_type="layer_norm"):
        super().__init__()
        norm_kwargs = {"bias": config.bias} if norm_type == "layer_norm" else {}

        # Normalization layers
        self.ln_x = get_norm(norm_type, config.n_embd, **norm_kwargs)
        self.ln_y = get_norm(norm_type, config.n_embd, **norm_kwargs)
        self.ln_mlp_1 = get_norm(norm_type, config.n_embd, **norm_kwargs)
        ca_attn_cfg = (
            dict(type="CrossAttention", dim=config.n_embd, num_heads=config.n_head, qkv_bias=config.bias ,causal=False)
        )
        # 使用 build_from_cfg 构建 temporal_attn
        self.cross_attn = build_from_cfg(ca_attn_cfg, ATTENTION)
        # MLP layers
        self.mlp_1 = build_from_cfg(
            dict(type="MLP", config=config), PLUGIN_LAYERS
        )
        self.torch_cp_enabled = False

    def forward_func(self, x, y, kvcache=None):
        B, T, C = x.shape  # x: (batch_size, timesteps, spatial_size, channels)
        # cross-attention
        attn_out, cache_ele_t = self.cross_attn(self.ln_x(x), self.ln_y(y), kvcache)
        x = x + attn_out
        x = x + self.mlp_1(self.ln_mlp_1(x))
        
        return x, cache_ele_t if kvcache is not None else None

    def forward(self, x, y, kvcache=None):

        return self.forward_func(x, y, kvcache)








# ----------------------------------------------------------------------------------MaplikeHead--------------------------------------------------------------------------------------------
@PLUGIN_LAYERS.register_module()
class OffsetLearn3DKeyPointsGenerator(BaseModule): 
    def __init__(
        self,
        embed_dims: int = 256,
        num_sample: int = 20,
        num_learnable_pts: int = 0,
        fix_height: Tuple = (0,),
        ground_height: int = 0,
        full_scene: bool = False,
    ):
        super(OffsetLearn3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_sample = num_sample
        self.num_learnable_pts = num_learnable_pts
        self.num_pts = num_sample * len(fix_height) * num_learnable_pts
        if self.num_learnable_pts > 0:
            self.learnable_fc = Linear(self.embed_dims, self.num_pts * 2)

        self.fix_height = np.array(fix_height)
        self.ground_height = ground_height

        self.full_scene = full_scene

        # self.fix_offset = 

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor,
        instance_feature=None,
        T_cur2temp_list=None,
        cur_timestamp=None,
        temp_timestamps=None,
    ):

        if self.full_scene:
            
            pass

        else:
            assert self.num_learnable_pts > 0, 'No learnable pts'
            bs, num_anchor, _ = anchor.shape
            key_points = anchor.view(bs, num_anchor, self.num_sample, -1)
            fix_keypoints = key_points + zero_offset    # 第一组keypoints保持完全不动 

            offset = (
                self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_sample, len(self.fix_height), self.num_learnable_pts, 2)
            )        
            key_points = offset + key_points[..., None, None, :]
            key_points = torch.cat(
                [
                    key_points,
                    key_points.new_full(key_points.shape[:-1]+(1,), fill_value=self.ground_height),
                ],
                dim=-1,
            )
            fix_height = key_points.new_tensor(self.fix_height)
            height_offset = key_points.new_zeros([len(fix_height), 2])
            height_offset = torch.cat([height_offset, fix_height[:,None]], dim=-1)
            key_points = key_points + height_offset[None, None, None, :, None]
            key_points = key_points.flatten(2, 4)

        return key_points, None







@PLUGIN_LAYERS.register_module()
class NavigationAware(BaseModule): 
    def __init__(
        self,
        mode='sum'
    ):
        super(NavigationAware, self).__init__()
        self.mode = mode


    def forward(
        self,
        query,
        embedding,
    ):

        return query + embedding 





