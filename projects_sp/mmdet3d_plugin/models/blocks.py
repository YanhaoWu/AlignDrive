from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F
from einops import rearrange


from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn.bricks.transformer import FFN
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    FEEDFORWARD_NETWORK,
)
from mmcv.utils import Registry
from projects_sp.mmdet3d_plugin.ops import feature_maps_format
from mmdet.models.utils.transformer import inverse_sigmoid

try:
    from ..ops import deformable_aggregation_function as DAF
except:
    DAF = None
from projects_sp.mmdet3d_plugin.models.attention import get_1d_sincos_pos_embed_from_grid


__all__ = [
    "SpatialTemporalDFA",
    "DeformableFeatureAggregation",
    "DenseDepthNet",
    "AsymmetricFFN",
]


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers









@ATTENTION.register_module()
class SpatialTemporalDFA(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        temporal_fusion_module=None,
        use_temporal_anchor_embed=True,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
        num_temporal=1,     # 使用了几帧历史帧
        query_merage=False, # 是否在query level进行时空特征聚合
        inverse_temporal=False, # 是否在时序上进行逆序, 即先从时序信息中获取anchor再在当前帧获取
    ):
        super(SpatialTemporalDFA, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        if use_deformable_func:
            assert DAF is not None, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        self.query_merage = query_merage
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims
        self.kps_generator_spatial = build_from_cfg(kps_generator, PLUGIN_LAYERS)   # 分别使用当前帧以及历史帧
        # 根据num_temporal来构建多个kps_generator
        self.num_temporal = num_temporal
        # 使用 nn.ModuleList 存储模块
        self.kps_generator_temporal = nn.ModuleList()
        self.temporal_weights_fc = nn.ModuleList()
        self.num_pts = self.kps_generator_spatial.num_pts
        self.inverse_temporal=inverse_temporal
        for i in range(num_temporal):

            if self.query_merage:
                kps_gen = Linear(self.embed_dims, 3)    # 是学习offset
                self.kps_generator_temporal.append(kps_gen)
            else:
                # 创建 kps_generator_temporal 并添加到 ModuleList
                kps_gen = build_from_cfg(kps_generator, PLUGIN_LAYERS)
                self.kps_generator_temporal.append(kps_gen)

            # 创建 temporal_weights_fc 并添加到 ModuleList
            fc_layer = nn.Linear(embed_dims, num_groups * num_levels * self.num_pts)
            self.temporal_weights_fc.append(fc_layer)

        
        # Linear压缩时空特征
        self.sp_fc = Linear(embed_dims * (num_temporal + 1), embed_dims)

        if temporal_fusion_module is not None:
            if "embed_dims" not in temporal_fusion_module:
                temporal_fusion_module["embed_dims"] = embed_dims
            self.temp_module = build_from_cfg(
                temporal_fusion_module, PLUGIN_LAYERS
            )
        else:
            self.temp_module = None
        self.output_proj = Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            self.spatial_weights_fc = Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.spatial_weights_fc = Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )
        # self.init_weight()
    def init_weight(self):
        constant_init(self.spatial_weights_fc, val=0.0, bias=0.0)
        constant_init(self.temporal_weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)


    def aggregate_spatial_feature(self, instance_feature, feature_maps, anchor, anchor_embed, metas):

        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator_spatial(anchor, instance_feature)      # 里面有可学习的offset model
        weights = self._get_weights(instance_feature, anchor_embed, weights_fc=self.spatial_weights_fc, metas=metas)

        if self.use_deformable_func:
            points_2d = (
                self.project_points(
                    key_points,
                    metas["projection_mat"],
                    metas.get("image_wh"),
                )
                .permute(0, 2, 3, 1, 4)
                .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
            )
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )
            features = DAF(*feature_maps, points_2d, weights).reshape(
                bs, num_anchor, self.embed_dims
            )
        else:
            features = self.feature_sampling(
                feature_maps,
                key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
            )
            features = self.multi_view_level_fusion(features, weights)
            features = features.sum(dim=2)  # fuse multi-point features
            points_2d = None

        return features, key_points




    def aggregate_temporal_feature(self, instance_feature, temporal_feature_map, anchor, anchor_embed, temporal_metas):

        bs, num_anchor = instance_feature.shape[:2]
        len_temporal = len(temporal_feature_map)
        temporal_features = []
        key_points_all = []
        for i in range(len_temporal):
            feature_maps = temporal_feature_map[i]
            metas = temporal_metas[i]
            key_points = self.kps_generator_temporal[i](anchor, instance_feature)
            weights = self._get_weights(instance_feature, anchor_embed, weights_fc=self.temporal_weights_fc[i], metas=metas)

            if self.use_deformable_func:
                points_2d = (
                    self.project_points(
                        key_points,
                        metas["projection_mat"],
                        metas.get("image_wh"),
                    )
                    .permute(0, 2, 3, 1, 4)
                    .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
                )
                weights = (
                    weights.permute(0, 1, 4, 2, 3, 5)
                    .contiguous()
                    .reshape(
                        bs,
                        num_anchor,
                        self.num_pts,
                        self.num_cams,
                        self.num_levels,
                        self.num_groups,
                    )
                )
                features = DAF(*feature_maps, points_2d, weights).reshape(
                    bs, num_anchor, self.embed_dims
                )
            else:
                features = self.feature_sampling(
                    feature_maps,
                    key_points,
                    metas["projection_mat"],
                    metas.get("image_wh"),
                )
                features = self.multi_view_level_fusion(features, weights)
                features = features.sum(dim=2)  # fuse multi-point features
                points_2d = None
            temporal_features.append(features)
            key_points_all.append(key_points)
        return temporal_features, key_points_all



    def aggregate_sptemporal_feature(self, instance_feature, spatial_feature_map, temporal_feature_map, anchor, anchor_embed, current_metas, temporal_metas, debug=False):
        # 在query level进行时空特征聚合
        bs, num_anchor = instance_feature.shape[:2]
        # 首先聚合当前帧的空间特征，为每一个anchor更新特征
        sp_features = []
        key_points_all = []
        key_points = self.kps_generator_spatial(anchor, instance_feature)      # 里面有可学习的offset model
        key_points_all.append(key_points)
        weights = self._get_weights(instance_feature, anchor_embed, weights_fc=self.spatial_weights_fc, metas=current_metas)
        spatial_feature_map = feature_maps_format(spatial_feature_map, inverse=True)[0]
        features = self.feature_sampling(
            spatial_feature_map,
            key_points,
            current_metas["projection_mat"],
            current_metas.get("image_wh"),
        )
        keypoints_features = self.multi_view_level_fusion(features, weights)      # 获得的是每个anchor在当前帧进行特征采样之后的特征
        # 在Temporal的feature Map进行Sparse aggregation
        sp_features.append(keypoints_features)
        len_temporal = len(temporal_feature_map)
        for i in range(len_temporal):
            feature_maps = temporal_feature_map[i]
            metas = temporal_metas[i]
            temporal_offset = self.kps_generator_temporal[i](keypoints_features)
            key_points = key_points + temporal_offset
            weights = self._get_weights(instance_feature, anchor_embed, weights_fc=self.temporal_weights_fc[i], metas=metas)
            feature_maps = feature_maps_format(feature_maps, inverse=True)[0]
            features = self.feature_sampling(
                feature_maps,
                key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
            )
            keypoints_features = self.multi_view_level_fusion(features, weights)
            sp_features.append(keypoints_features)
            key_points_all.append(key_points)
        sp_features = torch.cat(sp_features, dim=-1)
        features = self.sp_fc(sp_features)
        features = features.sum(dim=2)  # fuse multi-point features


        if debug:
            # 计算instance feature与keypoints的相似度并归一化
            pass

        return features, key_points_all



    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        current_metas: dict,
        temporal_feature_map: dict,
        temporal_metas: dict,
        **kwargs: dict,
    ):  
        similarity = None
        debug=kwargs.get("debug", False)
        if self.query_merage:
            sptemporal_features, keypoints = self.aggregate_sptemporal_feature(instance_feature, feature_maps, temporal_feature_map, anchor, anchor_embed, current_metas, temporal_metas, debug=debug)    
            features = sptemporal_features
        else:
            if self.inverse_temporal:
                temporal_features, key_points_temporal  = self.aggregate_temporal_feature(instance_feature, temporal_feature_map, anchor, anchor_embed, temporal_metas)
                # 在当前的feature Map进行Sparse aggregation
                spatial_features, key_points_sptial = self.aggregate_spatial_feature(temporal_features[-1], feature_maps, anchor, anchor_embed, current_metas)
            else:
                # 在当前的feature Map进行Sparse aggregation
                spatial_features, key_points_sptial = self.aggregate_spatial_feature(instance_feature, feature_maps, anchor, anchor_embed, current_metas)
                # 在Temporal的feature Map进行Sparse aggregation
                temporal_features, key_points_temporal  = self.aggregate_temporal_feature(spatial_features, temporal_feature_map, anchor, anchor_embed, temporal_metas)
            # 与历史信息进行信息聚合
            temporal_features = torch.cat(temporal_features, dim=-1)   # B,T,N,C
            sptemporal_features = torch.cat([spatial_features, temporal_features], dim=-1)
            features = self.sp_fc(sptemporal_features)

            # 暂时只保留当前帧的keypoints为了可视化
            keypoints = key_points_sptial



        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)

        return output, keypoints, similarity

    def _get_weights(self, instance_feature, anchor_embed, weights_fc=None, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                )
            )
            feature = feature[:, :, None] + camera_embed[:, None]

        weights = (
            weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        # Homogeneous divide in fp32: fp16 + tiny depth yields NaN in DivBackward0.
        depth = torch.clamp(points_2d[..., 2:3].float(), min=1e-5)
        points_2d = points_2d[..., :2].float() / depth
        points_2d = points_2d.to(dtype=pts_extend.dtype)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)        # (2*6, 18, 90, 2)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features




















































@ATTENTION.register_module()
class ObjectDFA(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        num_pts: int = -1, 
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        temporal_fusion_module=None,
        use_temporal_anchor_embed=True,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
    ):
        super(ObjectDFA, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        if use_deformable_func:
            assert DAF is not None, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        if kps_generator is not None:
            kps_generator["embed_dims"] = embed_dims
            self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)
            self.num_pts = self.kps_generator.num_pts
        else:
            self.kps_generator = None
            self.num_pts = num_pts

        # 融合多视图特征
        hidden_dim = int(embed_dims*num_cams/2)
        self.mutil_cam_fusion = nn.Sequential(
            nn.Linear(embed_dims*num_cams, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dims)
        )

        # 融合多视角特征
        hidden_dim = int(embed_dims*num_levels/2)

        self.mutil_level_fusion = nn.Sequential(
            nn.Linear(embed_dims*num_levels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dims)
        )

        self.output_proj = Linear(embed_dims, embed_dims)


    def init_weight(self):
        # constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        **kwargs: dict,
    ):
        feature_maps = feature_maps_format(feature_maps, inverse=True)[0]
        given_keypoints = kwargs.get("given_keypoints", None)
        key_points = given_keypoints
        
        sampling_features_list = []
        for keys in given_keypoints.keys():
            given_keypoint = given_keypoints[keys]
            sampling_features = self.feature_fusion(
                feature_maps,
                given_keypoint,
                metas
            )
            sampling_features_list.append(sampling_features)
        sampling_features = torch.cat(sampling_features_list, dim=1)
        return sampling_features, key_points



    def feature_fusion(
        self,
        feature_maps: torch.Tensor,
        key_points: torch.Tensor,
        metas: dict,
    ):

        features = self.feature_sampling(
            feature_maps,
            key_points,
            metas["projection_mat"],
            metas.get("image_wh"), 
        ) # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        # 先做多视角特征融合
        features = features.permute(0, 1, 3, 4, 2, 5) # bs, num_anchor, num_levels, num_pts, num_cams, embed_dims
        features = features.reshape(
            features.shape[0], features.shape[1], features.shape[2], features.shape[3], -1
        )
        features = self.mutil_cam_fusion(features) # bs, num_anchor, num_levels, num_pts, embed_dims
        # 进行多层级特征融合
        features = features.permute(0, 1, 3, 2, 4) # bs, num_anchor, num_pts, num_levels, embed_dims
        features = features.reshape(
            features.shape[0], features.shape[1], features.shape[2], -1
        )
        features = self.mutil_level_fusion(features) # bs, num_anchor, num_pts, embed_dims
        features = features.reshape(features.shape[0], -1, features.shape[3]) # bs, total_num_pts, embed_dims
        
        features = self.proj_drop(self.output_proj(features))

        return features



    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                )
            )
            feature = feature[:, :, None] + camera_embed[:, None]

        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        # Homogeneous divide in fp32: fp16 + tiny depth yields NaN in DivBackward0.
        depth = torch.clamp(points_2d[..., 2:3].float(), min=1e-5)
        points_2d = points_2d[..., :2].float() / depth
        points_2d = points_2d.to(dtype=pts_extend.dtype)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features























@ATTENTION.register_module()
class DeformableFeatureAggregation(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        num_pts: int = -1, 
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        temporal_fusion_module=None,
        use_temporal_anchor_embed=True,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        if use_deformable_func:
            assert DAF is not None, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        if kps_generator is not None:
            kps_generator["embed_dims"] = embed_dims
            self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)
            self.num_pts = self.kps_generator.num_pts
        else:
            self.kps_generator = None
            self.num_pts = num_pts

        if temporal_fusion_module is not None:
            if "embed_dims" not in temporal_fusion_module:
                temporal_fusion_module["embed_dims"] = embed_dims
            self.temp_module = build_from_cfg(
                temporal_fusion_module, PLUGIN_LAYERS
            )
        else:
            self.temp_module = None
        self.output_proj = Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            self.weights_fc = Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        **kwargs: dict,
    ):
        given_keypoints = kwargs.get("given_keypoints", None)
        if given_keypoints is not None:
            key_points = given_keypoints
        else:
            bs, num_anchor = instance_feature.shape[:2]
            key_points = self.kps_generator(anchor, instance_feature)   # (B, anchor, pts, 3)
            weights = self._get_weights(instance_feature, anchor_embed, metas)

        if self.use_deformable_func:
            points_2d = (
                self.project_points(
                    key_points,
                    metas["projection_mat"],
                    metas.get("image_wh"),
                )
                .permute(0, 2, 3, 1, 4)
                .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
            )
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )
            features = DAF(*feature_maps, points_2d, weights).reshape(
                bs, num_anchor, self.embed_dims
            )
        else:
            features = self.feature_sampling(
                feature_maps,
                key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
            )
            features = self.multi_view_level_fusion(features, weights)
            features = features.sum(dim=2)  # fuse multi-point features
            points_2d = None
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)

        return output, key_points

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                )
            )
            feature = feature[:, :, None] + camera_embed[:, None]

        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        # Homogeneous divide in fp32: fp16 + tiny depth yields NaN in DivBackward0.
        depth = torch.clamp(points_2d[..., 2:3].float(), min=1e-5)
        points_2d = points_2d[..., :2].float() / depth
        points_2d = points_2d.to(dtype=pts_extend.dtype)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features


@PLUGIN_LAYERS.register_module()
class DenseDepthNet(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_depth_layers=1,
        equal_focal=100,
        max_depth=60,
        loss_weight=1.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        self.depth_layers = nn.ModuleList()
        for i in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, feature_maps, focal=None, gt_depths=None):
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            bs = feat.shape[0]
            num_cam = feat.shape[1]
            depth = self.depth_layers[i](feat.flatten(end_dim=1).float()).exp()
            depth = depth.transpose(0, -1) * focal / self.equal_focal
            depth = depth.transpose(0, -1).contiguous()
            depth = rearrange(
                depth, "(b n) t h w -> b (n t) h w", b = bs, n=num_cam
            )
            depths.append(depth)
        if gt_depths is not None and self.training:
            loss = self.loss(depths, gt_depths)
            return loss
        return depths

    def loss(self, depth_preds, gt_depths):
        loss = 0.0
        if not isinstance(gt_depths, list):
            gt_depths = [gt_depths]
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)
            fg_mask = torch.logical_and(
                gt > 0.0, torch.logical_not(torch.isnan(pred))
            )
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)
            with autocast(enabled=False):
                error = torch.abs(pred - gt).sum()
                _loss = (
                    error
                    / max(1.0, len(gt) * len(depth_preds))
                    * self.loss_weight
                )
            loss = loss + _loss
        return loss


@FEEDFORWARD_NETWORK.register_module()
class AsymmetricFFN(BaseModule):
    def __init__(
        self,
        in_channels=None,
        pre_norm=None,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(AsymmetricFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        if in_channels is None:
            in_channels = embed_dims
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = (
            build_dropout(dropout_layer)
            if dropout_layer
            else torch.nn.Identity()
        )
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if in_channels == embed_dims
                else Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)


@PLUGIN_LAYERS.register_module()
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embd, 4 * config.n_embd, bias=config.bias
        )
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            4 * config.n_embd, config.n_embd, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x



@PLUGIN_LAYERS.register_module()
class temporal_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.n_embd
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

    def forward(self, x):   
        x = self.lstm(x)
        return x




# GAN loss For Trajectory Prediction
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x













@PLUGIN_LAYERS.register_module()
class GlobalFeatureExtractor(BaseModule):
    def __init__(
        self,
        embed_dims,
        feature_map_index,
        add_3dpositional_encoding=True,
        image_feature_process='MLP',    # 'MLP' or 'TokenLearner'   
    ):
        super(GlobalFeatureExtractor, self).__init__()
        self.pos3d = PETRPositionEmbedder(with_fpe=False, with_multiview=True, embed_dims=embed_dims, feature_level=-1, img_h=256, img_w=704)   # 这个feature_level似乎没用上
        self.feature_map_index = feature_map_index
        self.add_3dpositional_encoding = add_3dpositional_encoding
        self.add_seq_embedding = True
        self.num_cam = 6
        self.n_embd = embed_dims

        IMAGE_FEATURE_MAP_SIZE = [[64, 176], [32, 88], [16, 44], [8, 22]]   # 多尺度特征图每一层的数量
        self.num_token_per_feature = 0
        for index in self.feature_map_index:
            self.num_token_per_feature += IMAGE_FEATURE_MAP_SIZE[index][0] * IMAGE_FEATURE_MAP_SIZE[index][1]    

        n_pos = self.num_token_per_feature * len(self.feature_map_index) * self.num_cam
        self.img_pos =  nn.Parameter(torch.zeros(n_pos, embed_dims), requires_grad=False)
        pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dims, np.arange(n_pos))
        self.img_pos.data.copy_(torch.from_numpy(pos_embed).float())


    # def init_weight(self):
    #     constant_init(self.weights_fc, val=0.0, bias=0.0)
    #     xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def add_position_embedding(self, x):
        seq_len = x.shape[1]
        device = x.device
        pos = torch.arange(seq_len, device=device).unsqueeze(0)    # 增加batch和token维度
        pos = self.img_pos[pos]
        x = x + pos
        return x



    def forward(
        self,
        feature_maps: List[torch.Tensor],
        metas: dict,
        **kwargs: dict,
    ):
        feature_maps = feature_maps_format(feature_maps, inverse=True)[0]
        selected_feature_map = []
        img_metas = metas["img_metas"]
        for i in self.feature_map_index:
            if self.add_3dpositional_encoding:
                position_aware_feature = feature_maps[i] + self.pos3d(feature_maps[i], img_metas)
            else:
                position_aware_feature = feature_maps[i]
            selected_feature_map.append(position_aware_feature)

        feature_tokens = []
        for i in self.feature_map_index:
                # tokens = selected_feature_map[i].view(selected_feature_map[i].shape[0], selected_feature_map[i].shape[1], -1)
            tokens = selected_feature_map[i].permute(0, 1, 3, 4, 2).contiguous()  # B,N,C,H,W -> B,N,H,W,C
            tokens = tokens.view(selected_feature_map[i].shape[0], self.num_cam, -1, self.n_embd)
            feature_tokens.append(tokens)   
        feature_tokens = torch.cat(feature_tokens, dim=2)  # B, N_cam, N_token, C
        feature_tokens = rearrange(feature_tokens, 'B N T C -> B (N T) C')
        # 添加位置编码
        if self.add_seq_embedding:  # 原则上都要加，除非要做特殊的可视化
            feature_tokens = self.add_position_embedding(feature_tokens)
        
        return feature_tokens
