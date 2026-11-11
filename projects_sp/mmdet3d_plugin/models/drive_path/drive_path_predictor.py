from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import random

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
from mmdet.models import build_loss  

from ..blocks import DeformableFeatureAggregation as DFG
from ..blocks import linear_relu_ln
from ..instance_bank import topk

from projects_sp.mmdet3d_plugin.models.attention import gen_sineembed_for_position
from projects_sp.mmdet3d_plugin.models.instance_bank import Img_instance_queue
from projects_sp.mmdet3d_plugin.models.blocks import Discriminator
from projects_sp.mmdet3d_plugin.models.utils import check_collision_with_gt

from projects_sp.mmdet3d_plugin.models.longitudinal_planning.score_module.compute_b2d_score_wyh import compute_corners_torch

from projects_sp.mmdet3d_plugin.models.attention import get_1d_sincos_pos_embed_from_grid
from einops.einops import rearrange
from projects_sp.mmdet3d_plugin.core.box3d import *

__all__ = ["DrivePathPredictor"]

# Fuse longitudinal context with historical trajectories and current-frame decisions.
@HEADS.register_module()
class DrivePathPredictor(BaseModule):
    def __init__(
        self,
        use_ego_status: bool,
        use_previous_trajectory: bool,
        graph_model: dict,
        graph_model_no_distance: dict,
        img_graph_model: dict,
        temp_graph_model: dict,
        TemporalMixer: dict,
        norm_layer: dict,
        ffn: dict,
        refine_layer: dict,
        instance_bank: dict,
        anchor_encoder: dict,
        operation_order: List[str],
        num_det,
        num_map,
        longitudinal_layer_range=[0,1,2,3,4,5],
        p_topk=1,
        embed_dims=256,
        decode_perception_query=False,
        augment_bbox_pro=0.0,
        merage_traffic_light='none',
        img_encoder=None,
        use_gt_anchor=False, # 2阶段使用的anchor，是否使用GT
        longitudinal_train_use_grand=None,
        mutil_step_query=7,
        specialsign_module=None,
        percept_decoded_map=False,
        HistoryPlanBank=None,
        class_names=None,
        # drive_path_encode=True,
    ):
        super(DrivePathPredictor, self).__init__()
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.embed_dims = embed_dims
        self.use_ego_status = use_ego_status
        self.use_previous_trajectory = use_previous_trajectory
        self.decode_perception_query = decode_perception_query
        self.instance_bank = build(instance_bank, PLUGIN_LAYERS)
        self.anchor_encoder = build(anchor_encoder, POSITIONAL_ENCODING)
        self.plan_num_det = num_det
        self.plan_num_map = num_map
        self.augment_bbox_pro = augment_bbox_pro
        self.mutil_step_query = self.instance_bank.mutil_step_query
        self.use_gt_anchor = use_gt_anchor
        self.percept_decoded_map = percept_decoded_map
        self.class_names = class_names
        self.merage_traffic_light = merage_traffic_light
        # Always a list so find_traffic_lights() can safely len()-iterate.
        # None previously crashed when decode_perception_query_longitudinal=True but
        # class_names was omitted from the DrivePathPredictor config.
        self.traffic_light_idx = []
        if self.class_names is not None:
            for i in range(len(self.class_names)):
                if "traffic_light" in self.class_names[i]:
                    self.traffic_light_idx.append(i)
        # self.drive_path_encode = drive_path_encode
        # if self.drive_path_encode:
        #     self.drivepath_encoder = nn.Sequential(
        #         nn.Linear(96, self.embed_dims), # 96是 16*6，每个坐标点用16维表示，一共6个坐标点
        #         nn.ReLU(),
        #     )
        self.longitudinal_train_use_grand = longitudinal_train_use_grand
        if longitudinal_train_use_grand is None:
            self.longitudinal_train_use_grand = dict(
                temporal='temporal_interval_5_traj',
                spatial='spatial_interval_5_traj',
                augmentation='fake_defensive_eg_fut_trajs_5hz',
                start_iter=9999999,
            )
        self.iter_count = 0 


        if self.use_ego_status:
            # self.egp_status_encoder = nn.Sequential(
            #     *linear_relu_ln(self.embed_dims, 1, 1, 4),
            #     nn.Linear(self.embed_dims, self.embed_dims),
            # )
            posi_embedding = get_1d_sincos_pos_embed_from_grid(self.embed_dims, torch.arange(1024))
            self.egp_status_codebook = nn.Parameter(torch.zeros(1024, self.embed_dims), requires_grad=True)
            self.egp_status_codebook.data.copy_(torch.from_numpy(posi_embedding).float())

            self.egp_status_encoder = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(),
            )
        
        if self.merage_traffic_light != 'none':
            
            traffic_class_embedding = get_1d_sincos_pos_embed_from_grid(16, torch.arange(4))    # 4是交通灯的状态，无效, 红灯, unknown, 黄灯
            self.traffic_class_embedding = nn.Parameter(
                torch.tensor(traffic_class_embedding, dtype=torch.float32),
                requires_grad=True
            )

            self.traffic_light_proj = nn.Sequential(
                nn.Linear(16, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims)
            )






        # 轨迹编码相关
        # posi_embedding = get_1d_sincos_pos_embed_from_grid(self.embed_dims, torch.arange(7))
        # self.temporal_embed = nn.Parameter(torch.zeros(7, self.embed_dims), requires_grad=False)
        # self.temporal_embed.data.copy_(torch.from_numpy(posi_embedding).float())
        if self.decode_perception_query:

            if self.mutil_step_query == 1:
            # 如果是单步预测，那么输入维度就是 base_dim
                input_dim = 7 * 64  # 448 = 7 * 64.  64是每个时间步的特征维度, 7是self.mutil_step_query + 1, 包含一个当前时间步
            else:
                input_dim = 64


            self.traj_encoder = nn.Sequential(
                nn.Linear(input_dim, self.embed_dims),
                nn.ReLU(),
            )


            agent_position_embedding = get_1d_sincos_pos_embed_from_grid(self.embed_dims, torch.arange(30))
            self.register_buffer('agent_position_embedding', torch.tensor(agent_position_embedding, dtype=torch.float32), persistent=False)

        if HistoryPlanBank is not None:
            self.HistoryPlanBank = build(HistoryPlanBank, PLUGIN_LAYERS)
        else:
            self.HistoryPlanBank = None



        if self.mutil_step_query > 1:
            temporal_position_embed = get_1d_sincos_pos_embed_from_grid(self.embed_dims, torch.arange(self.mutil_step_query))
            self.register_buffer('temporal_position_embed', torch.tensor(temporal_position_embed, dtype=torch.float32), persistent=False)

        agents_class_embedding = get_1d_sincos_pos_embed_from_grid(self.embed_dims, torch.arange(12))    # 12是当前检测的数量，后面要变
        self.agents_class_embedding = torch.tensor(agents_class_embedding, dtype=torch.float32)
        # 封装为 nn.Parameter
        self.agents_class_embedding = nn.Parameter(self.agents_class_embedding, requires_grad=True)



        # 额外的图像编码头
        if img_encoder is not None:
            self.img_encoder = build(img_encoder, PLUGIN_LAYERS)
        else:
            self.img_encoder = None

        self.op_config_map = {
            "temp_cross_gnn": [graph_model, ATTENTION],
            "temporal_gnn": [graph_model, ATTENTION],
            "cross_gnn": [graph_model, ATTENTION],
            "reference_cross_gnn": [graph_model_no_distance, ATTENTION],
            "map_cross_gnn": [graph_model, ATTENTION],
            "cross_norm": [norm_layer, NORM_LAYERS],
            "gnn": [graph_model, ATTENTION],
            "temp_cross_norm": [norm_layer, NORM_LAYERS],
            "reference_cross_norm": [norm_layer, NORM_LAYERS],
            "mutil_step_fusion": [temp_graph_model, PLUGIN_LAYERS],
            "cross_norm": [norm_layer, NORM_LAYERS],
            "gnn_norm": [norm_layer, NORM_LAYERS],
            "mixer": [TemporalMixer, PLUGIN_LAYERS],
            "refine": [refine_layer, PLUGIN_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "image_cross_gnn": [img_graph_model, ATTENTION],
            "cross_norm": [norm_layer, NORM_LAYERS],
            "image_norm": [norm_layer, NORM_LAYERS],
            "SpecialSignAware": [graph_model_no_distance, ATTENTION],
        }
        self.operation_order_layer = operation_order.copy()

        if isinstance(operation_order[0], List):
            operation_order = [item for sublist in operation_order for item in sublist]
        self.operation_order = operation_order

        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )

        self.prediction = []
        self.scores = []
        self.quality = []
        self.predicted_distance = []
        self.longitudinal_layer_range = longitudinal_layer_range


    def init_weights(self):
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()


    def reset_output(self):
        self.prediction = []
        self.scores = []
        self.quality = []
        self.predicted_distance = []



    def obtain_image_tokens(self, img_features, metas):
        # 将图像特征转换为tokens
        if self.img_encoder is not None:
            img_tokens = self.img_encoder(img_features, metas)
        else:
            img_tokens = None
        
        return img_tokens       # 已加序列位置编码，空间3D编码取决于设定







    def forward(self, det_features, map_features, plan_features, layer_index, metas, ignore_query=False):
        start_index = 0
        for j in range(layer_index):
            start_index += len(self.operation_order_layer[j])
        operation_order = self.operation_order_layer[layer_index]
        instance_feature = plan_features['instance_feature']  # B, N, T, D
        anchor_embed = plan_features['anchor_embed']  # B, N, D
        temp_instance_feature = plan_features.get('temporal_instance_feature', None)  # B, N, T, D
        temp_instance_anchor = plan_features.get('temporal_anchor', None)  # B, N, T, D

        batch_size = instance_feature.shape[0]
        for index, op in enumerate(operation_order):
            # 因为instance_feature是在外面才更新进plan_features，所以在这个函数里面，不要轻易使用plan_features['instance_feature'],否则容易用错
            i = start_index + index
            if self.layers[i] is None:
                continue
            
            elif op == "temporal_gnn":
                num_instance = instance_feature.shape[1]
                num_horzion = instance_feature.shape[2]
                instance_feature = rearrange(instance_feature, 'b n t d -> (b n) t d')  # 避免不同模态之间交互



                anchor_embed = anchor_embed.reshape(
                    batch_size*num_instance, 1, self.embed_dims
                )
                if temp_instance_feature is not None:
                    temp_instance_feature = temp_instance_feature.reshape(
                        batch_size*num_instance, self.num_temporal_plan, self.embed_dims
                    )
                    temp_anchor_embed = temp_anchor_embed.reshape(
                        batch_size*num_instance, self.num_temporal_plan, self.embed_dims
                    )
                if attn_mask is not None:
                    attn_mask = attn_mask.reshape(
                        batch_size*num_instance, batch_size*num_instance
                    )
                    print("UNIMPLEMENTED for attn_mask")

    
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

                # 转换回原来的形状
                instance_feature = instance_feature.reshape(
                    batch_size, num_instance, self.embed_dims
                )
                anchor_embed = anchor_embed.reshape(
                    batch_size, num_instance, self.embed_dims
                )
                if temp_instance_feature is not None:
                    temp_anchor_embed = temp_anchor_embed.reshape(
                        batch_size, num_instance, self.num_temporal_plan, self.embed_dims
                    )
                    temp_instance_feature = temp_instance_feature.reshape(
                        batch_size, num_instance, self.num_temporal_plan, self.embed_dims
                    )

            elif op == "SpecialSignAware":

                # instance_feature = self.percept_traffic_lights(instance_feature, anchor_embed, det_features, index)  # 因为percept_traffic_lights需要用到plan_features['instance_feature']
               
               
                traffic_light_state = det_features['longitudinal_light_state_embedding'].unsqueeze(1).unsqueeze(1)  # B, 1, 1, D
                traffic_light_state = traffic_light_state.repeat(1, instance_feature.shape[1], 1, 1)  # B, N, T, D
                traffic_light_state = self.traffic_light_proj(traffic_light_state)


                instance_feature = rearrange(instance_feature, 'b n t d -> (b n) t d')  # 避免不同模态之间交互
                traffic_light_state = rearrange(traffic_light_state, 'b n t d -> (b n) t d')  # 避免不同模态之间交互

                instance_feature = self.layers[i](
                    instance_feature,
                    key=traffic_light_state,
                    value=traffic_light_state,
                )
                instance_feature = rearrange(instance_feature, '(b n) t d -> b n t d', b=bs, n=num_modal)


            elif op == "reference_cross_gnn":
                # 与Traffic Light交互
                if self.merage_traffic_light == 'add' or self.merage_traffic_light == 'add_res':
                    # 如果需要合并交通灯信息
                    traffic_light_state = det_features['longitudinal_light_state_embedding'].unsqueeze(1).unsqueeze(1)  # B, 1, 1, D
                    traffic_light_state = self.traffic_light_proj(traffic_light_state)
                    instance_feature = instance_feature + traffic_light_state  # B, N, T, D


                # 与参考的drivepath进行交互
                bs = det_features['instance_feature'].shape[0]  # 获取batch size
                num_modal = instance_feature.shape[1]  # 获取模态数
                num_t = instance_feature.shape[2]  # 获取时间步数
                drive_path_embed = plan_features['drive_path_embed']    # B, N, 256
                drive_path_embed = rearrange(drive_path_embed, 'b n d -> (b n) d').unsqueeze(1)  # (B N) T D
                instance_feature = rearrange(instance_feature, 'b n t d -> (b n) t d')  # 避免不同模态之间交互
                if self.longitudinal_train_use_grand.get('instance_feature', False):
                    reference_feature = plan_features['reference_feature']  # B, N, D
                    reference_feature = rearrange(reference_feature, 'b n d -> (b n) d').unsqueeze(1)  # (B N) T D
                    drive_path_embed = reference_feature + drive_path_embed  # 将参考特征与drive path特征相加
                    
                instance_feature = self.layers[i](
                    instance_feature,
                    key=drive_path_embed,
                    value=drive_path_embed,
                )
                instance_feature = rearrange(instance_feature, '(b n) t d -> b n t d', b=bs, n=num_modal)
            elif op == "map_cross_gnn":
                plan_features['instance_feature'] = instance_feature    # 因为cross_gnn_decode_query需要用到plan_features['instance_feature']
                instance_feature = self.perception_cross_gnn(
                    i,
                    layer_index,
                    map_features,
                    plan_features,
                    ignore_query=ignore_query,
                )
                # 更新instance feature
                plan_features['instance_feature'] = instance_feature

            elif op == "cross_gnn":
                if self.decode_perception_query:
                    plan_features['instance_feature'] = instance_feature    # 因为cross_gnn_decode_query需要用到plan_features['instance_feature']
                    instance_feature = self.cross_gnn_decode_query(
                        i,
                        det_features,
                        map_features,
                        plan_features,
                        ignore_query=ignore_query,
                    )
                else:
                    instance_feature = self.cross_gnn(
                        i,
                        det_features,
                        map_features,
                        plan_features,
                        ignore_query=ignore_query,
                    )
            elif op == "mutil_step_fusion":
                anchor_embedding = plan_features['anchor_embed']
                instance_feature = self.temporal_fusion(i, instance_feature, anchor_embedding)

            elif op == "image_cross_gnn":
                plan_features['instance_feature'] = instance_feature
                instance_feature = self.cross_gnn_with_image(
                    i,
                    plan_features,
                )
            elif op == "image_norm":
                image_tokens = plan_features['img_tokens']
                image_tokens = self.layers[i](image_tokens)
                plan_features['img_tokens'] = image_tokens

            elif op == "cross_norm" or op=='ffn' or op == 'reference_cross_norm':

                instance_feature = self.layers[i](instance_feature)

            elif op == "refine":
                # if self.use_ego_status:
                #     ego_status = metas['ego_status_digit']
                #     ego_status_embedding = self.egp_status_encoder(
                #         self.egp_status_codebook[ego_status]
                #     )
                #     meraged_instance_feature = instance_feature + ego_status_embedding.unsqueeze(1)
                # else:

                if self.merage_traffic_light == 'res' or self.merage_traffic_light == 'add_res':
                    traffic_light_state = det_features['longitudinal_light_state_embedding'].unsqueeze(1).unsqueeze(1)  # B, 1, 1, D
                    traffic_light_state = self.traffic_light_proj(traffic_light_state)
                    instance_feature = instance_feature + traffic_light_state  # B, N, T, D

                meraged_instance_feature = instance_feature

                anchor_embedding = plan_features['anchor_embed']
                anchor = plan_features['anchor']

                anchor, score, qt, predicted_distance = self.layers[i](
                    meraged_instance_feature,
                    # instance_feature,
                    anchor,
                    anchor_embedding,
                    drivepath=plan_features['drivepath'],
                    return_cls=True,
                )
                anchor_embed = self.anchor_encoder(anchor)             

                self.prediction.append(anchor)
                self.scores.append(score)
                self.quality.append(qt)
                self.predicted_distance.append(predicted_distance)
            else:
                raise ValueError(f"Unsupported operation: {op}")


        output = {
            "scores": self.scores,
            "prediction": self.prediction,
            "quality": self.quality,
            "predicted_distance": self.predicted_distance,
            "instance_feature": instance_feature,
            "anchor":anchor,
            "anchor_embed": anchor_embed,
            "score": score,
        }



        return output

    def update_instance_bank(self, instance_feature, anchor, confidence, metas=None):
        if self.HistoryPlanBank is not None:
            # 更新历史轨迹库
            self.HistoryPlanBank.cache(instance_feature, anchor, confidence, metas=metas)











    def update_output(self,  prediction, scores, drive_path, predicted_distance, prefix='longitudinal'):
        output = {}
        output[prefix + '_scores'] = scores
        output[prefix + '_prediction'] = prediction
        output[prefix + '_drivepath'] = drive_path
        output[prefix + '_predicted_distance'] = predicted_distance
        self.iter_count += 1
        return output

    # copy from detection head
    def encode_reg_target(self, box_target, device=None):
        outputs = []
        for box in box_target:
            output = torch.cat(
                [
                    box[..., [X, Y, Z]],
                    box[..., [W, L, H]].log(),
                    torch.sin(box[..., YAW]).unsqueeze(-1),
                    torch.cos(box[..., YAW]).unsqueeze(-1),
                    box[..., YAW + 1 :],
                ],
                dim=-1,
            )
            if device is not None:
                output = output.to(device=device)
            outputs.append(output)
        return outputs

    # copy from detection decoder
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



    def augment_bbox(self, trajs, bboxes, fake_trajs, fake_bboxes, bbox_class, valid_flag=False, given_index=None):
        # 使用fake_bboxes和fake_trajs替换掉bboxes和trajs中的一个
        # 1. 丢弃最后一个位置的bbox和轨迹
        # 2. 将fake结果随机插入到任意位置（不一定是最后一个）
        predefined_class = 0    # car


        if valid_flag:
            # 1. 丢弃最后一个元素（使用切片）
            bboxes = bboxes[:-1]  # shape: [N-1, ...]
            trajs = trajs[:-1]    # shape: [N-1, ...]
            bbox_class = bbox_class[:-1]  # shape: [N-1, ...]
            
            # 2. 随机选择一个插入位置（0到N-1+1，即可能插到末尾）
            if given_index is None:
                insert_idx = torch.randint(0, len(bboxes) + 1, (1,)).item()
            else:
                insert_idx = given_index
            
            # 3. 插入fake数据（torch.cat方式）
            bboxes = torch.cat([
                bboxes[:insert_idx],
                fake_bboxes,  # 确保fake_bboxes是 [1, ...] 形状
                bboxes[insert_idx:]
            ], dim=0)
            
            trajs = torch.cat([
                trajs[:insert_idx],
                fake_trajs.unsqueeze(0),  # 确保fake_trajs是 [1, ...] 形状
                trajs[insert_idx:]
            ], dim=0)

            bbox_class = torch.cat([
                bbox_class[:insert_idx],
                torch.full((1,), predefined_class, dtype=torch.long, device=bbox_class.device),  # 插入预定义类别
                bbox_class[insert_idx:]
            ], dim=0)






        return trajs, bboxes, bbox_class




    def obtain_perception_object_features(self, perception_instance, history_horizon=3, time_horizon=3, num_objects=10, data=None, allow_augmentation=True):
        # 将感知结果进行重新编码，方便模型学习
        if not self.decode_perception_query:
            bs = perception_instance["instance_feature"].shape[0]
            device = perception_instance["instance_feature"].device
            data['augment_bbox'] = torch.zeros(size=(bs,), dtype=torch.bool, device=device)
            return perception_instance, None, None


        # 解码并计算未来Objects的角点
        # 使用Topk  选择
        perception_instance_feature = perception_instance["instance_feature"]
        perception_anchor_embed = perception_instance["anchor_embed"]
        perception_classification = perception_instance["classification"][-1].sigmoid()
        perception_anchors = perception_instance["prediction"][-1]
        perception_confidence = perception_classification.max(dim=-1).values
        perception_trajs = perception_instance['traj'][-1]
        percetion_selected_confidence, (percetion_instance_feature_selected, perception_trajs_selected, perception_anchors_selected, perception_classification_selected) = topk(
            perception_confidence, num_objects, perception_instance_feature, perception_trajs, perception_anchors, perception_classification
        )
        perception_anchors_selected = self.decode_box(perception_anchors_selected)[:,:,:-1] # 最后一个是Vz, 不要了
        bs = perception_instance_feature.shape[0]

        perception_classification_selected = torch.argmax(perception_classification_selected, dim=-1)


        # 将confidence低于0.3的物体全部置为0
        confidence_thrhold = self.longitudinal_train_use_grand.get('confidence_thrhold', 0.3)
        mask = percetion_selected_confidence < confidence_thrhold
        perception_trajs_selected[mask] = 0.0
        perception_anchors_selected[mask] = 0.0
    

        global_idx, light_state, light_position, light_state_embedding = self.find_traffic_lights(perception_anchors_selected, perception_classification_selected)






        # 判断是否做数据增强
        augment_start_count = self.longitudinal_train_use_grand.get('start_iter', 9999999)
        if self.iter_count < augment_start_count:
            allow_augmentation = False
        elif self.iter_count == augment_start_count:
            print(f"Start to allow augmentation at iter {self.iter_count}")
            allow_augmentation = True
        else:
            allow_augmentation = True
        augment_bbox = random.random() < self.augment_bbox_pro
        if augment_bbox and self.training and allow_augmentation:
        # if augment_bbox:    # 仅在测试的时候用于可视化
        # if True:    # 仅在测试的时候用于可视化
            augment_bbox_flag = torch.zeros(bs).to(perception_instance_feature.device).bool()
            valid_flag = data['valid_flag'] # B 
            data['augment_bbox'] = valid_flag   
            augmented_trajs = []
            augmented_bboxes = []
            augmented_bboxes_class = []
            fake_bbox = data[self.longitudinal_train_use_grand['augmented_bbox']]
            fake_trajs = data[self.longitudinal_train_use_grand['augmented_bbox_trajs']]
            fake_trajs = rearrange(fake_trajs, 'b n d -> b (n d)')
            # 2025.6.16, 不考虑随机插入的情况下，这里应该可以做到batch并行
            for i in range(bs):
                trajs, bboxes, bboxes_class = self.augment_bbox(perception_trajs_selected[i], perception_anchors_selected[i], fake_trajs[i], fake_bbox[i],  perception_classification_selected[i], valid_flag=valid_flag[i], given_index=-1)
                augmented_trajs.append(trajs)
                augmented_bboxes.append(bboxes)
                augmented_bboxes_class.append(bboxes_class)

            perception_trajs_selected = torch.stack(augmented_trajs, dim=0)
            perception_anchors_selected = torch.stack(augmented_bboxes, dim=0)
            perception_classification_selected = torch.stack(augmented_bboxes_class, dim=0)

        else:
            # augment_bbox中所有flag都为False
            data['augment_bbox'] = torch.zeros(size=(bs,), dtype=torch.bool, device=perception_trajs_selected.device)



        # 获取物体的预测位置
        fut_bbox_batch = rearrange(perception_trajs_selected, 'b n (t c) -> b n t c', t=6, c=2) # 
        # 添加一个 (0,0) 作为初始位置
        fut_bbox_batch = torch.cat((torch.zeros(bs, num_objects, 1, 2).to(perception_trajs_selected.device), fut_bbox_batch), dim=2)

        # 创建Agents的初始状态
        half_fut_bbox_length = perception_anchors_selected[:, :, 3] / 2
        half_fut_bbox_width = perception_anchors_selected[:, :, 4] / 2
        bbox_init_center = perception_anchors_selected[:, :, :2]
        bbox_init_yaw = perception_anchors_selected[:, :, 6:7]
        bbox_init_state = torch.cat((bbox_init_center, bbox_init_yaw), dim=-1)

        # 去除batch维度
        fut_bbox_batch = rearrange(fut_bbox_batch, 'b n t c -> (b n) t c').detach()         # 差分形式，没有加上初始位置
        half_fut_bbox_length = rearrange(half_fut_bbox_length, 'b n -> (b n)').detach()
        half_fut_bbox_width = rearrange(half_fut_bbox_width, 'b n -> (b n)').detach()
        bbox_init_state = rearrange(bbox_init_state, 'b n c -> (b n) c').detach()
        

        if self.mutil_step_query > 1:       # 分步step query的结果
            agent_horizon = self.mutil_step_query           
            temporal_grand = self.longitudinal_train_use_grand['temporal']
            if temporal_grand == 'temporal_interval_2_traj':  # Agent的轨迹本身是在2Hz预测的，如果要做5Hz的预测，就需要做对齐
                fut_bbox_batch_sampled = self.interpolate_traj(fut_bbox_batch, input_frency=2, output_frequency=5)  # B, N, T, 2
                fut_bbox_batch = fut_bbox_batch_sampled[:, :agent_horizon]  # 只保留前agent_horizon个时间步的轨迹，更新为5hz的轨迹



        agent_corner, agent_traj = compute_corners_torch(fut_bbox_batch.clone(), half_fut_bbox_length.clone(), half_fut_bbox_width.clone(), bbox_init_state.clone(), return_trajs=True) # 并行维度0

        # 转换回原本的维度
        agent_corner = rearrange(agent_corner, '(b n) t m c -> b n t m c', b=bs, n=num_objects) # (B N) time, 4, 2 # 4是顶点数量
        agent_traj = rearrange(agent_traj, '(b n) t d -> b n t d', b=bs, n=num_objects)



        # 同时准备保存
        if 'longitudinal_agent_corner' not in perception_instance:
            perception_instance['longitudinal_agent_corner'] = []
            perception_instance['longitudinal_traj_embed'] = []
            perception_instance['longitudinal_distance_positions'] = []
            perception_instance['longitudinal_agent_traj'] = []
            perception_instance['longitudinal_agent_confidence']=[]  # B, N, T, 4, 2: Batch size, num_objects, time, num_coner, xy
            perception_instance['longitudinal_agent_cls']=[]
            perception_instance['longitudinal_light_state'] = []
            perception_instance['longitudinal_light_state_embedding'] = []  # B, 1
            

        perception_instance['longitudinal_agent_confidence'].append(percetion_selected_confidence)  # B, N, T, 4, 2: Batch size, num_objects, time, num_coner, xy
        perception_instance['longitudinal_agent_traj'].append(agent_traj)   # 
        perception_instance['longitudinal_agent_corner'].append(agent_corner)  # B, 30, 7, 4, 2: Batch size, num_objects, time, num_coner, xy
        perception_instance['longitudinal_agent_cls'].append(perception_classification_selected)
        perception_instance['longitudinal_light_state'].append(light_state.detach())  # B, 1
        if light_state_embedding is not None:
            perception_instance['longitudinal_light_state_embedding'] = light_state_embedding

        fut_t = agent_corner.shape[2]
        num_corner = agent_corner.shape[3]

        # 将角点重新编码
        agent_corner = rearrange(agent_corner, 'b n t m c -> b (n t m) c', b=bs, n=num_objects)
        # 进行傅立叶编码
        agent_corner_embed = gen_sineembed_for_position(agent_corner, hidden_dim=16)   # 每个点用16维表示
       
        # 将轨迹编码
        if self.mutil_step_query ==1:
            agent_corner_embed = rearrange(agent_corner_embed, 'b (n t m) c -> b n (t m c)', b=bs, n=num_objects, t=fut_t, m=num_corner)
            traj_embed = self.traj_encoder(agent_corner_embed)
            traj_embed = traj_embed.unsqueeze(2).expand(-1, -1, self.mutil_step_query, -1)  # 扩展到 (B, N, T, D)
        else:
            agent_corner_embed = rearrange(agent_corner_embed, 'b (n t m) c -> b n t (m c)', b=bs, n=num_objects, t=fut_t, m=num_corner)
            traj_embed = self.traj_encoder(agent_corner_embed)

        agents_class_embedding = self.agents_class_embedding[perception_classification_selected]  # B, N, D
        agents_class_embedding = agents_class_embedding.unsqueeze(2)
        traj_embed = traj_embed + agents_class_embedding  # B, N, T, D

        perception_instance['longitudinal_traj_embed'].append(traj_embed)


        distance_positions = perception_anchors_selected[:, :, :2].detach()  # det_anchors 的前两列是 x, y 坐标
        distance_positions = torch.norm(distance_positions, dim=-1)  # 计算每个 det_anchors 到原点的距离
        distance_positions = distance_positions.unsqueeze(-1)  # 添加一个维度，使其形状为 (B, num_objects, 1)
        distance_positions = distance_positions.expand(-1, -1, self.mutil_step_query)  # 扩展到 (B, num_objects, time)
 
        perception_instance['longitudinal_distance_positions'].append(distance_positions)

        return perception_instance, perception_trajs_selected, perception_anchors_selected






    def find_traffic_lights(self, perception_anchors_selected, perception_classification_selected):
        # perception_anchors_selected shape: [B, N, 3]
        # perception_classification_selected shape: [B, N]
        
        # 创建交通灯掩码 [B, N]
        traffic_light_mask = torch.zeros_like(perception_classification_selected, dtype=torch.bool)
        for i in range(len(self.traffic_light_idx)):
            traffic_light_mask |= (perception_classification_selected == self.traffic_light_idx[i])
        perception_anchors_selected_position_y = perception_anchors_selected[:, :, 1]  # [B, N]
        traffic_light_mask = traffic_light_mask & (perception_anchors_selected_position_y > 0)  # 只保留y大于0的交通灯



        # 初始化返回结果
        batch_size = perception_anchors_selected.size(0)
        global_idx = torch.full((batch_size,), 0, dtype=torch.long)
        light_state = torch.full((batch_size,), 0, dtype=torch.long)
        light_position = torch.zeros((batch_size, 3), dtype=perception_anchors_selected.dtype)
        
        # 对每个batch进行处理
        for b in range(batch_size):
            if torch.sum(traffic_light_mask[b]) > 0:
                # 获取当前batch的交通灯状态和位置
                traffic_lights_states = perception_classification_selected[b][traffic_light_mask[b]]  # [M]
                traffic_lights = perception_anchors_selected[b][traffic_light_mask[b]][:, :3]  # [M, 3]
                # 计算角度和距离
                x = traffic_lights[:, 0]
                y = traffic_lights[:, 1]

                
                angles = torch.rad2deg(torch.atan2(x, y+0.0001))  # [M]
                dists = torch.sqrt(x ** 2 + y ** 2)  # [M]
                
                # 找到角度最小的交通灯索引
                idx_min_angle = torch.argmin(torch.abs(angles))  # int
                
                # 检查是否满足条件
                angle_thresh = 15.0
                dist_thresh_max = 40.0
                dist_thresh_max_min = 1
                chosen_angle = angles[idx_min_angle].item()
                chosen_dist = dists[idx_min_angle].item()
                
                if abs(chosen_angle) <= angle_thresh and chosen_dist <= dist_thresh_max and chosen_dist >= dist_thresh_max_min:
                    # 保存结果
                    light_state[b] = traffic_lights_states[idx_min_angle].item() - self.traffic_light_idx[0] + 1    # 归一化到从1开始
                    light_position[b] = traffic_lights[idx_min_angle]
                    
                    # 获取全局索引
                    valid_indices = torch.nonzero(traffic_light_mask[b], as_tuple=False).squeeze(1)
                    global_idx[b] = valid_indices[idx_min_angle].item()
        

        if self.merage_traffic_light != 'none':
            light_state_embedding = self.traffic_class_embedding[light_state]  # B, D
        else:
            light_state_embedding = None

        return global_idx, light_state, light_position, light_state_embedding








    def interpolate_traj(self, trajs, input_frency=2, output_frequency=5):
        # 首先将位移差转换为绝对坐标
        trajs = torch.cumsum(trajs, dim=1)
        
        # 计算输入和输出的时间点
        input_time = torch.arange(trajs.size(1)) / input_frency
        output_time = torch.arange(trajs.size(1) * output_frequency // input_frency) / output_frequency
        
        interpolated = []
        for batch in trajs:
            # 手动实现线性插值
            x_interp = []
            y_interp = []
            for t in output_time:
                # 找到最近的输入时间点
                idx = torch.searchsorted(input_time, t, right=True)
                if idx == 0:
                    x = batch[0, 0]
                    y = batch[0, 1]
                elif idx == len(input_time):
                    x = batch[-1, 0]
                    y = batch[-1, 1]
                else:
                    # 线性插值
                    t0 = input_time[idx-1]
                    t1 = input_time[idx]
                    alpha = (t - t0) / (t1 - t0)
                    x = batch[idx-1, 0] * (1 - alpha) + batch[idx, 0] * alpha
                    y = batch[idx-1, 1] * (1 - alpha) + batch[idx, 1] * alpha
                x_interp.append(x)
                y_interp.append(y)
            
            interpolated_batch = torch.stack([
                torch.stack(x_interp),
                torch.stack(y_interp)
            ], dim=1)
            interpolated.append(interpolated_batch)
        
        interpolated = torch.stack(interpolated)
        
        # 将绝对坐标转换回位移差
        interpolated_diff = torch.diff(interpolated, dim=1)
        interpolated_diff = torch.cat([
            torch.zeros_like(interpolated_diff[:, :1]), 
            interpolated_diff
        ], dim=1)
        
        return interpolated_diff



    def cross_gnn(self, index, det_instance, map_instance, plan_instance, ignore_query=False):
        # 使用Plan query与Det与Map query进行cross Attention
       

        plan_instance_feature = plan_instance['instance_feature']
        plan_instance_position = plan_instance['anchor_embed']

        agent_instance_feature = det_instance["instance_feature"]
        agent_anchor_embed = det_instance["anchor_embed"]
        det_classification = det_instance["classification"][-1].sigmoid()
        det_anchors = det_instance["prediction"][-1]
        
        det_confidence = det_classification.max(dim=-1).values
        _, (agent_instance_feature_selected, agent_anchor_embed_selected, det_anchors_selected) = topk(
            det_confidence, self.plan_num_det, agent_instance_feature, agent_anchor_embed, det_anchors
        )

        # 不要地图信息, 只要动态物体
        if self.plan_num_map > 0:
            map_instance_feature = map_instance["instance_feature"]
            map_anchor_embed = map_instance["anchor_embed"]
            map_classification = map_instance["classification"][-1].sigmoid()
            map_anchors = map_instance["prediction"][-1]
            map_confidence = map_classification.max(dim=-1).values
            _, (map_instance_feature_selected, map_anchor_embed_selected) = topk(
                map_confidence, self.plan_num_map, map_instance_feature, map_anchor_embed
            )

            perception_instance_feature = torch.cat([agent_instance_feature_selected, map_instance_feature_selected], dim=1)
            perception_instance_position = torch.cat([agent_anchor_embed_selected, map_anchor_embed_selected], dim=1)
        
        else:

            perception_instance_feature = agent_instance_feature_selected
            perception_instance_position = agent_anchor_embed_selected

            

        plan_instance_feature = self.layers[index](
            plan_instance_feature,
            key=perception_instance_feature,
            value=perception_instance_position,
            query_pos=plan_instance_position,
            key_pos=perception_instance_position,
        )

        return plan_instance_feature


    def cross_gnn_with_image(self, index, plan_instance):
        # instance feature 与 图像特征进行cross Attention
        num_t = plan_instance['instance_feature'].shape[2]  # 获取时间步数
        num_n = plan_instance['instance_feature'].shape[1]  # 获取模态数

        img_tokens = plan_instance['img_tokens']  # B, N, D
        instance_feature = plan_instance['instance_feature']  # B, N, T, D
        anchor_embed = plan_instance['anchor_embed']  # B, N, D

        instance_feature = rearrange(instance_feature, 'b n t d -> b (t n) d')  # 不同时刻轨迹自行与环境交互
        

        instance_feature = self.layers[index](
            instance_feature,
            key=img_tokens,
            value=img_tokens,
        )
        instance_feature = rearrange(instance_feature, 'b (t n) d -> b n t d', t=num_t, n=num_n)

        return instance_feature





    def cross_gnn_decode_query(self, index, det_instance, map_instance, plan_instance, ignore_query=False):
        # 使用Plan query与Det与Map query进行cross Attention
       
        plan_instance_feature = plan_instance['instance_feature']   # B, N, T, D 
        num_t = plan_instance_feature.shape[2]  # 获取时间步数
        bs = plan_instance_feature.shape[0]  # 获取batch size
        plan_instance_position = plan_instance['anchor_embed']  # 很重要，anchor作为未来的Plan基准
        plan_instance_position = plan_instance_position.unsqueeze(2).expand(-1, -1, num_t, -1)  # 扩展到 (B, N, T, D) 

        det_traj_embed = det_instance['longitudinal_traj_embed'][-1]             # B, N, T, D
        det_traj_distance = det_instance['longitudinal_distance_positions'][-1]  # B, N, T  

        # 重塑plan query的维度
        plan_instance_feature = rearrange(plan_instance_feature, 'b n t d -> (b t) n d')  # 避免不同个时刻交互
        plan_instance_position = rearrange(plan_instance_position, 'b n t d -> (b t) n d')  # 避免不同个时刻交互

        # 重塑det query的维度
        det_traj_embed = rearrange(det_traj_embed, 'b n t d -> (b t) n d')
        det_traj_distance = rearrange(det_traj_distance, 'b n t -> (b t) n')  # (B T) N

        # 为agents加入序列编码，以区分不同的objects
        agent_sequential_embed = self.agent_position_embedding.unsqueeze(0) # B, N, D
        det_traj_embed = det_traj_embed + agent_sequential_embed



        # 不再做位置编码，因为感知的本身就已经是位置了
        plan_instance_feature = self.layers[index](
            plan_instance_feature+plan_instance_position,
            key=det_traj_embed,
            value=det_traj_embed,
            distances=det_traj_distance,
        )
        plan_instance_feature = rearrange(plan_instance_feature, '(b t) n d -> b n t d', b=bs, t=num_t)
        plan_instance_position = rearrange(plan_instance_position, '(b t) n d -> b n t d', b=bs, t=num_t)

        # temporal_position_embed = get_1d_sincos_pos_embed_from_grid(
        #     self.embed_dims, torch.arange(num_t)
        # )

        # temporal_position_embed = temporal_position_embed.unsqueeze(0).unsqueeze(1).expand(bs, -1, -1, -1)  # (B, 1, T, D)
        # plan_instance_feature = plan_instance_feature + temporal_position_embed


        return plan_instance_feature




    def temporal_fusion(self, index, instance_feature, anchor_embedding):
        # 对不同时间步的Plan query进行融合
        num_t = instance_feature.shape[2]  # 获取时间步数
        num_modal = instance_feature.shape[1]  # 获取模态数
        bs = instance_feature.shape[0]  # 获取batch size
        anchor_embedding = anchor_embedding.unsqueeze(-2)   # B, N ,1, D
        anchor_embedding = anchor_embedding.expand(-1, -1, num_t, -1)  # 扩展到 (B, N, T, D)
        # 只在时序维度进行融合
        temporal_embed = self.temporal_position_embed.unsqueeze(0).unsqueeze(0) # 时间位置编码构建
        instance_feature = instance_feature + temporal_embed
        instance_feature = rearrange(instance_feature, 'b n t d -> (b n) t d')  # 避免不同模态之间交互

        anchor_embedding = rearrange(anchor_embedding, 'b n t d -> (b n) t d')  # 避免不同模态之间交互
        
        
        # 进行casual self-attention
        instance_feature, _ = self.layers[index](instance_feature + anchor_embedding)

        instance_feature = rearrange(instance_feature, '(b n) t d -> b n t d', b=bs, n=num_modal)

        return instance_feature


    def get_feature(self, prediction, reference_feature, metas):

        anchors, instance_feature, reference_feature, drive_path = self.instance_bank.get(
            trajs = prediction,
            reference_feature = reference_feature,
            ego_speed = metas['ego_speed'],    # egospeed是已经加了噪声
        )

        anchor_embed = self.anchor_encoder(anchors)
        batch_size = anchors.shape[0]  # 获取batch size
        output = {}
        output['anchor'] = anchors
        output['anchor_embed'] = anchor_embed
        output['instance_feature'] = instance_feature
        output['drivepath'] = drive_path   # B,N,T,2
        output['reference_feature'] = reference_feature  # B, N, T, D
        # if self.drive_path_encode:
        #     drive_path_embed = gen_sineembed_for_position(drive_path, hidden_dim=16)
        #     drive_path_embed = rearrange(drive_path_embed, 'b n t d -> b n (t d)')  # 这样编码之后模型是否可以意识到方向？
        #     drive_path_embed = self.drivepath_encoder(drive_path_embed) # 
        #     output['drive_path_embed'] = drive_path_embed   # 将DrivePath编码
        # else:
        drive_path = rearrange(drive_path, 'b n t d -> b n (t d)')  # 这样编码之后模型是否可以意识到方向？
        drive_path_embed = self.anchor_encoder(drive_path)
        output['drive_path_embed'] = drive_path_embed


        # 获取时序信息
        if self.HistoryPlanBank is not None:
            # instance_feature放进去主要是为了得到dtype和device
            temporal_instance_feature,  temporal_anchor, _, time_interval = self.HistoryPlanBank.get(batch_size, instance_feature, anchors, metas=metas, dn_metas=None, force_check_timeinterval=True)
            output['temporal_instance_feature'] = temporal_instance_feature
            output['temporal_anchor'] = temporal_anchor
            output['time_interval'] = time_interval

        return output  

        # print("FINISH")


    def perception_cross_gnn(self, index, layer_index, perception_instance, plan_instance, ignore_query=False):
        # 使用Plan query与Map的历史query进行cross Attention

        plan_instance_feature = plan_instance['instance_feature']
        plan_instance_position = plan_instance['anchor_embed']
        num_t = plan_instance_feature.shape[2]  # 使用了几个query来表示一个Planning
        plan_instance_feature = rearrange(plan_instance_feature, 'b n t d -> b (n t) d')    #

        instance_feature = perception_instance["instance_feature"]
        anchor_embed = perception_instance["anchor_embed"]
        classification = perception_instance["classification"][-1].sigmoid()
        anchors = perception_instance["prediction"][-1]
        confidence = classification.max(dim=-1).values


        _, (instance_feature_selected, anchor_embed_selected, anchor_selected) = topk(
            confidence, self.plan_num_map, instance_feature, anchor_embed, anchors
        )


        positions = rearrange(anchor_selected, 'b n (d k) -> b n d k', d=20, k=2)  # map_anchors 的前两列是 x, y 坐标
        positions = positions.mean(dim=-2)  # Map的位置是一个平均值
        distance_map = torch.norm(positions, dim=-1)  # 计算每个 map_anchors 到原点的距离

        # 将instance feature等信息 拓展到与plan instance feature 一样的维度
        plan_instance_position = plan_instance_position.unsqueeze(2).repeat(1, 1, num_t, 1)
        plan_instance_position = rearrange(plan_instance_position, 'b n t d -> b (n t) d')


        plan_instance_feature = self.layers[index](
            plan_instance_feature,
            key=instance_feature_selected.detach(),
            value=instance_feature_selected.detach(),
            query_pos=plan_instance_position,
            key_pos=anchor_embed_selected.detach(),
            distances=distance_map.detach(),
        )
        # 恢复形状
        plan_instance_feature = rearrange(plan_instance_feature, 'b (n t) d -> b n t d', t=num_t)


        return plan_instance_feature







    def percept_traffic_lights(self, plan_instance_feature, plan_instance_position, det_instance, index):
        # 感知交通灯信息
        # 主要是为了与Perception的TrafficLight进行交互




        bs = plan_instance_feature.shape[0]  # 获取batch size
        num_t = plan_instance_feature.shape[2]  # 获取时间步数
        plan_instance_position = plan_instance_position.unsqueeze(2).expand(-1, -1, num_t, -1)  # 扩展到 (B, N, T, D) 

        det_traj_embed = det_instance['longitudinal_traj_embed'][-1]             # B, N, T, D
        det_traj_distance = det_instance['longitudinal_distance_positions'][-1]  # B, N, T  
        det_traj_class = det_instance['longitudinal_agent_cls'][-1]  # B, N, T, C

        # 重塑plan query的维度
        plan_instance_feature = rearrange(plan_instance_feature, 'b n t d -> (b t) n d')  # 避免不同个时刻交互
        plan_instance_position = rearrange(plan_instance_position, 'b n t d -> (b t) n d')  # 避免不同个时刻交互

        # 重塑det query的维度

        traffic_light_idx = self.traffic_light_idx
        traffic_light_mask = torch.zeros_like(det_traj_class, dtype=torch.bool)
        for idx in traffic_light_idx:
            traffic_light_mask = traffic_light_mask | (det_traj_class == idx)



        # 收集红绿灯相关instance features
        traffic_light_list = []
        max_num_tl = 0
        for i in range(det_traj_embed.shape[0]):
            tl_feat_i = det_traj_embed[i][traffic_light_mask[i]]  # [M_i, D]
            traffic_light_list.append(tl_feat_i)
            max_num_tl = max(max_num_tl, tl_feat_i.shape[0])

        # Padding → [B*T, M_max, D]
        traffic_light_padded = torch.zeros((det_traj_embed.shape[0], max_num_tl, det_traj_embed.shape[-1]),
                                        device=det_traj_embed.device)
        traffic_light_mask_pad = torch.zeros((det_traj_embed.shape[0], max_num_tl), dtype=torch.bool,
                                            device=det_traj_embed.device)

        for i, tl_feat_i in enumerate(traffic_light_list):
            traffic_light_padded[i, :tl_feat_i.shape[0]] = tl_feat_i
            traffic_light_mask_pad[i, :tl_feat_i.shape[0]] = True  # 有效红绿灯标为 





        det_traj_embed = det_traj_embed[traffic_light_mask]  # 只保留交通灯信息
        det_traj_distance = det_traj_distance[traffic_light_mask]  # 只保留交通灯信息


        det_traj_embed = rearrange(det_traj_embed, 'b n t d -> (b t) n d')      # 已经包含类别编码特征
        det_traj_distance = rearrange(det_traj_distance, 'b n t -> (b t) n')  # (B T) N

        # # 为agents加入序列编码，以区分不同的objects
        # agent_sequential_embed = self.agent_position_embedding.unsqueeze(0) # B, N, D
        # det_traj_embed = det_traj_embed + agent_sequential_embed



        # 不再做位置编码，因为感知的本身就已经是位置了
        plan_instance_feature = self.layers[index](
            plan_instance_feature+plan_instance_position,
            key=det_traj_embed,
            value=det_traj_embed,
            distances=det_traj_distance,
        )
        plan_instance_feature = rearrange(plan_instance_feature, '(b t) n d -> b n t d', b=bs, t=num_t)
        plan_instance_position = rearrange(plan_instance_position, '(b t) n d -> b n t d', b=bs, t=num_t)


        return plan_instance_feature
