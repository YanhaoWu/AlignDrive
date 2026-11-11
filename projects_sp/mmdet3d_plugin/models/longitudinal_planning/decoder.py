import re
from typing import Optional
from einops.einops import rearrange

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core.bbox.builder import BBOX_CODERS
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from projects_sp.mmdet3d_plugin.core.box3d import *
from projects_sp.mmdet3d_plugin.models.detection3d.decoder import *
from projects_sp.mmdet3d_plugin.models.utils import project_distance_to_drivepath
from projects_sp.mmdet3d_plugin.datasets.utils import box3d_to_corners
from mmcv.runner.base_module import BaseModule
from ..blocks import linear_relu_ln
from mmcv.cnn import Linear, Scale, bias_init_with_prob


@BBOX_CODERS.register_module()
class SparseBox3DMotionDecoder(SparseBox3DDecoder):
    def __init__(self):
        super(SparseBox3DMotionDecoder, self).__init__()

    def decode(
        self,
        cls_scores,
        box_preds,
        instance_id=None,
        quality=None,
        motion_output=None,
        output_idx=-1,
    ):
        squeeze_cls = instance_id is not None

        cls_scores = cls_scores[output_idx].sigmoid()

        if squeeze_cls:
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(dim=-1)

        box_preds = box_preds[output_idx]
        bs, num_pred, num_cls = cls_scores.shape
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1, sorted=self.sorted
        )
        if not squeeze_cls:
            cls_ids = indices % num_cls
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        if quality[output_idx] is None:
            quality = None
        if quality is not None:
            centerness = quality[output_idx][..., CNS]
            centerness = torch.gather(centerness, 1, indices // num_cls)
            cls_scores_origin = cls_scores.clone()
            cls_scores *= centerness.sigmoid()
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            if not squeeze_cls:
                cls_ids = torch.gather(cls_ids, 1, idx)
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, idx)
            indices = torch.gather(indices, 1, idx)

        output = []
        anchor_queue = motion_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = motion_output["period"]

        for i in range(bs):
            category_ids = cls_ids[i]
            if squeeze_cls:
                category_ids = category_ids[indices[i]]
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]
            if quality is not None:
                scores_origin = cls_scores_origin[i]
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]

            box = decode_box(box)
            trajs = motion_output["prediction"][-1]
            traj_cls = motion_output["classification"][-1].sigmoid()
            traj = trajs[i, indices[i] // num_cls]
            traj_cls = traj_cls[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                traj = traj[mask[i]]
                traj_cls = traj_cls[mask[i]]
            traj = traj.cumsum(dim=-2) + box[:, None, None, :2]
            output.append(
                {
                    "trajs_3d": traj.cpu(),
                    "trajs_score": traj_cls.cpu()
                }
            )

            temp_anchor = anchor_queue[i, indices[i] // num_cls]
            temp_period = period[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                temp_anchor = temp_anchor[mask[i]]
                temp_period = temp_period[mask[i]]
            num_pred, queue_len = temp_anchor.shape[:2]
            temp_anchor = temp_anchor.flatten(0, 1)
            temp_anchor = decode_box(temp_anchor)
            temp_anchor = temp_anchor.reshape([num_pred, queue_len, box.shape[-1]])
            output[-1]['anchor_queue'] = temp_anchor.cpu()
            output[-1]['period'] = temp_period.cpu()
        
        return output


@BBOX_CODERS.register_module()
class HierarchicalPlanningDecoder(object):
    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
        use_rescore=False,
    ):
        super(HierarchicalPlanningDecoder, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.use_rescore = use_rescore
    
    def decode(
        self, 
        det_output,
        motion_output,
        planning_output, 
        data,
    ):
        classification = planning_output['classification'][-1]
        prediction = planning_output['prediction'][-1]
        bs = classification.shape[0]
        classification = classification.reshape(bs, 3, self.ego_fut_mode)
        prediction = prediction.reshape(bs, 3, self.ego_fut_mode, self.ego_fut_ts, 2).cumsum(dim=-2)
        classification, final_planning = self.select(det_output, motion_output, classification, prediction, data)
        anchor_queue = planning_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = planning_output["period"]
        output = []
        for i, (cls, pred) in enumerate(zip(classification, prediction)):
            output.append(
                {
                    "planning_score": cls.sigmoid().cpu(),
                    "planning": pred.cpu(),
                    "final_planning": final_planning[i].cpu(),
                    "ego_period": period[i].cpu(),
                    "ego_anchor_queue": decode_box(anchor_queue[i]).cpu(),
                }
            )

        return output

    def select(
        self,
        det_output,
        motion_output,
        plan_cls,
        plan_reg,
        data,
    ):
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        motion_cls = motion_output["classification"][-1].sigmoid()
        motion_reg = motion_output["prediction"][-1]
        
        # cmd select
        bs = motion_cls.shape[0]
        bs_indices = torch.arange(bs, device=motion_cls.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)
        plan_cls_full = plan_cls.detach().clone()
        plan_cls = plan_cls[bs_indices, cmd]
        plan_reg = plan_reg[bs_indices, cmd]

        # rescore
        if self.use_rescore:
            plan_cls = self.rescore(
                plan_cls,
                plan_reg, 
                motion_cls,
                motion_reg, 
                det_anchors,
                det_confidence,
            )
        plan_cls_full[bs_indices, cmd] = plan_cls
        mode_idx = plan_cls.argmax(dim=-1)
        final_planning = plan_reg[bs_indices, mode_idx]
        return plan_cls_full, final_planning

    def rescore(
        self, 
        plan_cls,
        plan_reg, 
        motion_cls,
        motion_reg, 
        det_anchors,
        det_confidence,
        score_thresh=0.5,
        static_dis_thresh=0.5,
        dim_scale=1.1,
        num_motion_mode=1,
        offset=0.5,
    ):
        
        def cat_with_zero(traj):
            zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
            traj_cat = torch.cat([zeros, traj], dim=-2)
            return traj_cat
        
        def get_yaw(traj, start_yaw=np.pi/2):
            yaw = traj.new_zeros(traj.shape[:-1])
            yaw[..., 1:-1] = torch.atan2(
                traj[..., 2:, 1] - traj[..., :-2, 1],
                traj[..., 2:, 0] - traj[..., :-2, 0],
            )
            yaw[..., -1] = torch.atan2(
                traj[..., -1, 1] - traj[..., -2, 1],
                traj[..., -1, 0] - traj[..., -2, 0],
            )
            yaw[..., 0] = start_yaw
            # for static object, estimated future yaw would be unstable
            start = traj[..., 0, :]
            end = traj[..., -1, :]
            dist = torch.linalg.norm(end - start, dim=-1)
            mask = dist < static_dis_thresh
            start_yaw = yaw[..., 0].unsqueeze(-1)
            yaw = torch.where(
                mask.unsqueeze(-1),
                start_yaw,
                yaw,
            )
            return yaw.unsqueeze(-1)
        
        ## ego
        bs = plan_reg.shape[0]
        plan_reg_cat = cat_with_zero(plan_reg)
        ego_box = det_anchors.new_zeros(bs, self.ego_fut_mode, self.ego_fut_ts + 1, 7)
        ego_box[..., [X, Y]] = plan_reg_cat
        ego_box[..., [W, L, H]] = ego_box.new_tensor([4.08, 1.73, 1.56]) * dim_scale
        ego_box[..., [YAW]] = get_yaw(plan_reg_cat)

        ## motion
        motion_reg = motion_reg[..., :self.ego_fut_ts, :].cumsum(-2)
        motion_reg = cat_with_zero(motion_reg) + det_anchors[:, :, None, None, :2]
        _, motion_mode_idx = torch.topk(motion_cls, num_motion_mode, dim=-1)
        motion_mode_idx = motion_mode_idx[..., None, None].repeat(1, 1, 1, self.ego_fut_ts + 1, 2)
        motion_reg = torch.gather(motion_reg, 2, motion_mode_idx)

        motion_box = motion_reg.new_zeros(motion_reg.shape[:-1] + (7,))
        motion_box[..., [X, Y]] = motion_reg
        motion_box[..., [W, L, H]] = det_anchors[..., None, None, [W, L, H]].exp()
        box_yaw = torch.atan2(
            det_anchors[..., SIN_YAW],
            det_anchors[..., COS_YAW],
        )
        motion_box[..., [YAW]] = get_yaw(motion_reg, box_yaw.unsqueeze(-1))

        filter_mask = det_confidence < score_thresh
        motion_box[filter_mask] = 1e6

        ego_box = ego_box[..., 1:, :]
        motion_box = motion_box[..., 1:, :]

        bs, num_ego_mode, ts, _ = ego_box.shape
        bs, num_anchor, num_motion_mode, ts, _ = motion_box.shape
        ego_box = ego_box[:, None, None].repeat(1, num_anchor, num_motion_mode, 1, 1, 1).flatten(0, -2)
        motion_box = motion_box.unsqueeze(3).repeat(1, 1, 1, num_ego_mode, 1, 1).flatten(0, -2)

        ego_box[0] += offset * torch.cos(ego_box[6])
        ego_box[1] += offset * torch.sin(ego_box[6])
        col = check_collision(ego_box, motion_box)
        col = col.reshape(bs, num_anchor, num_motion_mode, num_ego_mode, ts).permute(0, 3, 1, 2, 4)
        col = col.flatten(2, -1).any(dim=-1)
        all_col = col.all(dim=-1)
        col[all_col] = False # for case that all modes collide, no need to rescore
        score_offset = col.float() * -999
        plan_cls = plan_cls + score_offset
        return plan_cls


def check_collision(boxes1, boxes2):
    '''
        A rough check for collision detection: 
            check if any corner point of boxes1 is inside boxes2 and vice versa.
        
        boxes1: tensor with shape [N, 7], [x, y, z, w, l, h, yaw]
        boxes2: tensor with shape [N, 7]
    '''
    col_1 = corners_in_box(boxes1.clone(), boxes2.clone())
    col_2 = corners_in_box(boxes2.clone(), boxes1.clone())
    collision = torch.logical_or(col_1, col_2)

    return collision

def corners_in_box(boxes1, boxes2):
    if  boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return False

    boxes1_yaw = boxes1[:, 6].clone()
    boxes1_loc = boxes1[:, :3].clone()
    cos_yaw = torch.cos(-boxes1_yaw)
    sin_yaw = torch.sin(-boxes1_yaw)
    rot_mat_T = torch.stack(
        [
            torch.stack([cos_yaw, sin_yaw]),
            torch.stack([-sin_yaw, cos_yaw]),
        ]
    )
    # translate and rotate boxes
    boxes1[:, :3] = boxes1[:, :3] - boxes1_loc
    boxes1[:, :2] = torch.einsum('ij,jki->ik', boxes1[:, :2], rot_mat_T)
    boxes1[:, 6] = boxes1[:, 6] - boxes1_yaw

    boxes2[:, :3] = boxes2[:, :3] - boxes1_loc
    boxes2[:, :2] = torch.einsum('ij,jki->ik', boxes2[:, :2], rot_mat_T)
    boxes2[:, 6] = boxes2[:, 6] - boxes1_yaw

    corners_box2 = box3d_to_corners(boxes2)[:, [0, 3, 7, 4], :2]
    corners_box2 = torch.from_numpy(corners_box2).to(boxes2.device)
    H = boxes1[:, [3]]
    W = boxes1[:, [4]]

    collision = torch.logical_and(
        torch.logical_and(corners_box2[..., 0] <= H / 2, corners_box2[..., 0] >= -H / 2),
        torch.logical_and(corners_box2[..., 1] <= W / 2, corners_box2[..., 1] >= -W / 2),
    )
    collision = collision.any(dim=-1)

    return collision



@BBOX_CODERS.register_module()
class MaplikePlanDecoder():
    def __init__(self):
        super(MaplikePlanDecoder, self).__init__()

    def precess_waypoints(self, output):
        # output: B, T, V
        if len(output.shape) == 3:  # The first dimension is the batch dimension.
            output = np.cumsum(output, axis=1)
        else:
            output = np.cumsum(output, axis=0)
        return output

    def decode(
        self,
        cls_scores,
        decoded_waypoints,
        instance_id=None,
        quality=None,
        motion_output=None,
        output_idx=-1,
    ):
        squeeze_cls = instance_id is not None

        cls_scores = cls_scores[output_idx].sigmoid()

        if squeeze_cls:
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(dim=-1)

        decoded_waypoints = decoded_waypoints[output_idx].cpu().numpy()
        bs, num_pred, num_cls = cls_scores.shape
        num_timtestep = int(decoded_waypoints.shape[-1] / 2)
        output = {}
        decoded_waypoints = rearrange(decoded_waypoints, 'b n (t m) -> b n t m', t=num_timtestep, m=2)   # Reshape (B, 30, 12) to (B, 30, 6, 2) coordinate pairs.
        mm_waypoints = []
        for i in range(num_pred):
            processed_waypoints = self.precess_waypoints(decoded_waypoints[:, i])
            mm_waypoints.append(processed_waypoints)

        output['mm_waypoints'] = np.stack(mm_waypoints, axis=1)
        output['mm_score'] = cls_scores.cpu().numpy()[:, :, 0]

        # Select the highest-scoring waypoint trajectory.
        max_socre_indices = output['mm_score'].argmax(axis=-1)
        output['waypoints'] = output['mm_waypoints'][np.arange(bs), max_socre_indices]
        

        # Preserve compatibility with the SparseDrive output format.
        decoded_output_list = []
        for i in range(bs):
            decoded_output_list.append({keys: {key: value[i] for key, value in output[keys].items()} for keys in output.keys()})



        return output


@PLUGIN_LAYERS.register_module()
class ConvParallelMLP(nn.Module):
    def __init__(self, num_mlps=10, input_dim=64, hidden_dim=128, output_dim=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1, groups=1)  # Equivalent to fc1.
        self.conv2 = nn.Conv1d(hidden_dim, output_dim * num_mlps, kernel_size=1)
        self.num_mlps = num_mlps
        self.output_dim = output_dim
        # Optional elementwise output scaling.
    
    def forward(self, x):
        # x shape: [batch_size, input_dim]
        x = x.unsqueeze(-1)  # [batch_size, input_dim, 1]
        x = self.conv2(nn.ReLU()(self.conv1(x)))  # [batch_size, output_dim * num_mlps, 1]
        return x.view(x.size(0), self.num_mlps, self.output_dim)



@PLUGIN_LAYERS.register_module()
class MaplikePlanQueryDecoder(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_sample: int = 20,
        coords_dim: int = 2,
        num_cls: int = 3,
        with_cls_branch: bool = True,
        with_status_branch: bool = True,
        offset_learn: bool = False,
        num_granularity=12,
        num_drivestyle=6,
        num_modal=6,
        mutil_head=False,
    ):
        super(MaplikePlanQueryDecoder, self).__init__()
        self.embed_dims = embed_dims
        self.num_sample = num_sample    # Use the fixed number of navigation commands.
        self.output_dim = self.num_sample * coords_dim
        self.num_cls = num_cls
        self.with_status_branch = with_status_branch
        self.offset_learn = offset_learn
        self.num_granularity = num_granularity
        self.num_modal = num_modal
        self.with_cls_branch = with_cls_branch
        self.mutil_head = mutil_head
        self.num_drivestyle = num_drivestyle
        if self.mutil_head:
            # Create mutil_refine_head parallel refinement heads.
            self.layers = ConvParallelMLP(num_mlps=num_granularity, input_dim=embed_dims, hidden_dim=embed_dims, output_dim=self.output_dim)
            if self.num_drivestyle > 0: # Additional granularities beyond the first four represent driving styles.
                self.syt_cls_layers = ConvParallelMLP(num_mlps=num_modal, input_dim=embed_dims, hidden_dim=embed_dims, output_dim=self.num_drivestyle)
            # if with_cls_branch:
            #     self.cls_layers = ConvParallelMLP(num_mlps=num_modal, input_dim=embed_dims, hidden_dim=embed_dims, output_dim=self.num_cls)
            # if self.with_status_branch:
            #     self.plan_status_branch = ConvParallelMLP(num_mlps=num_granularity, input_dim=embed_dims, hidden_dim=embed_dims, output_dim=10)
        else:
            self.layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 2, 2),
                Linear(self.embed_dims, self.output_dim),
                Scale([1.0] * self.output_dim),
            )
            if self.num_drivestyle > 0: # Additional granularities beyond the first four represent driving styles.
                self.syt_cls_layers = nn.Sequential(
                    *linear_relu_ln(embed_dims, 1, 2),
                    Linear(self.embed_dims, (self.num_drivestyle//2)),    # Select one of three driving styles, shared across sampling frequencies.
                ) 

        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, self.num_cls),
            ) 
        if self.num_drivestyle > 0: # Additional granularities beyond the first four represent driving styles.
            self.syt_cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, (self.num_drivestyle//2)),
            ) 

        if self.with_status_branch:
            self.plan_status_branch = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, 10),
            )


    def init_weight(self):
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor = 1.0,
        return_cls=True,
    ):
        # Predict trajectory offsets.
        output_cls = {}
        if self.mutil_head:
            # Predict trajectory offsets.
            bs = instance_feature.shape[0]
            instance_feature = rearrange(instance_feature, 'b (n m) d -> b n m d', n=self.num_modal, m=self.num_granularity)    # (B, N, M, D)
            anchor_embed = rearrange(anchor_embed, 'b (n m) d -> b n m d', n=self.num_modal, m=self.num_granularity)    # (B, N, M, D)

            # Fuse features across granularities within each mode.
            instance_feature = instance_feature.mean(dim=2)  # Aggregate along the granularity dimension.
            anchor_embed = anchor_embed.mean(dim=2)
            instance_feature = rearrange(instance_feature, 'b n d -> (b n) d')
            anchor_embed = rearrange(anchor_embed, 'b n d -> (b n) d')

            # Use separate MLPs to predict the different trajectories.
            output = self.layers(instance_feature + anchor_embed)  # offset learning
            output = rearrange(output, '(b n) m d -> b (n m) d', b=bs, n=self.num_modal, m=self.num_granularity) # B, 6, 12, 12

            output = output + anchor

            # Predict ego status for each mode.
            if self.with_status_branch:
                plan_ego_status = self.plan_status_branch(instance_feature)
                plan_ego_status = rearrange(plan_ego_status, '(b n) d -> b n d', b=bs, n=self.num_modal) # B, 6, 12, 12
            else:
                plan_ego_status = None
            # Predict a classification score for each mode.
            cls = self.cls_layers(instance_feature)     # anchor embedding?
            cls = rearrange(cls, '(b n) d -> b n d', b=bs, n=self.num_modal) # B, 6, 12, 12

            syt_cls = None
            # Compute driving-style scores.
            if self.num_drivestyle > 0:
                syt_cls = self.syt_cls_layers(instance_feature)
                syt_cls = rearrange(syt_cls, '(b n) d -> b n d', b=bs, n=self.num_modal)
            
            output_cls['cls'] = cls
            output_cls['syt_cls'] = syt_cls

            qt = None

            # 

        else:
            if self.offset_learn:
                output = self.layers(instance_feature + anchor_embed)   # offset learning
                output = output + anchor
            else:
                output = self.layers(instance_feature)  # Predict trajectories directly.

            # Predict ego status.
            if self.with_status_branch:
                plan_ego_status = self.plan_status_branch(instance_feature)
            else:
                plan_ego_status = None
            if return_cls:
                assert self.with_cls_branch, "Without classification layers !!!"
                cls = self.cls_layers(instance_feature)  ## NOTE anchor embed?
            else:
                cls = None
            qt = None
            syt_cls = None
            if self.num_drivestyle > 0:
                syt_cls = self.syt_cls_layers(instance_feature)
        
            output_cls['cls'] = cls
            output_cls['syt_cls'] = syt_cls


        return output, output_cls, qt, plan_ego_status






@PLUGIN_LAYERS.register_module()
class SpeedAwarePlanQueryDecoder(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_sample: int = 20,
        coords_dim: int = 2,
        num_cls: int = 3,
        with_cls_branch: bool = True,
        with_status_branch: bool = True,
        offset_learn: bool = False,
        num_granularity=12,
        num_drivestyle=6,
        num_modal=6,
        mutil_head=False,
        mutil_step_query=1,
        prediction_mode='waypoints'
    ):
        super(SpeedAwarePlanQueryDecoder, self).__init__()
        self.embed_dims = embed_dims
        self.mutil_step_query = mutil_step_query
        self.num_sample = num_sample
        self.prediction_mode = prediction_mode
        if self.prediction_mode == 'distance':
            coords_dim = 1
            # print("Setting the coords_dim to 1 for distance prediction")
        if self.mutil_step_query > 1:
            self.output_dim = num_sample // (self.mutil_step_query-1) * coords_dim
        else:
            self.output_dim = num_sample * coords_dim
        self.num_cls = num_cls
        self.with_status_branch = with_status_branch
        self.offset_learn = offset_learn
        self.num_granularity = num_granularity
        self.num_modal = num_modal
        self.with_cls_branch = with_cls_branch
        self.mutil_head = mutil_head
        self.num_drivestyle = num_drivestyle
        if self.mutil_head:
            # Create mutil_refine_head parallel refinement heads.
            self.regress_layers = ConvParallelMLP(num_mlps=num_granularity, input_dim=embed_dims, hidden_dim=embed_dims, output_dim=self.output_dim)
            self.score_layer = ConvParallelMLP(num_mlps=num_granularity, input_dim=embed_dims, hidden_dim=embed_dims, output_dim=1) # Predict one score per query.
        else:

            self.regress_layers = nn.Sequential(
                        *linear_relu_ln(embed_dims, 2, 2),
                        Linear(self.embed_dims, self.output_dim),
                        Scale([1.0] * self.output_dim),
                    )

            self.score_layer = nn.Sequential(
                        *linear_relu_ln(embed_dims, 1, 2),
                        Linear(self.embed_dims, self.num_cls),
                    )



    def init_weight(self):
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor = 1.0,
        return_cls=True,
        drivepath=None,
    ):
        # Predict trajectory offsets.
        if self.mutil_head:
            # Predict trajectory offsets.
            bs = instance_feature.shape[0]
            instance_feature = rearrange(instance_feature, 'b (n m) d -> b n m d', n=self.num_modal, m=self.num_granularity)    # (B, N, M, D)
            anchor_embed = rearrange(anchor_embed, 'b (n m) d -> b n m d', n=self.num_modal, m=self.num_granularity)    # (B, N, M, D)

            # Fuse features across granularities within each mode.
            instance_feature = instance_feature.mean(dim=2)  # Aggregate along the granularity dimension.
            anchor_embed = anchor_embed.mean(dim=2)
            instance_feature = rearrange(instance_feature, 'b n d -> (b n) d')
            anchor_embed = rearrange(anchor_embed, 'b n d -> (b n) d')

            # Use separate MLPs to predict the different trajectories.
            offset = self.regress_layers(instance_feature + anchor_embed)  # offset learning
            offset = rearrange(offset, '(b n) t d -> b (n t) d', b=bs, n=self.num_modal, t=self.num_granularity) # B, 6, 12, 12
            prediction = offset + anchor

            # Predict a score for each speed granularity.
            score = self.score_layer(instance_feature + anchor_embed)     # anchor embedding?
            score = rearrange(score, '(b m) n d -> b m n d', b=bs, m=self.num_modal, n=self.num_granularity) # B, 6, 4, 1
            
        else:
            bs = instance_feature.shape[0]
            num_t = instance_feature.shape[1]
            anchor_embed = anchor_embed.unsqueeze(-2)   # B, N, t, 1

            prediction = self.regress_layers(instance_feature + anchor_embed)   # offset learning


            score = self.score_layer(instance_feature)
            score = score.mean(dim=-2, keepdim=False)  # Predict one score per query.
            # prediction = rearrange(prediction, 'b (m n) d -> b m n d', b=bs, m=self.num_modal, n=self.num_granularity) # B, 6, 12, 12
            score = rearrange(score, 'b (m n) d -> b m n d', b=bs, m=self.num_modal, n=self.num_granularity)


            num_pt = prediction.shape[2] # Determine the number of predicted timesteps.
            if self.mutil_step_query > 1:
                prediction = prediction[:, :, 1:, :]
            else:
                prediction = prediction.squeeze(-2)  # 
                prediction = prediction.unsqueeze(-1)





        predicted_distance = None
        if self.prediction_mode == 'distance':
            # Compute the driving path.
            num_n = prediction.shape[1] # Determine the number of parallel predictions.
            predicted_distance=prediction
            drivepath = rearrange(drivepath, 'b n t d -> (b n) t d', b=bs)
            if self.offset_learn:
                # Apply the residual to the original prediction.
                anchor = rearrange(anchor, 'b m (t d) -> b m t d', t=self.num_sample, b=bs, m=num_n)
                anchor_distance = torch.norm(anchor, dim=-1, keepdim=True)  # Compute cumulative distances along the anchor.
                prediction = prediction + anchor_distance
                predicted_distance = prediction # Update the predicted longitudinal distances.

                prediction = rearrange(prediction, 'b m t d -> (b m) (t d)')  # B, M, 6, 1
                prediction = project_distance_to_drivepath(drivepath, prediction)   # Project distances onto the driving path and return consecutive-point offsets.


            prediction = rearrange(prediction, '(b m) t d -> b m (t d)', b=bs, m=num_n)






        else:
            prediction = rearrange(prediction, 'b m t d -> b m (t d)', b=bs)
            if self.offset_learn:
                prediction = prediction + anchor

        qt = None

        return prediction, score, qt, predicted_distance
