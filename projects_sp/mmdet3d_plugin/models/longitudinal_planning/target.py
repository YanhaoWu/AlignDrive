import torch

from mmdet.core.bbox.builder import BBOX_SAMPLERS
from einops import rearrange
import numpy as np
import pickle
import random
__all__ = ["MotionTarget", "PlanningTarget", "MultiGranularityPlanningTarget"]


def select_best_mode_indices(
    reg_preds, 
    reg_target,
    reg_weight,
):
    bs, num_pred, mode, ts, d = reg_preds.shape
    reg_preds_cum = reg_preds.cumsum(dim=-2)
    reg_target_cum = reg_target.cumsum(dim=-2)
    dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum, dim=-1)
    dist = dist * reg_weight.unsqueeze(2)
    dist = dist.mean(dim=-1)
    mode_idx = torch.argmin(dist, dim=-1)
    return mode_idx

def select_best_mode_predictions(
    reg_preds, 
    reg_target,
    reg_weight,
):
    bs, num_pred, mode, ts, d = reg_preds.shape
    reg_preds_cum = reg_preds.cumsum(dim=-2)
    reg_target_cum = reg_target.cumsum(dim=-2)
    dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum, dim=-1)
    dist = dist * reg_weight.unsqueeze(2)
    dist = dist.mean(dim=-1)
    mode_idx = torch.argmin(dist, dim=-1)
    mode_idx = mode_idx[..., None, None, None].repeat(1, 1, 1, ts, d)
    best_reg = torch.gather(reg_preds, 2, mode_idx).squeeze(2)
    return best_reg


@BBOX_SAMPLERS.register_module()
class MotionTarget():
    def __init__(
        self,
    ):
        super(MotionTarget, self).__init__()

    def sample(
        self,
        reg_pred,
        gt_reg_target,
        gt_reg_mask,
        motion_loss_cache,
    ):
        bs, num_anchor, mode, ts, d = reg_pred.shape
        reg_target = reg_pred.new_zeros((bs, num_anchor, ts, d))
        reg_weight = reg_pred.new_zeros((bs, num_anchor, ts))
        indices = motion_loss_cache['indices']
        num_pos = reg_pred.new_tensor([0])
        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(gt_reg_target[i]) == 0:
                continue
            reg_target[i, pred_idx] = gt_reg_target[i][target_idx]
            reg_weight[i, pred_idx] = gt_reg_mask[i][target_idx]
            num_pos += len(pred_idx)
        
        cls_target = select_best_mode_indices(reg_pred, reg_target, reg_weight)
        cls_weight = reg_weight.any(dim=-1)
        best_reg = select_best_mode_predictions(reg_pred, reg_target, reg_weight)

        return cls_target, cls_weight, best_reg, reg_target, reg_weight, num_pos


@BBOX_SAMPLERS.register_module()
class PlanningTarget():
    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
    ):
        super(PlanningTarget, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

    def sample(
        self,
        cls_pred,
        reg_pred,
        gt_reg_target,
        gt_reg_mask,
        data,
    ):
        gt_reg_target = gt_reg_target.unsqueeze(1)
        gt_reg_mask = gt_reg_mask.unsqueeze(1)

        bs = reg_pred.shape[0]
        bs_indices = torch.arange(bs, device=reg_pred.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)

        cls_pred = cls_pred.reshape(bs, 3, 1, self.ego_fut_mode)
        reg_pred = reg_pred.reshape(bs, 3, 1, self.ego_fut_mode, self.ego_fut_ts, 2)
        cls_pred = cls_pred[bs_indices, cmd]
        reg_pred = reg_pred[bs_indices, cmd]
        cls_target = select_best_mode_indices(reg_pred, gt_reg_target, gt_reg_mask)       # Select the best mode for the ground-truth command.
        cls_weight = gt_reg_mask.any(dim=-1)        # ?
        best_reg = select_best_mode_predictions(reg_pred, gt_reg_target, gt_reg_mask)   # Select the prediction closest to ground truth.

        return cls_pred, cls_target, cls_weight, best_reg, gt_reg_target, gt_reg_mask






@BBOX_SAMPLERS.register_module()
class MultiGranularityPlanningTarget():
    """Assign trajectory modes and targets across spatiotemporal granularities."""

    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
        plan_vocab_path=None,
        keypoint_guide=False,
        used_grand=None,
        SpeedAwareRange=None,
        rand_grand_pro=0,
        num_drivestyle=0,
    ):
        super(MultiGranularityPlanningTarget, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        with open(plan_vocab_path, 'rb') as f:
            plan_vocab = pickle.load(f)
        
        if used_grand == None:
            self.anchor_grand = list(plan_vocab.keys())
        else:
            self.anchor_grand = used_grand
       

        all_anchors = []
        max_len = 0
        for key in used_grand:
            all_anchors.append(plan_vocab[key])     # anchor[key]: N, M, T, 2
            if max_len < plan_vocab[key].shape[2]:
                max_len = plan_vocab[key].shape[2]
            # Pad trajectories to the longest horizon.
        for i in range(len(all_anchors)):
            if all_anchors[i].shape[2] < max_len:
                all_anchors[i] = np.pad(all_anchors[i], ((0, 0), (0, 0), (0, max_len - all_anchors[i].shape[2]), (0, 0)), mode='constant', constant_values=0)


        self.plan_vocab = np.stack(all_anchors, axis=2)    # Six modes with four granularities per mode: (6, N, 4, 6, 2).
        zeros_pad = np.zeros((self.plan_vocab.shape[0], self.plan_vocab.shape[1], self.plan_vocab.shape[2], 1, self.plan_vocab.shape[4]))  # (6,N,1,4,6,2)
        self.plan_vocab = np.concatenate((zeros_pad, self.plan_vocab), axis=3)  # (6,N,5,4,6,2)
        self.plan_vocab = self.plan_vocab[:, :, :, 1:] - self.plan_vocab[:, :, :, :-1]  # (6,N,4,6,2) - (6,N,4,6,2) = (6,N,4,6,2)


        self.plan_vocab = torch.from_numpy(self.plan_vocab).float()  # (6, N, 4, 6, 2)
        self.plan_vocab = rearrange(self.plan_vocab, 'm n g t d -> (m n) g t d')  # (6*N, 4, 6, 2)


        self.num_modal = self.plan_vocab.shape[0]
        self.num_grand = len(all_anchors)
        self.num_drivestyle = num_drivestyle
        # self.grand_name = ['spatial_interval_2_traj', 'spatial_interval_5_traj', 'temporal_interval_2_traj', 'temporal_interval_5_traj']
        self.grand_name = self.anchor_grand

        self.rand_grand_pro = rand_grand_pro
        self.keypoint_guide = keypoint_guide
        self.SpeedAwareRange = SpeedAwareRange



    def find_closest_trajectory_indices(self, gt_traj, pred_traj):
        """Find the prediction closest to ground truth by mean L2 distance.

        Args:
            gt_traj (torch.Tensor): Ground-truth positions, shape [B, T, 2].
            pred_traj (torch.Tensor): Predicted positions, shape [B, N, T, 2].

        Returns:
            torch.Tensor: Closest prediction index per sample, shape [B].
        """
        # Compute the L2 error between each prediction and ground truth.
        dist = torch.norm(pred_traj - gt_traj.unsqueeze(1), dim=-1)  # [B, N, T]

        # Average the error over time for each prediction.
        avg_dist = dist.mean(dim=-1)  # [B, N]

        # Select the prediction with the smallest mean error.
        winner_indices = torch.argmin(avg_dist, dim=1)  # [B]

        min_distance = avg_dist[torch.arange(avg_dist.shape[0]), winner_indices]

        return winner_indices



    def select_speed_granularity(self, data):
        """Select a speed granularity from the current ego speed."""
        ego_status = data['ego_status']  # (B, 10)
        speed = ego_status[:, 6]  # (B,) - speed values

        speed_idx = []
        for s in speed:
            found = False
            for idx, (lower, upper) in enumerate(self.SpeedAwareRange):
                if lower <= s < upper:  # Assuming ranges are [lower, upper)
                    speed_idx.append(idx)
                    found = True
                    break
            if not found:
                # Handle speeds outside all ranges (you might want to append -1 or len(self.SpeedAwareRange))
                speed_idx.append(-1)  # or some other default value     
        return speed_idx

    def select_distance_granularity(self, data, gt_horizon=3, gt_trajs=None):
        """Select a granularity using the net ground-truth displacement.

        Use data['gt_ego_fut_trajs'] when gt_trajs is not provided.
        gt_horizon specifies the trajectory duration in seconds (default: 3).
        """ 
        if gt_trajs is None:
            gt_trajs = data['gt_ego_fut_trajs']                 # B, T, 2
        gt_trajs_distance = torch.cumsum(gt_trajs, dim=1)[:, -1, :] # Net displacement at the final timestep.
        gt_trajs_distance = torch.norm(gt_trajs_distance, dim=-1)

        distance_range = []
        for idx, (lower, upper) in enumerate(self.SpeedAwareRange):
            average_speed = (lower + upper) / 2.0  # Use the midpoint of each speed interval.
            distance_range.append(average_speed * gt_horizon)    # Convert speed to displacement over the requested horizon.
        
        # Select a granularity based on displacement.
        distance_idx = []
        for d in gt_trajs_distance:
            found = False
            for i in range(len(distance_range)):
                distance = distance_range[i]
                if d < distance:  # Assuming ranges are [lower, upper)
                    distance_idx.append(i)
                    found = True
                    break
            if not found:
                # Handle speeds outside all ranges (you might want to append -1 or len(self.SpeedAwareRange))  
                distance_idx.append(
                    -1
                )  # or some other default value
                

        return distance_idx
        



    def sample(
        self,
        data,
        pred_traj,
        decoder_idx,
        use_vocab=True,
        augment_bbox=False,
    ):
        # This sampler is primarily used for the spatial planning stage.

        # Draw a uniform random value in [0, 1).
        rand_num = random.random()
        if rand_num < self.rand_grand_pro:
            # Randomly select a trajectory granularity.
            selected_grand = random.choice(range(self.num_grand))
        else:
            selected_grand = self.grand_name.index('spatial_interval_2_traj')
            # Use the spatial granularity when augmenting trajectories.


        if use_vocab:
            grand_name = self.grand_name[selected_grand]
            gt_traj = data[grand_name]
            gt_traj_mask = data[grand_name.replace('traj', 'mask')]
            anchor_traj = self.plan_vocab.to(gt_traj.device)[:, selected_grand].unsqueeze(0).repeat(gt_traj.shape[0], 1, 1, 1)   # (B, 6, 4, 2) 
            winner_modal = self.find_closest_trajectory_indices(gt_traj, anchor_traj)
            # Collect the ground-truth trajectories and masks.
            traj_list = []
            mask_list = []
            for grand in self.grand_name:
                traj = data[grand]
                mask = data[grand.replace('traj', 'mask')]
                traj_list.append(traj)
                mask_list.append(mask)
            # print("WARNING, THE DRIVING STYLE LOSS IS NOT FINISHED")
            winner_ds = None
        else:
            
            grand_name = self.grand_name[selected_grand]
            gt_traj = data[grand_name]  # B, 6, 2
            pred_traj = rearrange(pred_traj, 'b (m g) (t d) -> b m g t d', m=self.num_modal, g=self.num_grand, t=self.ego_fut_ts)  # (B, 6, 4, T, 2)
            anchor_traj = pred_traj[:, :, selected_grand]   # (B, 6, 4, 2)
            winner_modal = self.find_closest_trajectory_indices(gt_traj, anchor_traj)
            # Collect the ground-truth trajectories and masks.
            traj_list = []
            mask_list = []
            for grand in self.grand_name:
                
                if 'DS' not in grand:
                    traj = data[grand].clone()                          # B,6,2
                    mask = data[grand.replace('traj', 'mask')]
                else:
                    if 'DS_interval_2' in grand:
                        traj = data['temporal_interval_2_traj'].clone()
                        mask = data['temporal_interval_2_mask'].clone()
                    elif 'DS_interval_5' in grand:
                        traj = data['temporal_interval_5_traj'].clone()
                        mask = data['temporal_interval_5_mask'].clone()
                    else:
                        raise ValueError(f"Unknown grand name: {grand}")


                if augment_bbox:    # TODO: Validate the bounding-box augmentation path.
                    if grand == 'temporal_interval_5_traj':
                        # Augmentation may replace the ground-truth trajectory.
                        if 'fake_defensive_ego_fut_trajs' in data:
                            augment_flag = data['augment_bbox'] # B,
                            # Replace trajectories for the augmented samples.
                            defenssive_traj = data['fake_defensive_ego_fut_trajs']
                            aggressive_traj = data['fake_aggressive_ego_fut_trajs']
                            # Use defensive trajectories; random style selection is not implemented.
                            traj[augment_flag] = defenssive_traj[augment_flag].float()  # B,6,2
                    elif grand == 'temporal_interval_2_traj':
                        # Apply the same replacement to 5 Hz trajectories.
                        if 'fake_defensive_ego_fut_trajs_5hz' in data:
                            augment_flag = data['augment_bbox']
                            defenssive_traj = data['fake_defensive_ego_fut_trajs_5hz']
                            aggressive_traj = data['fake_aggressive_ego_fut_trajs_5hz']
                            # Replace ground truth only for valid augmented samples.
                            traj[augment_flag] = defenssive_traj[augment_flag].float()  # B,6,2
                    
                traj_list.append(traj)
                mask_list.append(mask)

            winner_ds = None
            if self.num_drivestyle > 0:

                gt_2hz_traj = data['temporal_interval_5_traj']
                abs_trajs = torch.cumsum(gt_2hz_traj, dim=1)
                average_speed = torch.norm(abs_trajs, dim=-1).mean(dim=-1) / 3  # Shape: [batch_size]

                # Initialize the driving-style index for each sample.
                winner_ds = torch.zeros(average_speed.size(0), dtype=torch.long, device=average_speed.device)

                # Assign samples to the configured speed intervals.
                for i in range(len(self.SpeedAwareRange)):
                    lower, upper = self.SpeedAwareRange[i]
                    # Identify samples within the current speed interval.
                    mask = (lower <= average_speed) & (average_speed < upper)
                    # Assign the interval index to matching samples.
                    winner_ds[mask] = i


        return winner_modal, winner_ds, traj_list, mask_list
