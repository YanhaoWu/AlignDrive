import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmdet.models.builder import LOSSES
from mmdet.models.losses import l1_loss, smooth_l1_loss
from projects_sp.mmdet3d_plugin.models.longitudinal_planning.score_module.compute_b2d_score_wyh import compute_corners_torch, generate_corners_torch, batch_signed_distance_rect, batch_min_vertex_distance
from einops.einops import rearrange

@LOSSES.register_module()
class PlanningRegLoss(nn.Module):
    def __init__(self, base_loss, loss_weight=1.0, num_horzion=6, temporal_weight=None, balance_setting=None):
        super().__init__()
        self.total_weighted = loss_weight
        self.num_horzion = num_horzion
        self.temporal_weight = [1.0 for _ in range(num_horzion)] if temporal_weight is None else temporal_weight
        self.balance_setting = balance_setting
        self.base_loss = build_from_cfg(base_loss, LOSSES)
        try:
            self.register_buffer('command_weights', torch.tensor([balance_setting['command'][str(i)] for i in range(len(balance_setting['command']))], dtype=torch.float32), persistent=False)
        except:
            self.command_weights = None 
            print("Command weights not found in balance_setting, using default weights.")

        if balance_setting is not None:
            self.balance_start_iter = balance_setting.get('command_scale_iter_map', 0)
        else:
            self.balance_start_iter = 0


        # Convert temporal weights to a tensor for subsequent calculations.
        # self.register_buffer('temporal_weight_tensor', 
        #                    torch.tensor(self.temporal_weight, dtype=torch.float32))


    def normalized_weights(self, weights):
        """Normalize a weight list to have a mean of one."""
        mean = sum(weights) / len(weights)
        return [w / mean for w in weights]


    def forward(self, pred, target, weight, name=None, command=None, iter_count=-1,
                sample_weight=None, longitudinal_weight=None):
        """
        Args:
            pred (Tensor): Prediction tensor of shape (B, 6, 2)
            target (Tensor): Target tensor of shape (B, 6, 2)
            weight (Tensor): Weight/Mask tensor of shape (B, 6, 1)
        Returns:
            Tensor: Weighted loss value
        """
        # Compute the base L1 loss.
        if pred.shape[1] != target.shape[1]:
            num_t = pred.shape[1]
            target_num_t = target.shape[1]
            bs = pred.shape[0]
            # Pad the target to the prediction horizon.
            target = torch.cat([target, torch.zeros(size=(bs, num_t - target_num_t, 2),device=pred.device, dtype=pred.dtype)], dim=1)
            weight = torch.cat([weight, torch.zeros(size=(bs, num_t - target_num_t, 1), device=weight.device, dtype=weight.dtype)], dim=1)
            assert 'interval_5' in name
        base_loss = torch.abs(pred - target)  # shape (B, 6, 2)
        
        # Broadcast spatial weights from (B, 6, 1) to (B, 6, 2).
        weighted_loss = base_loss * weight.expand_as(base_loss)
        if sample_weight is not None:
            weighted_loss = weighted_loss * sample_weight.view(-1, 1, 1)
        if longitudinal_weight is not None:
            weighted_loss[..., 0] = weighted_loss[..., 0] * longitudinal_weight.view(-1, 1)
        

        temporal_weighted = self.temporal_weight
        total_weighted = self.total_weighted
        if name is not None and self.balance_setting is not None:
            if name in self.balance_setting:
                # Apply the optional loss-balancing configuration.
                temporal_weighted = self.balance_setting[name].get('temporal_weighted', self.temporal_weight)
                
                total_weighted = self.balance_setting[name]['total_weighted']



        # Broadcast temporal weights from (6,) to (1, 6, 1).
        # Randomly select a temporal weighting profile.
        num_t = pred.shape[1]
        temporal_weighted = torch.tensor(temporal_weighted, dtype=torch.float32, device=pred.device)
        weighted_loss = weighted_loss * temporal_weighted.view(1, num_t, 1).expand_as(weighted_loss)
        
 
        
        # Assign per-sample weights based on the navigation command.
        if command is not None:
            command = torch.argmax(command, dim=-1) # B,
            if iter_count == self.balance_start_iter:  # Do not balance the loss before balance_start_iter.
                print("Starting to plan balance the loss at iter", iter_count)
            if iter_count > self.balance_start_iter:          # Enable balancing only after the configured iteration.
                # Look up each command weight in balance_setting.
                if self.command_weights is not None:
                    command_weights = self.command_weights[command]  # (6,)
                    command_weights = command_weights.unsqueeze(-1).unsqueeze(-1)  # Reshape (B,) to (B, 1, 1) for broadcasting over time and coordinates.
                    weighted_loss = weighted_loss * command_weights
                            
        # Average the weighted loss over all dimensions.
        loss = weighted_loss.mean()

        return loss * total_weighted






@LOSSES.register_module()
class PlanningCollisionLoss(nn.Module):
    def __init__(self, dist_thrhold=1.0, weights=1.0, reduction='sample_mean'):
        super().__init__()
        if reduction not in ('sample_mean', 'global_collision_mean'):
            raise ValueError('Unsupported collision reduction: ' + str(reduction))
        self.reduction = reduction

        # print("INIT COLLSION LOSS")
        self.dist_thrhold = dist_thrhold
        self.weights = weights




    # def find_interactive_agents(self, pred, future_agent_traj, current_agent_traj):
    # Identify interacting agents that approach the ego vehicle within a distance threshold.
    #     # current_agent_traj:, B, 2
    #     # future_agent_traj: B, T, 2
    #     # pred: B, T, 2
    #     current_interactivte_agents = 
        



    def forward(self, pred, agent_corner, agent_traj, valid_agent_mask=None,
                interactive_agents=None, name=None, sample_weight=None):
        # agent_corner:, B,T+1, 4, 2
        # agent_traj: B, N, T+1, 2
        # pred: B, T, 2
        # valid_agent_mask: B, N

                
        ego_corners, _ = generate_corners_torch(pred)
        num_t = pred.shape[1]
        num_object = agent_corner.shape[1]
        future_agents = agent_corner[:, :, 1:num_t+1,  :]
        # Expand ego corners to match the future-agent shape.
        ego_corners = ego_corners.unsqueeze(1).expand(-1, num_object, -1, -1, -1)

        # Compute distances between ego corners and future agents.
        batch_size = ego_corners.shape[0]
        ego_corners = rearrange(ego_corners, 'b n t d c -> (b n t) d c')
        neighbor_bbox = rearrange(future_agents, 'b n t d c -> (b n t) d c')

        # Filter out invalid agents.
        if valid_agent_mask is not None:
            valid_agent_mask = valid_agent_mask.unsqueeze(2).expand(-1, -1, num_t)  # B, N, T
            valid_agent_mask = rearrange(valid_agent_mask, 'b n t -> (b n t)')
            neighbor_bbox = neighbor_bbox[valid_agent_mask]  # Keep distances for valid agents only.
            ego_corners = ego_corners[valid_agent_mask]  # Keep ego corners corresponding to valid agents only.



        distances = batch_signed_distance_rect(ego_corners, neighbor_bbox) # (B*N, T, 4)
        






        # rectangle_distance = batch_min_vertex_distance(ego_corners, neighbor_bbox)  # (B*N, T, 4)
        # Positive signed distances indicate no collision; use rectangle distances.
        # Negative signed distances indicate collision; retain the signed distances.

        # distances = torch.where(distances > 0, rectangle_distance, distances)  # (B*N, T, 4)



        clip_distances = torch.maximum(1 - distances / self.dist_thrhold, torch.tensor(0.0, device=distances.device))
                

        active = clip_distances > 1
        if self.reduction == 'global_collision_mean':
            if sample_weight is not None and not torch.all(sample_weight == 1):
                raise ValueError('global_collision_mean does not support conditional sample weights')
            loss_collision = clip_distances[active].sum() / (active.float().sum() + 1e-5)
            return loss_collision * self.weights
        flat_batch = torch.arange(batch_size, device=pred.device).view(batch_size, 1, 1)
        flat_batch = flat_batch.expand(batch_size, num_object, num_t).reshape(-1)
        if valid_agent_mask is not None:
            flat_batch = flat_batch[valid_agent_mask]
        values = clip_distances.reshape(-1)
        sums = torch.zeros(batch_size, device=pred.device, dtype=values.dtype)
        counts = torch.zeros(batch_size, device=pred.device, dtype=values.dtype)
        active_flat = active.reshape(-1)
        sums.scatter_add_(0, flat_batch[active_flat], values[active_flat])
        counts.scatter_add_(0, flat_batch[active_flat], torch.ones_like(values[active_flat]))
        loss_per_sample = sums / (counts + 1e-5)
        if sample_weight is None:
            sample_weight = torch.ones(batch_size, device=pred.device, dtype=loss_per_sample.dtype)
        loss_collision = (loss_per_sample * sample_weight).mean()
        # loss_danger = torch.sum(clip_distances[clip_distances <= 1]) / (
        #     torch.sum((clip_distances[clip_distances <= 1].detach() > 0).float()) + 1e-5)

        # Combine the final loss; smaller values are better.
        # Optional exponential scaling of collision and danger losses.
        # loss = loss_collision.exp()
        loss = loss_collision




        return loss_collision * self.weights
        # return loss * total_weighted
