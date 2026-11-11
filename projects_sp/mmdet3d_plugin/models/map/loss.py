import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmdet.models.builder import LOSSES
from mmdet.models.losses import l1_loss, smooth_l1_loss


@LOSSES.register_module()
class LinesL1Loss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, beta=0.5):
        """
            L1 loss. The same as the smooth L1 loss
            Args:
                reduction (str, optional): The method to reduce the loss.
                    Options are "none", "mean" and "sum".
                loss_weight (float, optional): The weight of loss.
        """

        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.beta = beta

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
                shape: [bs, ...]
            target (torch.Tensor): The learning target of the prediction.
                shape: [bs, ...]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None. 
                it's useful when the predictions are not all valid.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.beta > 0:
            loss = smooth_l1_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor, beta=self.beta)
        
        else:
            loss = l1_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        
        num_points = pred.shape[-1] // 2
        loss = loss / num_points

        return loss*self.loss_weight


@LOSSES.register_module()
class SparseLineLoss(nn.Module):
    def __init__(
        self,
        loss_line,
        num_sample=20,
        roi_size=(30, 60),
        balance_setting=None,
    ):
        super().__init__()

        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.loss_line = build(loss_line, LOSSES)
        self.num_sample = num_sample
        self.roi_size = roi_size

    def forward(
        self,
        line,
        line_target,
        weight=None,
        avg_factor=None,
        prefix="",
        suffix="",
        **kwargs,
    ):

        output = {}
        line = self.normalize_line(line)
        line_target = self.normalize_line(line_target)
        line_loss = self.loss_line(
            line, line_target, weight=weight, avg_factor=avg_factor
        )
        output[f"{prefix}loss_line{suffix}"] = line_loss

        return output

    def normalize_line(self, line):
        if line.shape[0] == 0:
            return line

        line = line.view(line.shape[:-1] + (self.num_sample, -1))
        
        origin = -line.new_tensor([self.roi_size[0]/2, self.roi_size[1]/2])
        line = line - origin

        # transform from range [0, 1] to (0, 1)
        eps = 1e-5
        norm = line.new_tensor([self.roi_size[0], self.roi_size[1]]) + eps
        line = line / norm
        line = line.flatten(-2, -1)

        return line

#  Weighted SparseLine Loss
#  May lead unstable training, so not used now

# import torch
# import torch.nn as nn

# from mmcv.utils import build_from_cfg
# from mmdet.models.builder import LOSSES
# from mmdet.models.losses import l1_loss, smooth_l1_loss
# from einops import rearrange

# @LOSSES.register_module()
# class LinesL1Loss(nn.Module):

#     def __init__(self, reduction='mean', loss_weight=1.0, beta=0.5):
#         """
#             L1 loss. The same as the smooth L1 loss
#             Args:
#                 reduction (str, optional): The method to reduce the loss.
#                     Options are "none", "mean" and "sum".
#                 loss_weight (float, optional): The weight of loss.
#         """

#         super().__init__()
#         self.reduction = reduction
#         self.loss_weight = loss_weight
#         self.beta = beta


#     def forward(self,
#                 pred,
#                 target,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None,
#                 mask=None,):
#         """Forward function.
#         Args:
#             pred (torch.Tensor): The prediction.
#                 shape: [bs, ...]
#             target (torch.Tensor): The learning target of the prediction.
#                 shape: [bs, ...]
#             weight (torch.Tensor, optional): The weight of loss for each
#                 prediction. Defaults to None. 
#                 it's useful when the predictions are not all valid.
#             avg_factor (int, optional): Average factor that is used to average
#                 the loss. Defaults to None.
#             reduction_override (str, optional): The reduction method used to
#                 override the original reduction method of the loss.
#                 Defaults to None.
#         """
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)

#         if self.beta > 0:
#             # loss = smooth_l1_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor, beta=self.beta)

#             loss = smooth_l1_loss(
#                 pred, target, weight, reduction='none', avg_factor=avg_factor, beta=self.beta)
#             if mask is not None:
#                 # 将mask满足的位置的loss保留，其他位置的loss置为0
#                 loss[mask == 0] = 0.0  # B, N, 2
#             loss = loss.sum(dim=(-1, -2))/avg_factor  # average over the last two dimensions
#             # loss = loss.sum()/avg_factor
#         else:
#             # loss = l1_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor)

#             loss = l1_loss(
#                 pred, target, weight, reduction='none', avg_factor=avg_factor)

#             if mask is not None:
#                 loss[mask == 0] = 0.0  # B, N, 2

#             loss = loss.sum(dim=(-1, -2))/avg_factor
#         num_points = pred.shape[-1] // 2
#         loss = loss / num_points

#         return loss*self.loss_weight


# @LOSSES.register_module()
# class SparseLineLoss(nn.Module):
#     def __init__(
#         self,
#         loss_line,
#         num_sample=20,
#         roi_size=(30, 60),
#         balance_setting = None
#     ):
#         super().__init__()

#         def build(cfg, registry):
#             if cfg is None:
#                 return None
#             return build_from_cfg(cfg, registry)

#         self.loss_line = build(loss_line, LOSSES)
#         self.num_sample = num_sample
#         self.roi_size = roi_size
#         self.balance_setting = balance_setting
#         try:
#             self.command_weights = torch.tensor([balance_setting['command'][str(i)] for i in range(len(balance_setting['command']))], dtype=torch.float32, device='cuda')
#         except:
#             self.command_weights = None 
#         if balance_setting is not None:
#             self.balance_start_iter = balance_setting.get('command_scale_iter_map', 0)
#         else:
#             self.balance_start_iter = 0
            

        




#     def forward(
#         self,
#         line,
#         line_target,
#         weight=None,
#         avg_factor=None,
#         prefix="",
#         suffix="",
#         **kwargs,
#     ):

#         output = {}
#         line = self.normalize_line(line)
#         line_target = self.normalize_line(line_target)
#         bs = kwargs.get('bs', None)
#         mask = kwargs.get('mask', None) # B, N
#         if bs is not None:  
#             line = rearrange(line, '(b n) d -> b n d', b=bs)
#             line_target = rearrange(line_target, '(b n) d -> b n d', b=bs)
#             weight = rearrange(weight, '(b n) d -> b n d', b=bs) if weight is not None else None
#             mask = rearrange(mask, '(b n) -> b n', b=bs) if mask is not None else None

#         line_loss = self.loss_line(
#             line, line_target, weight=weight, avg_factor=avg_factor, mask=mask,
#         )
#         command = kwargs.get('command', None)
#         if command is not None:
#             current_iter = kwargs.get('iter_count', -1)
#             if current_iter == self.balance_start_iter:  # 在balance_start_iter之前不进行balance
#                 print("Starting to balance map loss the line loss at iter", current_iter)
#             if current_iter > self.balance_start_iter:          # 只有在一定iter之后才开始进行balance
#                 command = torch.argmax(command, dim=-1) # B,
#                 # 在balance_setting中获得command对应的权重
#                 if self.command_weights is not None:
#                     command_weights = self.command_weights[command]  # (6,)
#                     line_loss = line_loss * command_weights  # (B, T, 2) * (B, 1, 1) -> (B, T, 2)
#                     line_loss = line_loss.sum()  # (B, T, 2) -> (B, T)
#                 else:
#                     line_loss = line_loss.sum()
#             else:
#                 line_loss = line_loss.sum()
#         else:
#             line_loss = line_loss.sum()  # 已经除过avg_facor了，只需要求和现在


#         output[f"{prefix}loss_line{suffix}"] = line_loss

#         return output

#     def normalize_line(self, line):
#         if line.shape[0] == 0:
#             return line

#         line = line.view(line.shape[:-1] + (self.num_sample, -1))
        
#         origin = -line.new_tensor([self.roi_size[0]/2, self.roi_size[1]/2])
#         line = line - origin

#         # transform from range [0, 1] to (0, 1)
#         eps = 1e-5
#         norm = line.new_tensor([self.roi_size[0], self.roi_size[1]]) + eps
#         line = line / norm
#         line = line.flatten(-2, -1)

#         return line
