import torch
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from mmdet.core.bbox.match_costs import build_match_cost


def _smooth_l1_pair_sum(pred, gt, beta):
    """Pairwise smooth-L1 summed over last dim.

    Args:
        pred: [c, D]
        gt: [g, D]
    Returns:
        [c, g] — same as
        smooth_l1_loss(pred[:,None].repeat(1,g,1), gt[None].repeat(c,1,1),
                       reduction='none', beta=beta).sum(-1)
    """
    diff = pred.unsqueeze(1) - gt.unsqueeze(0)  # [c, g, D]
    abs_diff = diff.abs()
    loss = torch.where(
        abs_diff < beta,
        0.5 * diff * diff / beta,
        abs_diff - 0.5 * beta,
    )
    return loss.sum(-1)


@MATCH_COST.register_module()
class LinesL1Cost(object):
    """LinesL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.0, beta=0.0, permute=False):
        self.weight = weight
        self.permute = permute
        self.beta = beta

    def __call__(self, lines_pred, gt_lines, **kwargs):
        """
        Args:
            lines_pred (Tensor): predicted normalized lines:
                [num_query, 2*num_points]
            gt_lines (Tensor): Ground truth lines
                [num_gt, 2*num_points] or [num_gt, num_permute, 2*num_points]
        Returns:
            torch.Tensor: reg_cost value with weight
                shape [num_pred, num_gt]
        """
        if self.permute:
            assert len(gt_lines.shape) == 3
        else:
            assert len(gt_lines.shape) == 2

        num_pred, num_gt = len(lines_pred), len(gt_lines)
        if self.permute:
            # permute-invariant labels
            gt_lines = gt_lines.flatten(0, 1)  # (num_gt*num_permute, 2*num_pts)

        num_pts = lines_pred.shape[-1] // 2
        device = lines_pred.device

        if self.beta > 0:
            # Assign cost is detached before Hungarian — safe to compute without
            # keeping a full [Q, G, D] on GPU (G can be thousands after permute).
            G = gt_lines.shape[0]
            D = lines_pred.shape[-1]
            # Always CPU for smooth-L1 assign: peak is transient but still OOMs
            # L20 (~41GB steady) when map GT is dense after permute.
            use_cpu = True
            pred = lines_pred.detach()
            gt = gt_lines.detach()
            if use_cpu:
                pred = pred.float().cpu()
                gt = gt.float().cpu()

            q_chunk, g_chunk = 32, 128
            rows = []
            for i in range(0, num_pred, q_chunk):
                p = pred[i : i + q_chunk]
                cols = []
                for j in range(0, G, g_chunk):
                    cols.append(_smooth_l1_pair_sum(p, gt[j : j + g_chunk], self.beta))
                rows.append(torch.cat(cols, dim=1))
            dist_mat = torch.cat(rows, dim=0)
            if use_cpu:
                dist_mat = dist_mat.to(device)
        else:
            dist_mat = torch.cdist(lines_pred, gt_lines, p=1)

        dist_mat = dist_mat / num_pts

        if self.permute:
            # dist_mat: (num_pred, num_gt*num_permute)
            dist_mat = dist_mat.view(num_pred, num_gt, -1)  # (num_pred, num_gt, num_permute)
            dist_mat, gt_permute_index = torch.min(dist_mat, 2)
            return dist_mat * self.weight, gt_permute_index

        return dist_mat * self.weight


@MATCH_COST.register_module()
class MapQueriesCost(object):

    def __init__(self, cls_cost, reg_cost, iou_cost=None):

        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)

        self.iou_cost = None
        if iou_cost is not None:
            self.iou_cost = build_match_cost(iou_cost)

    def __call__(self, preds: dict, gts: dict, ignore_cls_cost: bool):

        # classification and bboxcost.
        cls_cost = self.cls_cost(preds['scores'], gts['labels'])

        # regression cost
        regkwargs = {}
        if 'masks' in preds and 'masks' in gts:
            assert isinstance(self.reg_cost, DynamicLinesCost), ' Issues!!'
            regkwargs = {
                'masks_pred': preds['masks'],
                'masks_gt': gts['masks'],
            }

        reg_cost = self.reg_cost(preds['lines'], gts['lines'], **regkwargs)
        if self.reg_cost.permute:
            reg_cost, gt_permute_idx = reg_cost

        # weighted sum of above three costs
        if ignore_cls_cost:
            cost = reg_cost
        else:
            cost = cls_cost + reg_cost

        # Iou
        if self.iou_cost is not None:
            iou_cost = self.iou_cost(preds['lines'],gts['lines'])
            cost += iou_cost

        if self.reg_cost.permute:
            return cost, gt_permute_idx
        return cost
