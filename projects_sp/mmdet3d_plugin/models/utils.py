import torch
import numpy as np

from projects_sp.mmdet3d_plugin.core.box3d import X, Y, W, L, H, YAW, SIN_YAW, COS_YAW
from projects_sp.mmdet3d_plugin.datasets.utils import box3d_to_corners


def augment_perception_with_fake_bbox(
    trajs, bboxes, fake_trajs, fake_bboxes, valid_flag=False, given_index=None
):

    """Replace the last perceived agent with a synthetic box and trajectory at a shared index."""
    if valid_flag:

        bboxes = bboxes[:-1]
        trajs = trajs[:-1]

        if given_index is None:
            insert_idx = torch.randint(0, len(bboxes) + 1, (1,)).item()
        else:
            insert_idx = given_index

        bboxes = torch.cat([bboxes[:insert_idx], fake_bboxes, bboxes[insert_idx:]], dim=0)

        trajs = torch.cat([trajs[:insert_idx], fake_trajs.unsqueeze(0), trajs[insert_idx:]], dim=0)

    return trajs, bboxes


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
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
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


def check_collision_with_gt(
    plan_reg,
    agents,
    motion,
    static_dis_thresh=0.5,
    dim_scale=1.1,
    num_motion_mode=1,
    offset=0.5,
    reshape_plan=False,
):
    if reshape_plan:
        plan_reg = plan_reg.reshape(plan_reg.shape[0], plan_reg.shape[1], 6, 2)
        plan_reg = plan_reg.cumsum(-2)

    if isinstance(plan_reg, np.ndarray):
        plan_reg = torch.tensor(plan_reg)
    if isinstance(agents, np.ndarray):
        agents = torch.tensor(agents)
    if isinstance(motion, np.ndarray):
        motion = torch.tensor(motion)

    def cat_with_zero(traj):
        zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
        traj_cat = torch.cat([zeros, traj], dim=-2)
        return traj_cat

    def get_yaw(traj, start_yaw=np.pi / 2):
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

    ego_fut_mode = plan_reg.shape[1]
    ego_fut_ts = plan_reg.shape[2]
    bs = plan_reg.shape[0]
    plan_reg_cat = cat_with_zero(plan_reg)
    ego_box = agents.new_zeros(bs, ego_fut_mode, ego_fut_ts + 1, 7)
    ego_box[..., [X, Y]] = plan_reg_cat
    ego_box[..., [W, L, H]] = ego_box.new_tensor([4.08, 1.73, 1.56]) * dim_scale
    ego_box[..., [YAW]] = get_yaw(plan_reg_cat)

    motion_reg = motion[..., :ego_fut_ts, :].cumsum(-2)
    motion_reg = cat_with_zero(motion_reg) + agents[:, :, None, :2]

    motion_box = motion_reg.new_zeros(motion_reg.shape[:-1] + (7,))
    motion_box[..., [X, Y]] = motion_reg
    motion_box[..., [W, L, H]] = agents[..., None, [W, L, H]]
    box_yaw = torch.atan2(
        agents[..., SIN_YAW],
        agents[..., COS_YAW],
    )
    motion_box[..., [YAW]] = get_yaw(motion_reg, box_yaw)

    ego_box = ego_box[..., 1:, :]
    motion_box = motion_box[..., 1:, :]

    bs, num_ego_mode, ts, _ = ego_box.shape
    bs, num_anchor, ts, _ = motion_box.shape
    ego_box = ego_box[:, None].repeat(1, num_anchor, 1, 1, 1).flatten(0, -2)
    motion_box = motion_box.unsqueeze(2).repeat(1, 1, num_ego_mode, 1, 1).flatten(0, -2)

    ego_box[0] += offset * torch.cos(ego_box[6])
    ego_box[1] += offset * torch.sin(ego_box[6])
    col = check_collision(ego_box, motion_box)
    col = col.reshape(bs, num_anchor, num_ego_mode, ts).permute(0, 2, 1, 3)
    col = col.max(-1)[0]

    return col


def cumulative_distance_from_steps(steps: torch.Tensor) -> torch.Tensor:
    """Convert (B, N, 2) displacement vectors to (B, N + 1) cumulative distances."""
    dists = torch.norm(steps, dim=-1)
    return torch.cat(
        [torch.zeros(dists.shape[0], 1, device=steps.device), torch.cumsum(dists, dim=1)], dim=1
    )


def build_drive_path_absolute(drive_steps: torch.Tensor) -> torch.Tensor:
    """Integrate (B, N, 2) path displacements, prepending the origin."""
    return torch.cat(
        [
            torch.zeros((drive_steps.shape[0], 1, 2), device=drive_steps.device),
            torch.cumsum(drive_steps, dim=1),
        ],
        dim=1,
    )


def batch_linear_interp(query: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Interpolate query (B, Q) on x (B, N) with vector values y (B, N, 2)."""
    B, Nq = query.shape
    _, Nx = x.shape

    inds = torch.searchsorted(x, query, right=True)
    inds = torch.clamp(inds, 1, Nx - 1)

    x0 = x.gather(1, inds - 1)
    x1 = x.gather(1, inds)

    idx0 = (inds - 1).unsqueeze(-1).expand(-1, -1, 2)
    idx1 = inds.unsqueeze(-1).expand(-1, -1, 2)

    y0 = torch.gather(y, 1, idx0)
    y1 = torch.gather(y, 1, idx1)

    t = (query - x0) / (x1 - x0 + 1e-8)
    t = t.unsqueeze(-1)
    return y0 + t * (y1 - y0)


def project_stepwise_waypoints_to_drivepath(
    waypoints: torch.Tensor, drive_path: torch.Tensor
) -> torch.Tensor:
    """Project waypoint step lengths onto a drive path, returning absolute positions including the origin."""

    wp_dists = cumulative_distance_from_steps(waypoints)
    dp_path = build_drive_path_absolute(drive_path)
    dp_dists = cumulative_distance_from_steps(drive_path)

    projected = batch_linear_interp(wp_dists, dp_dists, dp_path)
    return projected


def project_distance_to_drivepath(drive_path, distance):
    """Project (B, N) distance increments onto (B, T, 2) path steps and return position increments."""

    segment_lengths = torch.norm(drive_path, dim=2)

    cum_lengths = torch.cumsum(segment_lengths, dim=1)

    distance = torch.cumsum(distance, dim=1)

    cum_lengths = torch.cat([torch.zeros_like(cum_lengths[:, :1]), cum_lengths], dim=1)

    cum_positions = torch.cat(
        [torch.zeros_like(drive_path[:, :1, :]), torch.cumsum(drive_path, dim=1)], dim=1
    )

    total_length = cum_lengths[:, -1:]

    distance = distance.clone()
    distance = torch.where(distance > total_length, total_length, distance)

    ratios = (distance.unsqueeze(2) - cum_lengths[:, :-1].unsqueeze(1)) / (
        segment_lengths.unsqueeze(1) + 1e-6
    )

    ratios = torch.clamp(ratios, 0, 1)

    start_positions = cum_positions[:, :-1].unsqueeze(1)
    end_positions = cum_positions[:, 1:].unsqueeze(1)

    ratios = ratios.unsqueeze(3)

    projected_positions = start_positions + ratios * (end_positions - start_positions)

    mask = (distance.unsqueeze(2) >= cum_lengths[:, :-1].unsqueeze(1)) & (
        distance.unsqueeze(2) <= cum_lengths[:, 1:].unsqueeze(1)
    )

    result = torch.sum(projected_positions * mask.unsqueeze(3), dim=2)

    result = torch.cat([torch.zeros_like(result[:, :1, :]), result], dim=1)
    result = result[:, 1:, :] - result[:, :-1, :]

    return result
