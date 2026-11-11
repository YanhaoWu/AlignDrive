import copy
from shapely.geometry import Polygon
from shapely.creation import linestrings
from shapely import Point, creation
from shapely.strtree import STRtree
import shapely
from shapely import LineString

import cv2
import numpy as np
import torch

import math
import numpy as np
from pyquaternion import Quaternion
import pyquaternion
from scipy.interpolate import CubicSpline

from projects_sp.mmdet3d_plugin.core.box3d import *
import torch.nn.functional as F
from scipy.spatial.distance import cdist
# from projects_sp.mmdet3d_plugin.models.longitudinal_planning.score_module.compute_b2d_score_wyh import compute_corners_torch, evaluate_coll
from einops.einops import rearrange
from scipy.optimize import minimize



def safe_atan2(y, x):
    # 稳定版的 atan2 计算
    # 位移太小的部分,打上Mask
    magnitude = torch.sqrt(x**2 + y**2)
    mask = magnitude < 1e-1  # 小于1cm的位移

    return torch.atan2(y, x), mask

def compute_corners_torch(proposals, half_lengths=None, half_widths=None, init_state=None):
    """
    计算车辆角点坐标，支持不同 batch 的车辆尺寸（half_lengths 和 half_widths），但时间步长一致。
    
    Args:
        proposals (torch.Tensor): 位移增量 [dx, dy]，shape (B, T, 2)
        half_lengths (torch.Tensor): 每个 batch 的车长一半，shape (B,)
        half_widths (torch.Tensor): 每个 batch 的车宽一半，shape (B,)
        bbox_init_state (torch.Tensor): 初始状态 [x, y, yaw]，shape (B, 3)，可选
    Returns:
        torch.Tensor: 角点坐标，shape (B, T, 4, 2)
    """
    device = proposals.device
    B, T, _ = proposals.shape
    
    # 1. 输入是位移增量 [dx, dy]
    deltas = proposals  # shape: (B, T, 2)

    # 2. 计算每个增量的方向角 yaw
    raw_angles, mask = safe_atan2(deltas[..., 1], deltas[..., 0])  # dy, dx

    # 初始化角度结果张量
    angles = torch.zeros_like(raw_angles)

    if init_state is not None:
        init_yaw = init_state[:, 2]  # shape: (B,)
    else:
        init_yaw = torch.zeros(B, device=device)

    # 3. 用初始角度或上一帧角度替换掉 mask 为 True 的角度
    angles[:, 0] = torch.where(mask[:, 0], init_yaw, raw_angles[:, 0])

    for t in range(1, T):
        angles[:, t] = torch.where(
            mask[:, t],
            angles[:, t - 1],  # 如果这一帧被 mask，就用上一帧的角度
            raw_angles[:, t]
        )

    # 4. 计算角度变化并解缠
    delta_yaws = torch.diff(angles, dim=1)
    delta_yaws = (delta_yaws + np.pi) % (2 * np.pi) - np.pi

    # 6. 累计 yaw 角度
    yaw_angles = torch.cat([
        angles[:, :1],
        angles[:, :1] + delta_yaws.cumsum(dim=1)
    ], dim=1)


    # 7. 累计位置（车体中心）
    positions = torch.cumsum(proposals, dim=1)
    if init_state is not None:
        # 全部位置加上初始中心位置
        init_center = init_state[:, :2]  # shape: (B, 2)
        positions += init_center.unsqueeze(1)  # shape: (B, T, 2)

        # init_yaw = init_state[:, 2]  # shape: (B,)
        # mask


    # 8. 定义车辆尺寸（支持不同 batch 的尺寸）
    if half_lengths is None:
        half_lengths = torch.full((B,), 2.042, device=device)  # 默认车长一半
    if half_widths is None:
        half_widths = torch.full((B,), 0.925, device=device)   # 默认车宽一半

    # 9. 局部坐标系下的角点（x轴是左右，y轴是前后）
    # 每个 batch 的角点不同（因为尺寸不同）
    corners_local_y = torch.stack([
        half_widths, -half_widths, -half_widths, half_widths
    ], dim=-1).unsqueeze(1)  # shape: (B, 1, 4)

    corners_local_x = torch.stack([
        half_lengths, half_lengths, -half_lengths, -half_lengths
    ], dim=-1).unsqueeze(1)  # shape: (B, 1, 4)

    # 10. 旋转角点到全局坐标
    cos_yaw = torch.cos(yaw_angles)[..., None]  # shape: (B, T, 1)
    sin_yaw = torch.sin(yaw_angles)[..., None]

    # ✅ 标准旋转公式
    corners_global_x = cos_yaw * corners_local_x - sin_yaw * corners_local_y + positions[..., 0:1]
    corners_global_y = sin_yaw * corners_local_x + cos_yaw * corners_local_y + positions[..., 1:2]

    corners_global = torch.stack([corners_global_x, corners_global_y], dim=-1)  # shape: (B, T, 4, 2)

    return corners_global
    # FRONT_LEFT = 0
    # REAR_LEFT = 1
    # REAR_RIGHT = 2
    # FRONT_RIGHT = 3

def evaluate_coll(fut_box_corners,_ego_coords):
    # fut_box_corners: [N, 6, 2]
    # _ego_coords: [M, 6, 2]
    num_agents = fut_box_corners.shape[0]
    n_future = _ego_coords.shape[1]
    _num_proposals=_ego_coords.shape[0]


    fut_mask=fut_box_corners.any(-1).any(-1)

    node_capacity=10

    _ego_polygons = shapely.creation.polygons(_ego_coords)
    collision_matrix = np.zeros([_num_proposals, num_agents, n_future], dtype=bool)

    proposal_fault_collided_track_ids = {
        proposal_idx:[]
        for proposal_idx in range(_num_proposals)
    }
    proposal_collided_track_ids = {
        proposal_idx:[]
        for proposal_idx in range(_num_proposals)
    }
    temp_collided_track_ids = {
        proposal_idx:[]
        for proposal_idx in range(_num_proposals)
    }
    ttc_collided_track_ids = {
        proposal_idx:[]
        for proposal_idx in range(_num_proposals)
    }

    key_agent_corners = np.zeros([_num_proposals,6, 4, 2])
    key_agent_labels = np.zeros([_num_proposals, 6],dtype=bool)
    key_agent_ids = []
    collision_all = np.zeros([_num_proposals,n_future],dtype=bool)
    ttc_collision_all = np.zeros([_num_proposals,n_future],dtype=bool)


    for time_idx in range(n_future):
        geometries=fut_box_corners[:,time_idx][fut_mask[:,time_idx]].numpy()
        _geometries = [Polygon(geometry) for geometry in geometries]
        _str_tree = STRtree(_geometries, node_capacity)
        # ego_polygons = _ego_polygons[:, time_idx,0]
        ego_polygons = _ego_polygons[:, time_idx]
        intersecting = _str_tree.query(ego_polygons, predicate="intersects")

        if len(fut_box_corners) == 0 or len(intersecting[0]) == 0:
            continue
        token_list=np.arange(len(fut_box_corners))[fut_mask[:,time_idx].cpu().numpy()]

        for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
            token=token_list[geometry_idx]
            if token in proposal_collided_track_ids[proposal_idx] or len(proposal_fault_collided_track_ids[proposal_idx]):
                continue

            proposal_fault_collided_track_ids[proposal_idx].append(token)
            collision_all[proposal_idx][time_idx]=True
            key_agent_labels[proposal_idx][:time_idx+1]=fut_mask[token][:time_idx+1]
            key_agent_corners[proposal_idx][:time_idx+1]=fut_box_corners[token][:time_idx+1]
            key_agent_ids.append(token)
            collision_matrix[proposal_idx, token, time_idx] = True


    return collision_all,ttc_collision_all,key_agent_corners[:-1],key_agent_labels[:-1], key_agent_ids, collision_matrix

def box3d_to_corners(box3d):
    if isinstance(box3d, torch.Tensor):
        box3d = box3d.detach().cpu().numpy()
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # use relative origin [0.5, 0.5, 0]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box3d[:, None, [W, L, H]] * corners_norm.reshape([1, 8, 3])

    # rotate around z axis
    rot_cos = np.cos(box3d[:, YAW])
    rot_sin = np.sin(box3d[:, YAW])
    rot_mat = np.tile(np.eye(3)[None], (box3d.shape[0], 1, 1))
    rot_mat[:, 0, 0] = rot_cos
    rot_mat[:, 0, 1] = -rot_sin
    rot_mat[:, 1, 0] = rot_sin
    rot_mat[:, 1, 1] = rot_cos
    corners = (rot_mat[:, None] @ corners[..., None]).squeeze(axis=-1)
    corners += box3d[:, None, :3]
    return corners


def plot_rect3d_on_img(
    img, num_rects, rect_corners, color=(0, 255, 0), thickness=1
):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = (
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 7),
        (4, 5),
        (4, 7),
        (2, 6),
        (5, 6),
        (6, 7),
    )
    h, w = img.shape[:2]
    for i in range(num_rects):
        corners = np.clip(rect_corners[i], -1e4, 1e5).astype(np.int32)
        for start, end in line_indices:
            if (
                (corners[start, 1] >= h or corners[start, 1] < 0)
                or (corners[start, 0] >= w or corners[start, 0] < 0)
            ) and (
                (corners[end, 1] >= h or corners[end, 1] < 0)
                or (corners[end, 0] >= w or corners[end, 0] < 0)
            ):
                continue
            if isinstance(color[0], int):
                cv2.line(
                    img,
                    (corners[start, 0], corners[start, 1]),
                    (corners[end, 0], corners[end, 1]),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            else:
                cv2.line(
                    img,
                    (corners[start, 0], corners[start, 1]),
                    (corners[end, 0], corners[end, 1]),
                    color[i],
                    thickness,
                    cv2.LINE_AA,
                )

    return img.astype(np.uint8)


def draw_lidar_bbox3d_on_img(
    bboxes3d, raw_img, lidar2img_rt, img_metas=None, color=(0, 255, 0), thickness=1
):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`LiDARInstance3DBoxes`):
            3d bbox in lidar coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        img_metas (dict): Useless here.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    # corners_3d = bboxes3d.corners
    corners_3d = box3d_to_corners(bboxes3d)
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1
    )
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)


def draw_points_on_img(points, img, lidar2img_rt, color=(0, 255, 0), circle=4):
    img = img.copy()
    N = points.shape[0]
    points = points.cpu().numpy()
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = (
        np.sum(points[:, :, None] * lidar2img_rt[:3, :3], axis=-1)
        + lidar2img_rt[:3, 3]
    )
    pts_2d[..., 2] = np.clip(pts_2d[..., 2], a_min=1e-5, a_max=1e5)
    pts_2d = pts_2d[..., :2] / pts_2d[..., 2:3]
    pts_2d = np.clip(pts_2d, -1e4, 1e4).astype(np.int32)

    for i in range(N):
        for point in pts_2d[i]:
            if isinstance(color[0], int):
                color_tmp = color
            else:
                color_tmp = color[i]
            cv2.circle(img, point.tolist(), circle, color_tmp, thickness=-1)
    return img.astype(np.uint8)


def draw_lidar_bbox3d_on_bev(
    bboxes_3d, bev_size, bev_range=115, color=(255, 0, 0), thickness=3):
    if isinstance(bev_size, (list, tuple)):
        bev_h, bev_w = bev_size
    else:
        bev_h, bev_w = bev_size, bev_size
    bev = np.zeros([bev_h, bev_w, 3])

    marking_color = (127, 127, 127)
    bev_resolution = bev_range / bev_h
    for cir in range(int(bev_range / 2 / 10)):
        cv2.circle(
            bev,
            (int(bev_h / 2), int(bev_w / 2)),
            int((cir + 1) * 10 / bev_resolution),
            marking_color,
            thickness=thickness,
        )
    cv2.line(
        bev,
        (0, int(bev_h / 2)),
        (bev_w, int(bev_h / 2)),
        marking_color,
    )
    cv2.line(
        bev,
        (int(bev_w / 2), 0),
        (int(bev_w / 2), bev_h),
        marking_color,
    )
    if len(bboxes_3d) != 0:
        bev_corners = box3d_to_corners(bboxes_3d)[:, [0, 3, 4, 7]][
            ..., [0, 1]
        ]
        xs = bev_corners[..., 0] / bev_resolution + bev_w / 2
        ys = -bev_corners[..., 1] / bev_resolution + bev_h / 2
        for obj_idx, (x, y) in enumerate(zip(xs, ys)):
            for p1, p2 in ((0, 1), (0, 2), (1, 3), (2, 3)):
                if isinstance(color[0], (list, tuple)):
                    tmp = color[obj_idx]
                else:
                    tmp = color
                cv2.line(
                    bev,
                    (int(x[p1]), int(y[p1])),
                    (int(x[p2]), int(y[p2])),
                    tmp,
                    thickness=thickness,
                )
    return bev.astype(np.uint8)


def draw_lidar_bbox3d(bboxes_3d, imgs, lidar2imgs, color=(255, 0, 0)):
    vis_imgs = []
    for i, (img, lidar2img) in enumerate(zip(imgs, lidar2imgs)):
        vis_imgs.append(
            draw_lidar_bbox3d_on_img(bboxes_3d, img, lidar2img, color=color)
        )

    num_imgs = len(vis_imgs)
    if num_imgs < 4 or num_imgs % 2 != 0:
        vis_imgs = np.concatenate(vis_imgs, axis=1)
    else:
        vis_imgs = np.concatenate([
            np.concatenate(vis_imgs[:num_imgs//2], axis=1),
            np.concatenate(vis_imgs[num_imgs//2:], axis=1)
        ], axis=0)

    bev = draw_lidar_bbox3d_on_bev(bboxes_3d, vis_imgs.shape[0], color=color)
    vis_imgs = np.concatenate([bev, vis_imgs], axis=1)
    return vis_imgs



def output_to_nusc_box(detection, threshold=None):
    from nuscenes.utils.data_classes import Box as NuScenesBox

    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()
    if "instance_ids" in detection:
        ids = detection["instance_ids"]  # .numpy()
    if threshold is not None:
        if "cls_scores" in detection:
            mask = detection["cls_scores"].numpy() >= threshold
        else:
            mask = scores >= threshold
        box3d = box3d[mask]
        scores = scores[mask]
        labels = labels[mask]
        ids = ids[mask]

    if hasattr(box3d, "gravity_center"):
        box_gravity_center = box3d.gravity_center.numpy()
        box_dims = box3d.dims.numpy()
        nus_box_dims = box_dims[:, [1, 0, 2]]
        box_yaw = box3d.yaw.numpy()
    else:
        box3d = box3d.numpy()
        box_gravity_center = box3d[..., :3].copy()
        box_dims = box3d[..., 3:6].copy()
        nus_box_dims = box_dims[..., [1, 0, 2]]
        box_yaw = box3d[..., 6].copy()

    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    # box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if hasattr(box3d, "gravity_center"):
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (*box3d[i, 7:9], 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        if "instance_ids" in detection:
            box.token = ids[i]
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(
    info,
    boxes,
    classes,
    eval_configs,
    eval_version="detection_cvpr_2019",
    filter_with_cls_range=True,
):
    box_list = []
    for i, box in enumerate(boxes):
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))
        # filter det in ego.
        if filter_with_cls_range:
            cls_range_map = eval_configs.class_range
            radius = np.linalg.norm(box.center[:2], 2)
            det_range = cls_range_map[classes[box.label]]
            if radius > det_range:
                continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list

def obtain_map_info(nusc,
                    nusc_maps,
                    sample,
                    patch_size=(102.4, 102.4),
                    canvas_size=(256, 256),
                    layer_names=['lane_divider', 'road_divider'],
                    thickness=10):
    """
    Export 2d annotation from the info file and raw data.
    """
    l2e_r = sample['lidar2ego_rotation']
    l2e_t = sample['lidar2ego_translation']
    e2g_r = sample['ego2global_rotation']
    e2g_t = sample['ego2global_translation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    nusc_map = nusc_maps[log['location']]
    if layer_names is None:
        layer_names = nusc_map.non_geometric_layers

    l2g_r_mat = (l2e_r_mat.T @ e2g_r_mat.T).T
    l2g_t = l2e_t @ e2g_r_mat.T + e2g_t
    patch_box = (l2g_t[0], l2g_t[1], patch_size[0], patch_size[1])
    patch_angle = math.degrees(Quaternion(matrix=l2g_r_mat).yaw_pitch_roll[0])

    map_mask = nusc_map.get_map_mask(
        patch_box, patch_angle, layer_names, canvas_size=canvas_size)
    map_mask = map_mask[-2] | map_mask[-1]
    map_mask = map_mask[np.newaxis, :]
    map_mask = map_mask.transpose((2, 1, 0)).squeeze(2)  # (H, W, C)

    erode = nusc_map.get_map_mask(patch_box, patch_angle, [
                                  'drivable_area'], canvas_size=canvas_size)
    erode = erode.transpose((2, 1, 0)).squeeze(2)

    map_mask = np.concatenate([erode[None], map_mask[None]], axis=0)
    return map_mask



def relative_to_absolute(trajectory: torch.Tensor) -> torch.Tensor:
    """
    将相对坐标（Δx, Δy）转换为绝对轨迹点，起点为 (0,0)
    trajectory: [N, T, 2] 相对位移
    return:     [N, T, 2] 绝对坐标
    """
    return torch.cumsum(trajectory, dim=1)

def compute_frenet_lateral_error_from_relative(drive_path_rel: torch.Tensor, reg_rel: torch.Tensor) -> torch.Tensor:
    """
    drive_path_rel: [N, T, 2] - 参考路径的相对坐标
    reg_rel:        [N, T, 2] - 预测路径的相对坐标
    return:         [N, T]    - 每个时间点的横向偏移量 l
    """
    # 转为绝对坐标，起点为 (0,0)
    drive_path = relative_to_absolute(drive_path_rel)  # [N, T, 2]
    reg = relative_to_absolute(reg_rel)                # [N, T, 2]

    N, T, _ = drive_path.shape
    l_errors = torch.zeros(N, T, device=drive_path.device)

    for t in range(T - 1):
        p0 = drive_path[:, t]      # [N, 2]
        p1 = drive_path[:, t + 1]  # [N, 2]
        q = reg[:, t]              # [N, 2]

        # 路径切线方向
        t_vec = p1 - p0
        t_norm = F.normalize(t_vec, dim=1)

        # 法向方向
        n_vec = torch.stack([-t_norm[:, 1], t_norm[:, 0]], dim=1)

        # 预测点相对路径点偏移
        v = q - p0
        l = (v * n_vec).sum(dim=1)  # 横向分量
        l_errors[:, t] = l

    # 最后一个点没有 p_{t+1} 参考，我们复制 t-1 的偏移
    l_errors[:, -1] = l_errors[:, -2]

    return l_errors




def obtain_traj_given_speed(drive_path, speed, num_horzion=6):
    """
    根据当前速度从DrivePath中插值出对应的轨迹

    参数:
        drive_path: torch.Tensor, 形状为(N, 6, 2)，每一段的相对坐标差（位移向量）
        speed: torch.Tensor, 形状为(N, 6), 表示任意两个坐标点之间距离

    返回:
        torch.Tensor: 插值后的轨迹，形状为(N, 6, 2)，表示每个时间步的目标位置
    """
    N = drive_path.size(0)
    dtype = drive_path.dtype
    # 1. 从差分路径恢复出绝对路径（起点为 0,0）
    abs_pos = torch.cumsum(drive_path, dim=1)  # (N, 6, 2)
    abs_pos = torch.cat([torch.zeros(N, 1, 2, device=drive_path.device), abs_pos], dim=1)  # (N, 7, 2)

    # 2. 计算每段的实际距离
    seg_vecs = abs_pos[:, 1:] - abs_pos[:, :-1]     # (N, 6, 2)
    seg_lens = torch.norm(seg_vecs, dim=2)          # (N, 6)
    cum_dist = torch.cumsum(seg_lens, dim=1)        # (N, 6)
    cum_dist = torch.cat([torch.zeros(N, 1, device=drive_path.device), cum_dist], dim=1)  # (N, 7)

    # 3. 计算目标距离：速度 × 时间（假设每秒插一个点，共 6 秒）
    time_steps = torch.arange(1, num_horzion+1, device=drive_path.device).to(dtype).view(1, -1)  # (6,)
    target_dists = torch.cumsum(speed, dim=1)
    # time_steps = time_steps.repeat(N, 1)  # (N, 6)
    # target_dists = speed * time_steps  # (N, 6)

    # 4. 查找每个目标距离落在哪个段内（返回 seg_idx ∈ [0, 6]）
    seg_idx = torch.searchsorted(cum_dist, target_dists, right=False)  # (N, 6)
    seg_idx_clamped = seg_idx.clamp(1, 6)  # 用于后续 gather，避免越界

    # 5. 获取插值段起止位置（索引）
    gather_idx0 = seg_idx_clamped - 1
    gather_idx1 = seg_idx_clamped

    # 6. 获取插值段起止的绝对位置 (pos0, pos1)
    pos0 = torch.gather(abs_pos, 1, gather_idx0.unsqueeze(-1).expand(-1, -1, 2))  # (N, 6, 2)
    pos1 = torch.gather(abs_pos, 1, gather_idx1.unsqueeze(-1).expand(-1, -1, 2))  # (N, 6, 2)

    # 7. 获取插值段起止的累计距离 (d0, d1)
    d0 = torch.gather(cum_dist, 1, gather_idx0)  # (N, 6)
    d1 = torch.gather(cum_dist, 1, gather_idx1)  # (N, 6)

    # 8. 插值比例
    denom = (d1 - d0).clamp(min=1e-6)  # 避免除0
    ratio = (target_dists - d0) / denom  # (N, 6)

    # 9. 插值位置
    traj = pos0 + ratio.unsqueeze(-1) * (pos1 - pos0)  # (N, 6, 2)

    # 10. 修正边界情况：seg_idx == 0（落在第一段前）
    mask0 = (seg_idx == 0)
    if mask0.any():
        first_vec = seg_vecs[:, 0].unsqueeze(1).expand(-1, num_horzion, -1)  # (N, 6, 2)
        ratio0 = (target_dists / seg_lens[:, 0:1].clamp(min=1e-6))  # (N, 6)
        traj[mask0] = ratio0.unsqueeze(-1)[mask0] * first_vec[mask0]

    # 11. 超出末尾段：seg_idx >= 6
    mask_out = (seg_idx >= drive_path.shape[1])
    if mask_out.any():
        traj[mask_out] = abs_pos[:, -1].unsqueeze(1).expand(-1, num_horzion, -1)[mask_out]

    traj = torch.cat(
        [abs_pos[:, 0:1], traj], dim=1
    )  # 在第一段前加上起点 (N, 7, 2)

    traj = traj[:, 1:] - traj[:, :-1]  # 相对坐标差 (N, 6, 2)

    return traj




















class TrajGen:
    def __init__(self, positions, route_name):
        """
        初始化轨迹生成器

        Args:
            positions: np.ndarray, shape (N, 2), 未来N个轨迹点 (x, y)
            route_name: str, 路径名称或标识
        """
        self.drive_path = np.asarray(positions)
        self.route_name = route_name

        self.spline = None  # 用于保存插值表达式
        self.arc_lengths = None  # 弧长参数化

    def init_drive_path(self):
        """
        对drive_path进行弧长参数化并构建二维样条插值表达式，
        方便后续根据参数（如时间或弧长）获取轨迹点。
        """
        # 1. 计算弧长参数
        deltas = np.diff(self.drive_path, axis=0)
        dists = np.linalg.norm(deltas, axis=1)
        self.arc_lengths = np.insert(np.cumsum(dists), 0, 0.0)

        # 2. 构建二维样条插值 (arc_length -> (x,y))
        # 注意 axis=0 表示对每个维度同时插值，保持轨迹形状
        self.spline = CubicSpline(self.arc_lengths, self.drive_path, axis=0)

    def get_position(self, s):
        """
        根据弧长参数 s 获取轨迹上的点坐标

        Args:
            s: float or np.ndarray, 弧长参数

        Returns:
            coords: np.ndarray, shape (len(s), 2), 轨迹点坐标
        """
        if self.spline is None:
            raise ValueError("请先调用 init_drive_path() 初始化插值表达式")
        return self.spline(s)

    def __call__(self):
        # 这个留着后续根据速度等条件计算轨迹
        pass


def interpolate_traj(trajs, input_frency=2, output_frequency=5):
    # 首先将位移差转换为绝对坐标
    # trajs: [B, T+1, 2] 位移差
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

def compute_min_distance_between_vehicles(trajs, bboxes_batch, fut_bbox_batch, ego_frecy, agent_frecy=2):
    """
    计算两车在时空中最接近时刻的距离
    
    参数:
        trajs: 自车轨迹 [T, 2]
        bboxes_batch: 其他车辆的初始状态 [N, 9] (x, y, z, length, width, height, yaw)
        fut_bbox_batch: 其他车辆的未来轨迹 [N, T, 2] (x, y, length, width)
        
    返回:
        min_distance: 两车在整个时间范围内的最小距离
        min_distance_time: 最小距离发生的时间步
    """
    # 计算Agents的角点
    if ego_frecy != agent_frecy:
        agent_horizon = fut_bbox_batch.shape[1]
        fut_bbox_batch_sampled = interpolate_traj(fut_bbox_batch, input_frency=agent_frecy, output_frequency=ego_frecy)  # B, N, T, 2
        fut_bbox_batch = fut_bbox_batch_sampled[:, :agent_horizon]  # 只保留前6个时间步的轨迹，更新为5hz的轨迹




    fut_bbox_batch = fut_bbox_batch[:, 1:] - fut_bbox_batch[:, :-1]  # 变换为增量形式
    half_fut_bbox_length = bboxes_batch[:, 3] / 2
    half_fut_bbox_width = bboxes_batch[:, 4] / 2
    bbox_init_center = bboxes_batch[:, :2]
    bbox_init_yaw = bboxes_batch[:, 6:7]
    bbox_init_state = torch.cat((bbox_init_center, bbox_init_yaw), dim=-1)
    agent_corners = compute_corners_torch(fut_bbox_batch.clone(), 
                                        half_fut_bbox_length.clone(), 
                                        half_fut_bbox_width.clone(), 
                                        bbox_init_state.clone())  # [N, T, 4, 2]

    # 计算自车的角点
    device = trajs.device
    trajs = trajs[1:] - trajs[:-1]  # 变换为增量形式
    trajs = trajs.unsqueeze(0)  # [1, T-1, 2]
    init_state = torch.zeros((trajs.shape[0], 3), dtype=torch.float32, device=device)
    init_state[:, -1] = 3.14 / 2  # 自车默认90度朝前
    ego_corners = compute_corners_torch(trajs, init_state=init_state)  # [1, T-1, 4, 2]
    ego_corners = rearrange(ego_corners, '(b m) t c d -> b m t c d', b=1, m=1)[0]  # [1, T-1, 4, 2]

    # 确保时间步一致
    min_time_steps = min(agent_corners.shape[1], ego_corners.shape[1])
    
    min_distance = float('inf')
    min_distance_time = -1
    
    # 遍历每个时间步，计算两车的最小距离
    min_distance, min_distance_time, distances = calculate_safe_distance(
        agent_corners[:, :min_time_steps], 
        ego_corners[:, :min_time_steps],
        max_look_ahead=3    # 只看当前时刻的距离
    )
    
    return min_distance, min_distance_time, distances




def calculate_safe_distance(agent_corners, ego_corners, max_look_ahead=1):
    min_distance = float('inf')
    min_distance_time = -1
    distances = [ [] for i in range(ego_corners.shape[1])]  # 初始化距离列表
    
    min_time_steps = min(agent_corners.shape[1], ego_corners.shape[1])
    
    for t in range(min_time_steps):
        current_ego_corners = ego_corners[0, t]  # [4, 2]
        
        # 确定当前t能检查的未来步数（不超过max_look_ahead，且不越界）
        actual_look_ahead = min(max_look_ahead, min_time_steps - t - 1)
        distance_t = float('inf')
        # 检查当前时刻和未来actual_look_ahead个时刻的距离
        for look_ahead in range(actual_look_ahead + 1):  # 0表示当前时刻，1表示t+1，依此类推
            check_time = t + look_ahead
            current_agent_corners = agent_corners[:, check_time]  # [N, 4, 2]
            
            for agent_idx in range(current_agent_corners.shape[0]):
                agent_corner_set = current_agent_corners[agent_idx]  # [4, 2]
                
                dist_matrix = torch.cdist(current_ego_corners.unsqueeze(0), 
                                        agent_corner_set.unsqueeze(0))  # [1, 4, 4]
                current_min_dist = dist_matrix.min().item()
                
                # 记录每个时刻的距离
                distances[t].append(current_min_dist)

                # 只在当前时刻检查最小距离
                if look_ahead == 0:
                    if current_min_dist < min_distance:
                        min_distance = current_min_dist
                        min_distance_time = t        

    return min_distance, min_distance_time, distances




def evaluate_collsion_corners(trajs, bboxes_batch, fut_bbox_batch):

    # 计算Agents的角点
    half_fut_bbox_length = bboxes_batch[:, 3] / 2
    half_fut_bbox_width = bboxes_batch[:, 4] / 2
    bbox_init_center = bboxes_batch[:, :2]
    bbox_init_yaw = bboxes_batch[:, 6:7]
    bbox_init_state = torch.cat((bbox_init_center, bbox_init_yaw), dim=-1)
    agent_corner = compute_corners_torch(fut_bbox_batch.clone(), half_fut_bbox_length.clone(), half_fut_bbox_width.clone(), bbox_init_state.clone()) # 并行维度0

    # 计算自车的角点
    device = trajs.device
    trajs = trajs[1:] -  trajs[:-1] # 变换为增量形式
    trajs = trajs.unsqueeze(0) # b m 6, 2
    # trajs = rearrange(trajs, 'b m (t c) -> (b m) t c', t=6,c=2) # trajs的shape是 [bs, T, 12]，需要变成 [bs, T, 1, 2]，以便和ego_corners对齐
    init_state = torch.zeros((trajs.shape[0], 3), dtype=torch.float32, device=device) # 初始化状态
    init_state[:,-1] = 3.14/2   # 自车默认90度朝前
    ego_corners = compute_corners_torch(trajs, init_state=init_state) # 并行维度0
    ego_corners = rearrange(ego_corners, '(b m) t c d -> b m t c d', b=1, m=1)[0]  # 没有batch
    # ego_corners = 

    collision_matrix, ttc_collision_all, key_agent_corners, key_agent_labels, key_agent_ids, collision_agent = evaluate_coll(agent_corner.cpu(), ego_corners.cpu())
    collision_matrix = collision_matrix[0]  # (6,)

    # 找到第一个变为True的位置，否则返回-1
    first_collision_idx = np.where(collision_matrix)[0][0] if np.any(collision_matrix) else -1

    return first_collision_idx



def generate_collision_positions(trajs, collision_idx, safety_threshold=3.0, min_speed=0.1, distances=None):
    """
    Calculate safe travel distance before potential collision occurs.
    
    Args:
        trajs: (N, 2) array of trajectory positions (x, y)
        collision_idx: Index where collision occurs
        safety_threshold: Minimum safe distance
        min_speed: Minimum speed threshold to consider movement
        distances: List of List[float], where each sublist contains:
                   [d_curr, d_future1, d_future2, ..., d_futureN]
    
    Returns:
        safe_travel_dist: Safe distance the ego vehicle can travel before potential collision
    """
    trajs = trajs[:, :2]
    N = trajs.shape[0]

    # Step 1: Cumulative distance
    diffs = trajs[1:] - trajs[:-1]
    step_dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.insert(np.cumsum(step_dists), 0, 0.0)

    if distances is not None:
        assert len(distances) == (N-1)

        # Step 2: 找到最大可达的安全索引（运动状态下）
        max_safe_motion_idx = -1
        for i in range(min(collision_idx + 1, N)):
            if distances[i][0] > safety_threshold:
                max_safe_motion_idx = i
            else:
                break  # 到了太近的时刻

        # Step 3: 从 max_safe_motion_idx 倒着查，找一个“ego 停下后也不会被撞”的位置
        safe_idx = -1
        for i in range(max_safe_motion_idx, -1, -1):
            future_dists = distances[i][1:]  # 未来几步
            if all(d > safety_threshold for d in future_dists):
                safe_idx = i
                break  # 找到了能静止且未来也安全的位置

        safe_travel_dist = cum_dists[safe_idx] if safe_idx != -1 else 0.0

    else:
        # fallback：直接用碰撞点前一帧
        safe_idx = max(0, collision_idx - 1)
        safe_travel_dist = cum_dists[safe_idx]

    # Step 4: 平均速度检查
    if collision_idx > 0:
        time_to_collision = collision_idx
        avg_speed = cum_dists[collision_idx] / time_to_collision
        if avg_speed < min_speed:
            safe_travel_dist = 0.0

    return safe_travel_dist



def generate_stop_trajs(trajs, safe_travel_dist, safety_threshold=5.0, min_speed=0.1):
    """
    安全优先的防御性轨迹生成函数（在原轨迹上插值，不改变横向形状), 该轨迹固定在碰撞点前方停下
    """
    trajs = trajs[:, :2]
    N = trajs.shape[0]

    # Step 1: 计算累计距离
    diffs = trajs[1:] - trajs[:-1]
    step_dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.insert(np.cumsum(step_dists), 0, 0.0)

    total_dist = cum_dists[-1]
    if safe_travel_dist < min_speed:
        safe_travel_dist = min_speed
    if safe_travel_dist > total_dist:
        safe_travel_dist = total_dist

    # Step 3: 沿原始轨迹插值获得新的轨迹点
    target_dists = np.linspace(0, safe_travel_dist, N)
    new_trajs = np.zeros_like(trajs)
    seg_idx = 0

    for i, d in enumerate(target_dists):
        while seg_idx < len(cum_dists) - 1 and cum_dists[seg_idx + 1] < d:
            seg_idx += 1
        # 当前段：在 trajs[seg_idx] 和 trajs[seg_idx + 1] 之间插值
        seg_start = trajs[seg_idx]
        seg_end = trajs[min(seg_idx + 1, N - 1)]
        seg_len = cum_dists[seg_idx + 1] - cum_dists[seg_idx]
        if seg_len > 1e-4:
            ratio = (d - cum_dists[seg_idx]) / seg_len
        else:
            ratio = 0.0
        new_trajs[i] = seg_start * (1 - ratio) + seg_end * ratio

    # Step 4: 转换为差分形式
    # defensive_trajs = np.zeros_like(new_trajs)
    defensive_trajs = new_trajs[1:] - new_trajs[:-1]
    return defensive_trajs


def generate_two_style_trajs(trajs,min_distance, min_distance_time, distances, stop_trajs):
    # 生成两种风格的轨迹：防御性和攻击性
    # trajs: (N, 2) - 原始轨迹
    # stop_trajs: (N, 2) - 防御性轨迹，该轨迹停在碰撞点前方   
    # 判断distances的变化趋势, 获取distances的第0位
    current_distance = [distances[t][0] for t in range(len(distances))]  # 当前时刻的距离
    current_distance = torch.tensor(current_distance, dtype=torch.float32)

    diffs = torch.diff(current_distance)
    trend = torch.mean(diffs)  # 平均变化率

    # 判断aggressive轨迹是否可以保持原本的trajs
    aggressive_valid_flag = False
    aggressive_trajs = stop_trajs
    if min_distance >= 3:
        aggressive_valid_flag = True
    elif trend > 0.5 and min_distance >= 2:
        aggressive_valid_flag = True
    
    if aggressive_valid_flag:
        # 不会碰撞，返回原始轨迹
        aggressive_trajs = trajs[1:] - trajs[:-1]



    # 判断defensive轨迹是否需要保留原本的trajs
    defensive_valid_flag = False
    defensive_trajs = stop_trajs
    if min_distance >= 4:
        defensive_valid_flag = True
    elif trend > 1 and min_distance >= 3:
        defensive_valid_flag = True
    if defensive_valid_flag:
        # 不会碰撞，返回原始轨迹
        defensive_trajs = trajs[1:] - trajs[:-1]
    
    return defensive_trajs, aggressive_trajs, aggressive_valid_flag


    # print("Min_distance", min_distance)
    # print("Min_distance_time", min_distance_time)
    # print("Distances", distances)
    # print("Trend", trend)










def generate_rear_side_bbox(trajs, idx_range=(2, 6), collision_time_step_range=(2, 8), approach_distance_range=(3, 6), rear_offset_range=(3, 6)):
    # 生成侧后方来车碰撞的情况
    num_t = len(trajs)
    collision_idx = np.random.randint(idx_range[0], idx_range[1])
    collision_point = trajs[collision_idx]
    
    # 确定侧后方方向 (根据自车横向位移方向决定)
    side_direction = 1 if trajs[-1, 0] > trajs[0, 0] else -1  # 向右变道则从右侧来车，反之亦然
    
    # 生成侧后方来车参数
    approach_distance = np.random.uniform(approach_distance_range[0], approach_distance_range[1])
    rear_offset = np.random.uniform(rear_offset_range[1], rear_offset_range[1])  # 后方偏移量
    
    # 假设他车的碰撞时间, 比较早的发生碰撞
    collision_time_step = np.random.randint(collision_time_step_range[0], collision_time_step_range[1])


    # 计算假车初始位置 (侧后方)
    ego_postion = np.array([0, 0])  # 假设自车初始位置在原点
    fake_init_pos =  ego_postion + np.array([side_direction * approach_distance, -rear_offset])
    
    # 简化版轨迹生成 - 直线运动到碰撞点
    fake_traj = np.zeros((num_t, 2))
    for i in range(num_t):
        # 调整alpha的计算，使得在collision_time_step时到达碰撞点
        if i <= collision_time_step:
            alpha = i / collision_time_step if collision_time_step > 0 else 1
            fake_traj[i] = fake_init_pos * (1-alpha) + collision_point * alpha
        else:
            # 碰撞后的轨迹可以继续沿相同方向保持相同的速度
            # 计算碰撞前的速度向量（最后一个点与碰撞点的差值）
            if collision_time_step > 0:
                velocity_vector = collision_point - fake_traj[collision_time_step - 1]
            else:
                velocity_vector = collision_point - fake_init_pos
            
            # 碰撞后保持相同速度继续移动
            steps_after_collision = i - collision_time_step
            fake_traj[i] = collision_point + velocity_vector * steps_after_collision

    yaw = np.arctan2(fake_traj[1, 1] - fake_traj[0, 1], fake_traj[1, 0] - fake_traj[0, 0])
    velocity = np.linalg.norm(fake_traj[-1] - fake_traj[0]) / 3  # 假设3秒内到达

    return fake_init_pos, fake_traj, yaw, velocity, collision_idx


def genereate_close_bbox(trajs, idx_range=(2, 6), collision_time_step=(2, 4), close_distance=(4, 7)):
    # 他车从横向近处通过，这种情况一般需要停车等待
    num_t = len(trajs)

    collision_idx = np.random.randint(idx_range[0], idx_range[1])
    collision_point = trajs[collision_idx]
    # 假设他车的碰撞时间, 比较早的发生碰撞
    collision_time_step = np.random.randint(collision_time_step[0], collision_time_step[1])

    approach_distance = np.random.uniform(close_distance[0], close_distance[1])
    # 角度更大概率与自车垂直
    if np.random.rand() < 0.5:
        collision_angle =  np.random.uniform(-np.pi / 6, np.pi / 6)  
        fake_init_pos = collision_point + np.array([
                approach_distance * np.cos(collision_angle + np.pi),
                approach_distance * np.sin(collision_angle + np.pi)
            ])

    else:
        collision_angle =  np.random.uniform(-np.pi / 6, np.pi / 6)  
        fake_init_pos = collision_point + np.array([
                approach_distance * np.cos(collision_angle),
                approach_distance * np.sin(collision_angle)
            ])
    
    # 生成他车轨迹，确保在指定时间步到达碰撞点
    fake_traj = np.zeros((num_t, 2))
    for i in range(num_t):
        # 调整alpha的计算，使得在collision_time_step时到达碰撞点
        if i <= collision_time_step:
            alpha = i / collision_time_step if collision_time_step > 0 else 1
            fake_traj[i] = fake_init_pos * (1-alpha) + collision_point * alpha
        else:
            # 碰撞后的轨迹可以继续沿相同方向保持相同的速度
            # 计算碰撞前的速度向量（最后一个点与碰撞点的差值）
            if collision_time_step > 0:
                velocity_vector = collision_point - fake_traj[collision_time_step - 1]
            else:
                velocity_vector = collision_point - fake_init_pos
            
            # 碰撞后保持相同速度继续移动
            steps_after_collision = i - collision_time_step
            fake_traj[i] = collision_point + velocity_vector * steps_after_collision

    yaw = np.arctan2(fake_traj[1, 1] - fake_traj[0, 1], fake_traj[1, 0] - fake_traj[0, 0])
    velocity = np.linalg.norm(fake_traj[-1] - fake_traj[0]) / 3  # 假设3秒内到达

    return fake_init_pos, fake_traj, yaw, velocity, collision_idx

def generate_far_bbox(trajs, idx_range=(2, 6), collision_time_step_range=(4, 12), approach_distance_range=(5, 15)):
    # 他车从较远处通过，碰撞时间较晚或者者不发生碰撞
    num_t = len(trajs)

    collision_idx = np.random.randint(idx_range[0], idx_range[1])
    collision_point = trajs[collision_idx]
    # 假设他车的碰撞时间, 比较晚的发生碰撞
    collision_time_step = np.random.randint(collision_time_step_range[0], collision_time_step_range[1])
    approach_distance = np.random.uniform(approach_distance_range[0], approach_distance_range[1])

    # 角度更大概率与自车垂直
    if np.random.rand() < 0.5:
        collision_angle =  np.random.uniform(-np.pi / 6, np.pi / 6)  
        fake_init_pos = collision_point + np.array([
                approach_distance * np.cos(collision_angle + np.pi),
                approach_distance * np.sin(collision_angle + np.pi)
            ])

    else:
        collision_angle =  np.random.uniform(-np.pi / 6, np.pi / 6)  
        fake_init_pos = collision_point + np.array([
                approach_distance * np.cos(collision_angle),
                approach_distance * np.sin(collision_angle)
            ])

    
    # 生成他车轨迹，确保在指定时间步到达碰撞点
    fake_traj = np.zeros((num_t, 2))
    for i in range(num_t):
        # 调整alpha的计算，使得在collision_time_step时到达碰撞点
        if i <= collision_time_step:
            alpha = i / collision_time_step if collision_time_step > 0 else 1
            fake_traj[i] = fake_init_pos * (1-alpha) + collision_point * alpha
        else:
            # 碰撞后的轨迹可以继续沿相同方向保持相同的速度
            # 计算碰撞前的速度向量（最后一个点与碰撞点的差值）
            if collision_time_step > 0:
                velocity_vector = collision_point - fake_traj[collision_time_step - 1]
            else:
                velocity_vector = collision_point - fake_init_pos
            
            # 碰撞后保持相同速度继续移动
            steps_after_collision = i - collision_time_step
            fake_traj[i] = collision_point + velocity_vector * steps_after_collision
                
            

    yaw = np.arctan2(fake_traj[1, 1] - fake_traj[0, 1], fake_traj[1, 0] - fake_traj[0, 0])
    velocity = np.linalg.norm(fake_traj[-1] - fake_traj[0]) / 3  # 假设3秒内到达

    return fake_init_pos, fake_traj, yaw, velocity, collision_idx






def generate_fake_bbox_v2(drive_path, trajs, trajs_5hz, bboxes, bbox_trajs, safety_threshold=3.0):
    """
    生成一个与自车轨迹未来会碰撞但不与其他车辆碰撞的假车辆
    新增侧后方来车情况的处理
    
    参数:
        drive_path: (6,2) 自车按距离采样的轨迹, 记录的是两点之间的offset
        trajs: (6,2) 自车按时间采样的轨迹, 记录的是两点之间的offset
        trajs_5hz: (N,2) 自车按5Hz采样的轨迹, 记录的是两点之间的offset
        bboxes: (N,9) 现有车辆的bbox信息 [x,y,z,w,h,l,yaw,v,a]
        bbox_trajs: (N,9,6,2) 现有车辆的未来轨迹offset, 记录的是两点之间的offset
        safety_threshold: float 安全距离阈值(米)
        
    返回:
        fake_bbox: (9,) 假车辆的bbox信息
        fake_traj: (6,2) 假车辆的未来轨迹
        modified_traj: (6,2) 修改后的自车轨迹(在安全距离前停止)
        modified_traj_5hz: (6,2) 修改后的自车5Hz轨迹(在安全距离前停止)
        valid_flag: bool 是否有效
        aggressive_valid: bool 激进轨迹是否有效
    """
    # 0.将相对量转换为绝对量
    drive_path = np.cumsum(drive_path, axis=0)
    drive_path = np.concatenate([np.zeros((1, 2)), drive_path], axis=0)
    trajs = np.cumsum(trajs, axis=0)
    trajs = np.concatenate([np.zeros((1, 2)), trajs], axis=0)
    trajs_5hz = np.cumsum(trajs_5hz, axis=0)
    trajs_5hz = np.concatenate([np.zeros((1, 2)), trajs_5hz], axis=0)
    bbox_trajs = np.cumsum(bbox_trajs, axis=0)
    bboxes_posi = bboxes[:, :2]
    bboxes_posi = np.expand_dims(bboxes_posi, axis=1)
    bbox_trajs = bboxes_posi + bbox_trajs
    valid_flag = True
    aggressive_valid = True

    # 0. 如果自车轨迹本身就过短，定义一个valid flag为False
    max_bbox_trajs = np.linalg.norm(trajs[-1])
    if max_bbox_trajs.max() < 5:
        valid_flag = False
        aggressive_valid = False

    # 1. 判断自车是否处于横向位移较大的情况（可能变道或转弯）
    lateral_movement = np.abs(trajs[-1, 0] - trajs[0, 0])  # y轴位移
    longitudinal_movement = np.abs(trajs[-1, 1] - trajs[0, 1])  # x轴位移
    # is_lateral_movement = lateral_movement > 0.3 * longitudinal_movement
    # is_lateral_movement还需要考虑绝对横向位移量
    is_lateral_movement = lateral_movement > 2



    # 2. 根据是否横向位移决定生成前方碰撞车还是侧后方来车
    # fake_init_pos, fake_traj, yaw, velocity, collision_idx = generate_far_bbox(trajs)
    # fake_init_pos, fake_traj, yaw, velocity, collision_idx = genereate_close_bbox(trajs)
    # fake_init_pos, fake_traj, yaw, velocity, collision_idx = generate_rear_side_bbox(trajs)

    if is_lateral_movement:
        rand_val = np.random.rand()
        if rand_val > 0.5:  # 50%概率生成侧后方来车
            fake_init_pos, fake_traj, yaw, velocity, collision_idx = generate_rear_side_bbox(trajs)
        elif rand_val > 0.25:  # 25%概率生成近距离通过
            fake_init_pos, fake_traj, yaw, velocity, collision_idx = genereate_close_bbox(trajs)
        else:  # 25%概率生成远距离通过
            fake_init_pos, fake_traj, yaw, velocity, collision_idx = generate_far_bbox(trajs)
    else:
        # 如果不是侧向移动的情况
        choice = np.random.choice(2)
        if choice == 0:  # 33%概率生成侧后方来车
            fake_init_pos, fake_traj, yaw, velocity, collision_idx = generate_far_bbox(trajs)
        elif choice == 1:  # 33%概率生成近距离通过
            fake_init_pos, fake_traj, yaw, velocity, collision_idx = genereate_close_bbox(trajs)





    # 3. 生成假车辆bbox信息
    vehicle_width = np.random.uniform(1.5, 2.5)
    vehicle_length = np.random.uniform(4.0, 5.5)
    vehicle_height = np.random.uniform(1.4, 2.0)
    fake_bbox = np.array([
        fake_init_pos[0], fake_init_pos[1], 0,
        vehicle_length, vehicle_width, vehicle_height,
        yaw, velocity, 0
    ])

    # 4.转换为相对原始位置的轨迹
    fake_init_pos = fake_init_pos.reshape(-1, 2)
    fake_traj = fake_traj - fake_init_pos

    # 5. 计算轨迹与插入车辆的最小距离
    min_distance_2hz, min_distance_time_2hz, distances_2hz = compute_min_distance_between_vehicles(torch.tensor(trajs), torch.tensor(fake_bbox.reshape(-1, 9)), torch.tensor(fake_traj).unsqueeze(0))
    min_distance_5hz, min_distance_time_5hz, distances_5hz = compute_min_distance_between_vehicles(torch.tensor(trajs_5hz), torch.tensor(fake_bbox.reshape(-1, 9)), torch.tensor(fake_traj).unsqueeze(0))




    safe_travel_dist = generate_collision_positions(trajs, collision_idx, safety_threshold, distances=distances_2hz)
    # print("Safe travel distance:", safe_travel_dist)
    # print("distance 2Hz:", distances_2hz)

    # 处理2Hz轨迹
    stop_trajs_2hz = generate_stop_trajs(trajs, safe_travel_dist, safety_threshold)
    defensive_trajs, aggressive_trajs, aggressive_valid = generate_two_style_trajs(trajs, min_distance_2hz, min_distance_time_2hz, distances_2hz, stop_trajs_2hz)

    # 处理5Hz轨迹
    stop_trajs_5hz = generate_stop_trajs(trajs_5hz, safe_travel_dist, safety_threshold)
    defensive_trajs_5hz, aggressive_trajs_5hz, aggressive_valid_5hz = generate_two_style_trajs(trajs_5hz, min_distance_5hz, min_distance_time_5hz, distances_5hz, stop_trajs_5hz)
    # aggressive_valid_5hz暂时没使用

    # 6. 检查与其他车辆碰撞
    # collision_mask = check_collisions(fake_bbox, fake_traj[1:], bboxes, bbox_trajs)
    # if collision_mask:
    #     valid_flag = False 

    # 7.fake_traj转换为相对量, ego的traj已经转换了
    fake_bbox = fake_bbox.reshape(-1, 9)
    fake_traj = fake_traj[1:] - fake_traj[:-1]

    return fake_bbox, fake_traj, defensive_trajs, aggressive_trajs, defensive_trajs_5hz, aggressive_trajs_5hz, valid_flag, aggressive_valid


def generate_bbox_attritube(trajs, drive_path, paramterss):

    # 1. 判断自车是否处于横向位移较大的情况（可能变道或转弯）
    lateral_movement = np.abs(trajs[-1, 0] - trajs[0, 0])  # y轴位移
    longitudinal_movement = np.abs(trajs[-1, 1] - trajs[0, 1])  # x轴位移
    # is_lateral_movement = lateral_movement > 0.3 * longitudinal_movement
    # is_lateral_movement还需要考虑绝对横向位移量
    is_lateral_movement = lateral_movement > 2

    rear_side_bbox_collision_time_range=paramterss['rear_side_bbox_collision_time_range']
    rear_side_bbox_collision_range=paramterss['rear_side_bbox_collision_range']
    rear_approach_distance_range= paramterss['rear_approach_distance_range']  # 侧后方来车的接近距离范围
    rear_offset_range=paramterss['rear_offset_range']  # 侧后方来车的后方偏移范围

    close_bbox_collision_time_range=paramterss['close_bbox_collision_time_range']  # 近距离通过碰撞时间范围
    close_bbox_collision_range=paramterss['close_bbox_collision_range']  # 近距离通过碰撞时间范围
    close_distance=paramterss['close_distance']

    far_bbox_collision_time_range=paramterss['far_bbox_collision_time_range']  # 远距离通过碰撞时间范围
    far_bbox_collision_range=paramterss['far_bbox_collision_range']
    far_bbox_distance_range=paramterss['far_bbox_distance_range']   # 8, 15






    if is_lateral_movement:
        rear_side_pro = 0.5
        close_bbox_pro = 0.25
        far_bbox_pro = 0.25
    else:
        rear_side_pro = 0.25
        close_bbox_pro = 0.25
        far_bbox_pro = 0.5



    # 根据概率随机选择生成方式
    choice = np.random.choice(['rear_side', 'close', 'far'], 
                            p=[rear_side_pro, close_bbox_pro, far_bbox_pro])

    if choice == 'rear_side':
        fake_init_pos, fake_traj, yaw, velocity, collision_idx = generate_rear_side_bbox(
            drive_path, 
            rear_side_bbox_collision_range,
            rear_side_bbox_collision_time_range, 
            rear_approach_distance_range, 
            rear_offset_range
        )
    elif choice == 'close':
        fake_init_pos, fake_traj, yaw, velocity, collision_idx = genereate_close_bbox(
            drive_path, 
            close_bbox_collision_range, 
            close_bbox_collision_time_range, 
            close_distance
        )
    else:  # choice == 'far'
        fake_init_pos, fake_traj, yaw, velocity, collision_idx = generate_far_bbox(
            drive_path, 
            far_bbox_collision_range,
            far_bbox_collision_time_range,
            far_bbox_distance_range,
        )

    # fake_init_pos, fake_traj, yaw, velocity, collision_idx = generate_far_bbox(
    #     drive_path, 
    #     far_bbox_collision_range,
    #     far_bbox_collision_time_range,
    #     far_bbox_distance_range,
    # )

    # 3. 生成假车辆bbox信息
    vehicle_width = np.random.uniform(1.5, 2.5)
    vehicle_length = np.random.uniform(4.0, 5.5)
    vehicle_height = np.random.uniform(1.4, 2.0)
    fake_bbox = np.array([
        fake_init_pos[0], fake_init_pos[1], 0,
        vehicle_length, vehicle_width, vehicle_height,
        yaw, velocity, 0
    ])

    # 4.转换为相对原始位置的轨迹
    fake_init_pos = fake_init_pos.reshape(-1, 2)
    fake_traj = fake_traj - fake_init_pos



    output_dict = {}
    output_dict['fake_init_pos'] = fake_init_pos
    output_dict['fake_traj'] = fake_traj
    output_dict['vehicle_width'] = vehicle_width
    output_dict['vehicle_length'] = vehicle_length
    output_dict['vehicle_height'] = vehicle_height
    output_dict['yaw'] = yaw
    output_dict['velocity'] = velocity
    output_dict['collision_idx'] = collision_idx
    output_dict['fake_bbox'] = fake_bbox
    

    return output_dict


def resample_trajs(trajs, input_freq, output_freq):
    """
    trajs: (T, 2) 的二维轨迹，等时间间隔采样，第一个点为 (0,0)
    input_freq: 原始采样频率（Hz）
    output_freq: 目标采样频率（Hz）

    return: 插值后的轨迹，形状为 (T_new, 2)
    """

    T = trajs.shape[0]
    duration = (T - 1) / input_freq  # 总时长（秒）

    # 计算每个点对应的累积时间
    t_input = np.linspace(0, duration, T)

    # 输出轨迹的目标时间戳
    T_new = int(duration * output_freq) + 1
    t_output = np.linspace(0, duration, T_new)

    # 在时间维度上插值，每个时间戳落在哪两个原始点之间
    new_trajs = np.zeros((T_new, 2))
    for i, t in enumerate(t_output):
        # 找到当前t对应的插值区间 [t_input[j], t_input[j+1]]
        idx = np.searchsorted(t_input, t) - 1
        idx = np.clip(idx, 0, T - 2)

        t0, t1 = t_input[idx], t_input[idx + 1]
        ratio = (t - t0) / (t1 - t0 + 1e-8)  # 避免除0

        p0, p1 = trajs[idx], trajs[idx + 1]
        new_trajs[i] = p0 + ratio * (p1 - p0)

    return new_trajs




def generate_fake_bbox(drive_path, trajs, trajs_5hz, bboxes, bbox_trajs, safety_threshold=3.0):
    """
    生成一个与自车轨迹未来会碰撞但不与其他车辆碰撞的假车辆
    新增侧后方来车情况的处理
    
    参数:
        drive_path: (6,2) 自车按距离采样的轨迹, 记录的是两点之间的offset
        trajs: (6,2) 自车按时间采样的轨迹, 记录的是两点之间的offset 
        trajs_5hz: (N,2) 自车按5Hz采样的轨迹, 记录的是两点之间的offset
        bboxes: (N,9) 现有车辆的bbox信息 [x,y,z,w,h,l,yaw,v,a]
        bbox_trajs: (N,9,6,2) 现有车辆的未来轨迹offset, 记录的是两点之间的offset
        safety_threshold: float 安全距离阈值(米)
        
    返回:
        fake_bbox: (9,) 假车辆的bbox信息
        fake_traj: (6,2) 假车辆的未来轨迹
        modified_traj: (6,2) 修改后的自车轨迹(在安全距离前停止)
        modified_traj_5hz: (6,2) 修改后的自车5Hz轨迹(在安全距离前停止)
        valid_flag: bool 是否有效
        aggressive_valid: bool 激进轨迹是否有效
    """
    # 0.将相对量转换为绝对量
    drive_path = np.cumsum(drive_path, axis=0)
    drive_path = np.concatenate([np.zeros((1, 2)), drive_path], axis=0)
    trajs = np.cumsum(trajs, axis=0)
    trajs = np.concatenate([np.zeros((1, 2)), trajs], axis=0)
    trajs_5hz = np.cumsum(trajs_5hz, axis=0)
    trajs_5hz = np.concatenate([np.zeros((1, 2)), trajs_5hz], axis=0)
    bbox_trajs = np.cumsum(bbox_trajs, axis=0)
    bboxes_posi = bboxes[:, :2]
    bboxes_posi = np.expand_dims(bboxes_posi, axis=1)
    bbox_trajs = bboxes_posi + bbox_trajs
    valid_flag = True
    aggressive_valid = True

    # 0. 如果自车轨迹本身就过短，定义一个valid flag为False
    max_bbox_trajs = np.linalg.norm(trajs[-1])
    if max_bbox_trajs.max() < 5:
        valid_flag = False
        aggressive_valid = False



    paramters_dict = {
        # 参数没有调，肯定存在问题
        'rear_side_bbox_collision_range': (2, 6),  # 
        'rear_approach_distance_range':(3, 6), 
        'rear_side_bbox_collision_time_range': (2, 6),  # 
        'rear_offset_range': (3, 6),  # '      
        'close_bbox_collision_time_range': (2, 6),  # 
        'close_bbox_collision_range': (1, 3),  # 
        'close_distance': (4, 8),  # 近距离的距离
        'far_bbox_collision_time_range': (2, 6),  # 
        'far_bbox_collision_range': (4, 6),  # 
        'far_bbox_distance_range': (8, 15)  # 
    }

    if drive_path.shape[0] <= 10:
        paramters_dict_5hz = {  # 适配1.5S的5Hz轨迹
            # rear tested
            'rear_side_bbox_collision_range': (4, 5),  # 发生碰撞的是自车轨迹上的哪个点哪个点
            'rear_side_bbox_collision_time_range': (2, 8),  # 发生碰撞的是什么时候，可以大于6 注意，超过3就意味着1.5s内不会碰撞, 创造部分太远的车，避免完全不起步
            'rear_approach_distance_range':(3, 4),     # 侧后方来车的横向距离
            'rear_offset_range': (5, 6),            # 侧后方来车的纵向距离

            # close tested
            'close_bbox_collision_range': (3, 5),  # 发生碰撞的是自车轨迹上的哪个点哪个点
            'close_bbox_collision_time_range': (0, 4),  # 发生碰撞的是什么时候，可以大于6
            'close_distance': (3, 5),  # 近距离车辆的距离

            # far tested
            'far_bbox_collision_range': (2, 3),  # 发生碰撞的是自车轨迹上的哪个点哪个点
            'far_bbox_collision_time_range': (5, 8),  # 发生碰撞的是什么时候，可以大于6
            'far_bbox_distance_range': (8, 15)  # 远距离通过的距离范围
        }
    else:
        paramters_dict_5hz = {  # 适配3S的5Hz轨迹
            # rear tested 用于加强模型对侧后方车辆的响应
            'rear_side_bbox_collision_range': (5, 6),  # 发生碰撞的是自车轨迹上的哪个点哪个点
            'rear_side_bbox_collision_time_range': (4, 20),  # 发生碰撞的是什么时候，可以大于6 注意，超过3就意味着1.5s内不会碰撞, 创造部分太远的车，避免完全不起步
            'rear_approach_distance_range':(3, 4),     # 侧后方来车的横向距离
            'rear_offset_range': (5, 6),            # 侧后方来车的纵向距离

            # close tested 用于模型加强面对近处车辆的急刹能力
            'close_bbox_collision_range': (3, 10),  # 发生碰撞的是自车轨迹上的哪个点哪个点
            'close_bbox_collision_time_range': (2, 8),  # 发生碰撞的是什么时候，可以大于6
            'close_distance': (4, 9),  # 近距离车辆的距离

            # far tested 用于让模型意识到太远的车不需要刹ting
            'far_bbox_collision_range': (2, 5),         # 发生碰撞的是自车轨迹上的哪个点哪个点
            'far_bbox_collision_time_range': (10, 18),  # 发生碰撞的是什么时候，可以大于6
            'far_bbox_distance_range': (20, 25)         # 远距离通过的距离范围. 这两个值要配合一下，一般车辆的速度都是8m/s往上的
        }


    # generate_bbox_attritube中默认总共进行时是3S，所以如果是只预测1.5S的轨迹，速度属性是错误的。 暂时没有修改为更鲁棒的写法，因为模型使用的是顶点，没有直接使用速度属性
    generated_bbox = generate_bbox_attritube(trajs, drive_path, paramters_dict)
    generated_bbox_5hz = generate_bbox_attritube(trajs_5hz, drive_path, paramters_dict_5hz)

    # 5. 计算轨迹与插入车辆的最小距离
    min_distance_2hz, min_distance_time_2hz, distances_2hz = compute_min_distance_between_vehicles(torch.tensor(trajs), torch.tensor(generated_bbox['fake_bbox'].reshape(-1, 9)), torch.tensor(generated_bbox['fake_traj']).unsqueeze(0), ego_frecy=2)
    min_distance_5hz, min_distance_time_5hz, distances_5hz = compute_min_distance_between_vehicles(torch.tensor(trajs_5hz), torch.tensor(generated_bbox_5hz['fake_bbox'].reshape(-1, 9)), torch.tensor(generated_bbox_5hz['fake_traj']).unsqueeze(0), ego_frecy=5)


    safe_travel_dist = generate_collision_positions(trajs, generated_bbox['collision_idx'], safety_threshold, distances=distances_2hz)
    safe_travel_dist_5hz = generate_collision_positions(trajs_5hz, generated_bbox_5hz['collision_idx'], safety_threshold, distances=distances_5hz)

    # print("Safe travel distance:", safe_travel_dist)
    # print("distance 2Hz:", distances_2hz)

    # 处理2Hz轨迹
    stop_trajs_2hz = generate_stop_trajs(trajs, safe_travel_dist, safety_threshold)
    defensive_trajs, aggressive_trajs, aggressive_valid = generate_two_style_trajs(trajs, min_distance_2hz, min_distance_time_2hz, distances_2hz, stop_trajs_2hz)

    # 处理5Hz轨迹
    stop_trajs_5hz = generate_stop_trajs(trajs_5hz, safe_travel_dist_5hz, safety_threshold)
    defensive_trajs_5hz, aggressive_trajs_5hz, aggressive_valid_5hz = generate_two_style_trajs(trajs_5hz, min_distance_5hz, min_distance_time_5hz, distances_5hz, stop_trajs_5hz)

    fake_bbox = generated_bbox['fake_bbox'].reshape(-1, 9)
    fake_traj = generated_bbox['fake_traj'][1:] - generated_bbox['fake_traj'][:-1]
    # fake_traj = fake_traj[None, ...]

    if generated_bbox_5hz['fake_traj'].shape[0] > 7: # 输出的是2Hz的轨迹，降采样维为5hz
        resampled_trajs = resample_trajs(generated_bbox_5hz['fake_traj'], input_freq=5, output_freq=2)
        generated_bbox_5hz['fake_traj'] = resampled_trajs
        

    fake_bbox_5hz = generated_bbox_5hz['fake_bbox'].reshape(-1, 9)
    fake_traj_5hz = generated_bbox_5hz['fake_traj'][1:] - generated_bbox_5hz['fake_traj'][:-1]
    # fake_traj_5hz = fake_traj_5hz[None, ...]



    return fake_bbox, fake_traj, fake_bbox_5hz, fake_traj_5hz, defensive_trajs, aggressive_trajs, defensive_trajs_5hz, aggressive_trajs_5hz, valid_flag, aggressive_valid

















def check_collisions(fake_bbox, fake_traj, bboxes, bbox_trajs):
    """
    检查假车辆是否与现有车辆发生碰撞
    """
    # 简化的碰撞检测 - 实际实现需要更精确
    for i in range(len(bboxes)):
        other_bbox = bboxes[i]
        other_traj = bbox_trajs[i]
        
        # 检查初始位置是否太近
        init_dist = np.linalg.norm(fake_bbox[:2] - other_bbox[:2])
        if init_dist < 5.0:  # 初始位置至少5米远
            return True
        
        # 检查轨迹点是否太近
        for t in range(fake_traj.shape[0]):
            fake_pos = fake_traj[t]
            other_pos = other_bbox[:2] + other_traj[t]
            if np.linalg.norm(fake_pos - other_pos) < 3.0:  # 轨迹点至少3米远

                return True
    return False







def project_agents_to_frenet(drivepath, agents):
    """
    将其他车辆的轨迹投影到自车路径的Frenet坐标系
    
    Args:
        drivepath: (N, T, 2) 自车的N条候选轨迹，每条轨迹含T个时间步的(x,y)坐标
        agents: (M, 6, 2) 其他车辆在未来6个时刻的(x,y)轨迹点
    
    Returns:
        projected_agents: (N, M, T, 2) 投影后的Frenet坐标(s,d)，其中:
                         - [...,0]为纵向坐标s（沿参考路径的弧长）
                         - [...,1]为横向坐标d（带符号的垂直距离）
    """
    N, T, _ = drivepath.shape
    M, _, _ = agents.shape
    projected_agents = np.zeros((N, M, T, 2))
    distance_threshold = 20   # 10m外的车不考虑
    # 1. 对每条自车轨迹构建参考路径的样条插值
    for n in range(N):
        # 自车轨迹的绝对坐标 (T,2)
        ego_path = drivepath[n]
        ego_path = np.concatenate([np.zeros((1, 2)), ego_path], axis=0)  # 添加起点(0,0)以便计算弧长
        # 计算累积弧长s（严格递增）
        s = np.zeros(T+1)
        for t in range(1, T+1):
            s[t] = s[t-1] + np.linalg.norm(ego_path[t] - ego_path[t-1])
        # s = s[1:]
        # ego_path = ego_path[1:]  # 去掉起点(0,0)
        # 确保s严格递增（避免重复点导致的问题）
        s = np.maximum.accumulate(s) + 1e-6 * np.arange(T+1)  # 添加微小扰动

        # 构建样条插值
        cs_x = CubicSpline(s, ego_path[:, 0])
        cs_y = CubicSpline(s, ego_path[:, 1])

        # 2. 对他车轨迹的每个点进行投影
        for m in range(M):
            # 他车绝对轨迹 (6,2)
            agent_path = agents[m]
            
            # 检查他车是否在自车路径附近（粗略过滤）
            min_distance = np.min([
                np.linalg.norm(agent_path[t]) 
                for t in range(T)
            ])
            if min_distance > distance_threshold:
                projected_agents[n, m, :] = [s[-1], distance_threshold]  # 标记为远处
                continue  # 跳过优化

            for t in range(T):
                # 若他车轨迹时间步不足，跳过（或插值补齐）
                if t >= agents.shape[1]:
                    projected_agents[n, m, t] = [s[-1], 0.0]
                    continue
                
                point = agent_path[t]  # 他车在第t时刻的位置
                
                # 3. 优化目标：找到使距离最小的s_optimal
                def distance_func(s_query):
                    path_point = np.array([cs_x(s_query), cs_y(s_query)])
                    return np.linalg.norm(point - path_point)
                
                # 初始猜测：用当前时间步对应的s值（假设时间对齐）
                s_guess = s[min(t, T-1)]
                
                # 约束s在有效范围内
                # bounds = [(max(0, s_guess - distance_threshold), min(s[-1], s_guess + distance_threshold))]
                search_range = min(distance_threshold, s[-1]/2)
                bounds = [ (s_guess - search_range, s_guess + search_range)]

                # 数值优化（L-BFGS-B方法加速）
                res = minimize(distance_func, s_guess, bounds=bounds, method='L-BFGS-B')
                s_optimal = res.x[0]
                
                # 4. 计算横向距离d（带符号）
                r = np.array([cs_x(s_optimal), cs_y(s_optimal)])  # 参考路径上的最近点
                dr = np.array([cs_x(s_optimal, 1), cs_y(s_optimal, 1)])  # 切向量（一阶导）
                n_f = np.array([-dr[1], dr[0]])  # 法向量
                n_f /= (np.linalg.norm(n_f) + 1e-6)  # 归一化，避免除零
                
                d = np.dot(point - r, n_f)  # 带符号的横向距离
                
                projected_agents[n, m, t] = [s_optimal, d]
    
    return projected_agents


def find_critical_agents(projected_agents, ego_trajs, ego_speed=None, s_thresh=5.0, d_thresh=3):
    """
    根据Frenet坐标筛选关键交互物体
    
    Args:
        projected_agents: (N, M, T, 2) 投影后的(s,d)坐标
        ego_trajs: (N, T, 2) 自车轨迹两点之间的offset
        ego_speed: 自车速度（可选，用于运动趋势判断）
        s_thresh: 纵向安全阈值（米）
        d_thresh: 横向安全阈值（米）
    
    Returns:
        critical_agents: 列表，每个元素为元组 (自车轨迹索引n, 他车索引m, 冲突时间步列表t)
    """
    N, M, T, _ = projected_agents.shape
    critical_agents = []
    
    for n in range(N):
        critical_agents_n = []
        s_ego = np.zeros(T+1)
        for t in range(1, T):
            s_ego[t] = s_ego[t-1] + np.linalg.norm(ego_trajs[n][t])
        s_ego = s_ego[1:]  # 去掉最后一个时间步的累积距离

        for m in range(M):
            conflict_steps = []
            
            for t in range(T):
                s_other, d_other = projected_agents[n, m, t]
                
                # 条件1：纵向冲突
                longitudinal_conflict = (abs(s_other - s_ego[t]) < s_thresh)
                
                # 条件2：横向冲突
                lateral_conflict = (abs(d_other) < d_thresh)
                
                # 条件3：运动趋势（需速度信息）
                trend_conflict = False
                if ego_speed is not None and t < T-1:
                    # 计算他车相对s方向的速度
                    s_next = projected_agents[n, m, t+1, 0]
                    v_other_s = s_next - s_other  # 他车纵向速度（近似）
                    v_relative_s = v_other_s - ego_speed
                    trend_conflict = (v_relative_s * (s_other - s_ego[t])) < 0
                
                # 记录冲突时间步
                if longitudinal_conflict and (lateral_conflict or trend_conflict):
                    conflict_steps.append(t)
            
            if len(conflict_steps) > 0:
                critical_agents_n.append([m, conflict_steps])
            else:
                # 如果没有冲突时间步，仍然记录他车索引
                critical_agents_n.append([m, []])
        critical_agents.append(critical_agents_n)
    return critical_agents
