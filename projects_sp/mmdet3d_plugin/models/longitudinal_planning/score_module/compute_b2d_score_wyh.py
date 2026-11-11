import os
import numpy as np
import torch
from shapely.geometry import Polygon
from shapely.creation import linestrings
from shapely import Point, creation
from shapely.strtree import STRtree
import shapely
from shapely import LineString
from einops import rearrange
import time


def safe_atan2(y, x, eps=1e-6):
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
        y = y.float()
    # 添加保护机制防止零除和数值不稳定
    mask = (x.abs() > eps) | (y.abs() > eps)
    x_safe = torch.where(mask, x, torch.ones_like(x)*eps)
    y_safe = torch.where(mask, y, torch.ones_like(y)*eps)
    # 添加梯度保护
    x_safe = x_safe.clamp(min=-1e6, max=1e6)
    y_safe = y_safe.clamp(min=-1e6, max=1e6)

    magnitude = torch.sqrt(x**2 + y**2)
    distance_mask = magnitude < 1e-1  # 小于1cm的位移
    return torch.atan2(y_safe, x_safe), mask, distance_mask


def compute_corners_torch(proposals, half_lengths=None, half_widths=None, init_state=None, return_trajs=False):
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
    raw_angles, _, mask = safe_atan2(deltas[..., 1], deltas[..., 0])  # dy, dx

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

    if return_trajs:
        return corners_global, positions
    else:
        return corners_global


@torch.no_grad()
def evaluate_coll_fast(fut_box_corners: torch.Tensor,  _ego_coords: torch.Tensor):
    """
    GPU 上的矩形-矩形精确距离/碰撞检测，替代 CPU shapely/STRtree。

    fut_box_corners: [N, T, K, 2]  (K>=4，默认取前4点作为矩形四点)
    _ego_coords    : [M, T, K, 2]

    返回 (两个，且只有这两个被使用)：
      collision_all      [M, T]     (torch.bool)  —— 任一有效他车在该时刻与自车相交
      min_distance_matrix [M, N, T] (torch.float32) —— 相交记 0，分离为最小间隔；
                                                     无效 agent 位置保持哨兵值 100

    唯一调用方是 score_prediction（本文件 :847），原来的 7 值返回里它只取第 0
    和第 6 个，其余 5 个（ttc_collision_all / key_agent_corners / key_agent_labels
    / collision_matrix[M,N,T] / key_agent_ids）当场丢弃。为了维护那 5 个死输出，
    旧实现对每个时间步都做一次 torch.nonzero + .any() + .tolist() + .item()，
    单次前向约 16 次 nonzero + 几十次 GPU->CPU 同步。现在直接返回调用方要的那两个。

    ⚠️ 返回值个数/含义在 2026-10-14 改过：曾经返回过 [M, N, T] 的 collision_matrix
    当第 0 个值，而调用方要的是 [M, T] 的 collision_all，导致 score_prediction:829
    形状不匹配。若再改这个签名，先看调用方解包。

    等价性：repo_snapshot/equiv_test2.py，243 组形状 x 有效性组合
    （M∈{1..12}、N∈{1..40}、T∈{1..16}、退化比例∈{0,0.3,0.5,0.9,1.0}），
    collision_all 与 min_distance_matrix 均 bit-identical；同文件还断言了
    "旧 ret[0] == 旧 ret[5].any(dim=1)"，即这里的归约确实是调用方要的那个量。
    """
    import torch

    assert fut_box_corners.ndim == 4 and _ego_coords.ndim == 4
    N, T, K, _ = fut_box_corners.shape
    M = _ego_coords.shape[0]
    assert _ego_coords.shape[1] == T and _ego_coords.shape[2] >= 4 and fut_box_corners.shape[2] >= 4

    device = fut_box_corners.device
    dtype = fut_box_corners.dtype

    # 仅取前4点作为矩形四点
    fut_rects = fut_box_corners[:, :, :4, :].contiguous()  # [N, T, 4, 2]
    ego_rects = _ego_coords[:, :, :4, :].contiguous()      # [M, T, 4, 2]

    # 有效 mask：该时间步是否存在一个"真实"的矩形框。
    #
    # 调用方 obtain_scored_waypoints 会把无效 agent（置信度 < 0.3，或类别属于
    # 交通灯/天空物体等）的四个角点整体置为同一个哨兵值，形成零面积退化矩形。
    # 这类矩形在下面的 SAT 归一化里会走到 0/0 产生 NaN，NaN 又经 min() 污染整行，
    # 于是 (dist < thr) 恒为 False，静默关闭下游的硬/软距离规则。所以这里按
    # "角点是否有跨度"判有效，而不是按"坐标是否非零"判有效。
    corner_extent = (
        fut_rects.max(dim=-2)[0] - fut_rects.min(dim=-2)[0]
    ).abs().sum(dim=-1)  # [N, T]
    fut_mask = corner_extent > 0  # [N, T] (bool)
    valid_mask = fut_mask.unsqueeze(0).expand(M, -1, -1)  # [M, N, T]

    # ---- 一次性批处理：把 (proposal, agent, timestep) 三元组全摊平成一次精确距离 ----
    # batch_exact_rect_distance 对每一行是独立的（每次 op 都只做 per-row 广播与
    # per-row reduction），所以把 [M, N, T] 摊平成 [M*N*T] 不会改变任何单个结果。
    # 退化矩形平移到 1e6 m：那个位置上 dist < 0 恒为 False、值也不会被读回，
    # 与旧代码用 nonzero 把它们压缩掉的效果完全一致（同时避开 0/0 的 NaN）。
    far = torch.tensor(1.0e6, dtype=dtype, device=device)
    safe_fut = torch.where(fut_mask[:, :, None, None], fut_rects, far).contiguous()  # [N, T, 4, 2]
    ego_flat = ego_rects.unsqueeze(1).expand(M, N, T, 4, 2).reshape(M * N * T, 4, 2)
    fut_flat = safe_fut.unsqueeze(0).expand(M, N, T, 4, 2).reshape(M * N * T, 4, 2)
    dist_all = batch_exact_rect_distance(ego_flat, fut_flat).view(M, N, T)

    # 活输出 #1：任一有效 agent 在该时刻与自车相交
    collision_all = ((dist_all < 0) & valid_mask).any(dim=1)  # [M, T]

    # 活输出 #2：[M, N, T] 精确间隔。
    # 下面这段前缀逐字保留自旧实现，只为复刻它的 dtype / device / 哨兵值行为；
    # 它算出来的每个格子要么随后被覆盖（valid），要么就等于字面量 100（退化），
    # 因此不可能影响最终结果。保留它是为了让"没改语义"这件事尽可能不依赖推理。
    distances = torch.norm(fut_rects.unsqueeze(0) - ego_rects.unsqueeze(1), dim=-1)
    min_distance_matrix = torch.min(distances, dim=-1)[0]
    min_distance_matrix = torch.where(valid_mask, min_distance_matrix, 100).to(
        min_distance_matrix.device
    )
    min_distance_matrix = torch.where(
        valid_mask,
        torch.clamp(dist_all, min=0.0).to(min_distance_matrix.dtype),
        min_distance_matrix,
    )

    return collision_all, min_distance_matrix


def evaluate_coll(fut_box_corners, _ego_coords):
    # fut_box_corners: [N, T, 2] - N哥他车未来轨迹的角点
    # _ego_coords: [M, T, 2] - M条自车轨迹在未来T个时刻角点

    
    num_agents = fut_box_corners.shape[0]
    n_future = _ego_coords.shape[1]
    _num_proposals = _ego_coords.shape[0]

    fut_mask = fut_box_corners.any(-1).any(-1)
    node_capacity = 10

    _ego_polygons = shapely.creation.polygons(_ego_coords)
    collision_matrix = np.zeros([_num_proposals, num_agents, n_future], dtype=bool)
    
    # 新增：存储最近距离的矩阵 [提案数, 他车数, 时间步]
    min_distance_matrix = np.full([_num_proposals, num_agents, n_future], np.inf)
    
    proposal_fault_collided_track_ids = {
        proposal_idx: []
        for proposal_idx in range(_num_proposals)
    }
    proposal_collided_track_ids = {
        proposal_idx: []
        for proposal_idx in range(_num_proposals)
    }
    temp_collided_track_ids = {
        proposal_idx: []
        for proposal_idx in range(_num_proposals)
    }
    ttc_collided_track_ids = {
        proposal_idx: []
        for proposal_idx in range(_num_proposals)
    }

    key_agent_corners = np.zeros([_num_proposals, n_future, 4, 2])
    key_agent_labels = np.zeros([_num_proposals, n_future], dtype=bool)
    key_agent_ids = []
    collision_all = np.zeros([_num_proposals, n_future], dtype=bool)
    ttc_collision_all = np.zeros([_num_proposals, n_future], dtype=bool)

    for time_idx in range(n_future):
        geometries = fut_box_corners[:, time_idx][fut_mask[:, time_idx]].numpy()
        _geometries = [Polygon(geometry) for geometry in geometries]
        _str_tree = STRtree(_geometries, node_capacity)
        ego_polygons = _ego_polygons[:, time_idx]
        
        # 计算所有他车与自车的距离
        for agent_idx in range(num_agents):
            if not fut_mask[agent_idx, time_idx]:
                continue
                
            agent_poly = Polygon(fut_box_corners[agent_idx, time_idx].numpy())
            
            for proposal_idx in range(_num_proposals):
                ego_poly = ego_polygons[proposal_idx]
                # 计算两个多边形之间的最小距离, 这种计算包括边界点，比仅使用顶点会更小
                distance = agent_poly.distance(ego_poly)
                min_distance_matrix[proposal_idx, agent_idx, time_idx] = distance
        
        # 原有的碰撞检测逻辑
        intersecting = _str_tree.query(ego_polygons, predicate="intersects")

        if len(fut_box_corners) == 0 or len(intersecting[0]) == 0:
            continue
            
        token_list = np.arange(len(fut_box_corners))[fut_mask[:, time_idx].cpu().numpy()]

        for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
            token = token_list[geometry_idx]
            if token in proposal_collided_track_ids[proposal_idx] or len(proposal_fault_collided_track_ids[proposal_idx]):
                continue

            proposal_fault_collided_track_ids[proposal_idx].append(token)
            collision_all[proposal_idx][time_idx] = True
            key_agent_labels[proposal_idx][:time_idx+1] = fut_mask[token][:time_idx+1]
            key_agent_corners[proposal_idx][:time_idx+1] = fut_box_corners[token][:time_idx+1]
            key_agent_ids.append(token)
            collision_matrix[proposal_idx, token, time_idx] = True

    return (
        collision_all, 
        ttc_collision_all, 
        key_agent_corners[:-1], 
        key_agent_labels[:-1], 
        key_agent_ids, 
        collision_matrix,
        min_distance_matrix  # 新增：返回最近距离矩阵
    )


def generate_corners_torch(trajs, data=None):
    # trasj: B, 6, 2


    # 获得自车以及其他车辆的角点坐标
    device = trajs.device
    # 由于一致，所以自车corner可以在batch间并行计算
    init_state = torch.zeros((trajs.shape[0], 3), dtype=torch.float32, device=device) # 初始化状态
    init_state[:,-1] = 3.14/2   # 自车默认90度朝前
    ego_corners = compute_corners_torch(trajs, init_state=init_state) # 并行维度0

    agent_corners = None
    if data is not None:
        fut_bbox = data['gt_agent_fut_trajs']
        bboxes = data['gt_bboxes_3d']
        bboxes_name = data['gt_labels_3d']
        bs = len(bboxes)    # 因为bboxes在每个batch中数量不固定，所以进来是以List的形式
        agent_corners = []

        for i in range(bs): # 逐batch的解
            bboxes_names_batch = bboxes_name[i]
            bboxes_batch = bboxes[i]
            fut_bbox_batch = fut_bbox[i]

            # 交通灯在天上，就不考虑撞击问题
            traffic_light_mask = bboxes_names_batch != 6  #     # 6是交通灯的label

            # 过滤掉交通灯
            bboxes_batch = bboxes_batch[traffic_light_mask]
            fut_bbox_batch = fut_bbox_batch[traffic_light_mask]  

            # 创建Agents的初始状态
            half_fut_bbox_length = bboxes_batch[:, 3] / 2
            half_fut_bbox_width = bboxes_batch[:, 4] / 2
            bbox_init_center = bboxes_batch[:, :2]
            bbox_init_yaw = bboxes_batch[:, 6:7]
            bbox_init_state = torch.cat((bbox_init_center, bbox_init_yaw), dim=-1)

            # 计算Agents的角点
            agent_corner = compute_corners_torch(fut_bbox_batch.clone(), half_fut_bbox_length.clone(), half_fut_bbox_width.clone(), bbox_init_state.clone()) # 并行维度0
            agent_corners.append(agent_corner)  # shape: [T, 4, 2]
        
    return ego_corners, agent_corners


def get_collision_matrix(trajs, data=None, agent_corners=None):
    device = trajs.device
    fut_bbox = data['gt_agent_fut_trajs']
    bboxes = data['gt_bboxes_3d']
    bboxes_name = data['gt_labels_3d']
    bs = len(bboxes)    # 因为bboxes在每个batch中数量不固定，所以进来是以List的形式

    # 由于一致，所以自车corner可以在batch间并行计算
    num_modal = trajs.shape[1]
    trajs = rearrange(trajs, 'b m (t c) -> (b m) t c', t=6,c=2) # trajs的shape是 [bs, T, 12]，需要变成 [bs, T, 1, 2]，以便和ego_corners对齐
    init_state = torch.zeros((trajs.shape[0], 3), dtype=torch.float32, device=device) # 初始化状态
    init_state[:,-1] = 3.14/2   # 自车默认90度朝前
    ego_corners = compute_corners_torch(trajs, init_state=init_state) # 并行维度0
    ego_corners = rearrange(ego_corners, '(b m) t c d -> b m t c d', b=bs, m=num_modal)  # 
    collision_all_bs = []
    collision_list = []
    min_distance_matrixs = []
    for i in range(bs): # 逐batch的解
        bboxes_names_batch = bboxes_name[i]
        bboxes_batch = bboxes[i]
        fut_bbox_batch = fut_bbox[i]

        # 交通灯在天上，就不考虑撞击问题
        traffic_light_mask = bboxes_names_batch != 6  #     # 6是交通灯的label

        # 过滤掉交通灯
        bboxes_batch = bboxes_batch[traffic_light_mask]
        fut_bbox_batch = fut_bbox_batch[traffic_light_mask]  

        # 创建Agents的初始状态
        half_fut_bbox_length = bboxes_batch[:, 3] / 2
        half_fut_bbox_width = bboxes_batch[:, 4] / 2
        bbox_init_center = bboxes_batch[:, :2]
        bbox_init_yaw = bboxes_batch[:, 6:7]
        bbox_init_state = torch.cat((bbox_init_center, bbox_init_yaw), dim=-1)

        # 计算Agents的角点
        agent_corner = compute_corners_torch(fut_bbox_batch.clone(), half_fut_bbox_length.clone(), half_fut_bbox_width.clone(), bbox_init_state.clone()) # 并行维度0

        # 计算碰撞
        collision_matrix, ttc_collision_all, key_agent_corners, key_agent_labels, key_agent_ids, collision_agent, min_distance_matrix = evaluate_coll(agent_corner.cpu(), ego_corners[i].cpu())

        collision_all_bs.append(collision_matrix)
        collision_list.append(collision_agent)
        min_distance_matrix.append(min_distance_matrix)

    collision_all_bs = np.stack(collision_all_bs, axis=0)  # shape: (bs, num_modal, n_future) 在每个时间步是否发生碰撞
    min_distance_matrix = np.stack(min_distance_matrix, axis=0)  # shape: (bs, num_modal, n_future) 在每个时间步的最小距离
    # collision_all_bs中是每个proposal最终
    return torch.tensor(collision_all_bs, dtype=torch.bool, device=device), collision_list, min_distance_matrix  # shape: (bs, num_modal, n_future)

def get_score(trajs, data):
    # trajs [bs, T, 12]: batch size, modal, 12
    # data 是一个字典，包含了gt_agent_fut_trajs, gt_bboxes_3d, gt_labels_3d等信息
    # collision_matrix中是每个Proposal在每个时间步是否发生碰撞
    # collision_list中是每个Proposal在每个时间步与哪个agents发生碰撞，因为agents数量不固定，所以不能堆叠

    set_scores = torch.zeros((trajs.shape[0], trajs.shape[1]), dtype=torch.float32, device=trajs.device)  # [bs, num_modal]

    # 获取碰撞信息
    collision_sores = torch.ones((trajs.shape[0], trajs.shape[1]), dtype=torch.float32, device=trajs.device)  # [bs, num_modal]
    collision_matrix, collision_list = get_collision_matrix(trajs, data)
    # 若发生碰撞，则将对应的分数设为0
    collision_sores[collision_matrix.any(dim=-1)] = 0.0
    # 从collision_matrix中获得每个proposal第一次发生碰撞的时间步
    collision_detected = collision_matrix.any(dim=-1)  # [bs, num_modal]
    first_collision_time = torch.where(
        collision_detected,
        collision_matrix.to(torch.long).argmax(dim=-1),
        collision_matrix.shape[-1]  # 设为最大时间步表示未碰撞
    )


    ttc_scores = torch.ones((trajs.shape[0], trajs.shape[1]), dtype=torch.float32, device=trajs.device)  # [bs, num_modal]
    # fisrt_collision_time小于3的ttc_scores设为0
    ttc_scores[first_collision_time < 3] = 0.0

    # 累加最终得分
    set_scores = collision_sores + ttc_scores  # [bs, num_modal]


    return set_scores, collision_matrix, collision_list


def batch_min_vertex_distance(rect1, rect2, beta=10.0):
    """
    Computes the differentiable minimum vertex distance between two sets of rectangles.
    
    Args:
        rect1: [B, 4, 2] - First set of rectangles (4 vertices each).
        rect2: [B, 4, 2] - Second set of rectangles (4 vertices each).
        beta: Temperature parameter for softmin (higher = sharper approximation of min).
    
    Returns:
        [B] - Soft minimum vertex distance for each pair of rectangles.
    """
    B = rect1.shape[0]
    
    # Compute pairwise distances: [B, 4, 4]
    diff = rect1.unsqueeze(2) - rect2.unsqueeze(1)  # [B, 4, 4, 2]
    distances = torch.norm(diff, dim=3)  # [B, 4, 4]
    
    # Flatten to [B, 16] and compute softmin
    distances_flat = distances.view(B, -1)  # [B, 16]
    
    # Option 1: LogSumExp-based softmin (more numerically stable)
    min_distances = -torch.logsumexp(-beta * distances_flat, dim=1) / beta
    
    # Option 2: Direct softmin (less stable for large beta)
    # min_distances = (distances_flat * torch.softmin(distances_flat, dim=1)).sum(dim=1)
    
    return min_distances


# copy from DiffusionPlanner 

# ego_size = [get_pacifica_parameters().length, get_pacifica_parameters().width]

def batch_signed_distance_rect(rect1, rect2):
    '''
    rect1: [B, 4, 2]
    rect2: [B, 4, 2]
    
    return [B] (signed distance between two rectangles)
    '''
    B, _, _ = rect1.shape
    norm_vec = torch.stack([rect1[:, 0] - rect1[:, 1], 
                             rect1[:, 1] - rect1[:, 2], 
                             rect2[:, 0] - rect2[:, 1], 
                             rect2[:, 1] - rect2[:, 2]], dim=1) # [B, 4, 2]
    norm_vec = norm_vec / torch.norm(norm_vec, dim=2, keepdim=True)
    
    proj1 = torch.einsum('bij,bkj->bik', norm_vec, rect1) # [B, 4, 2] * [B, 4, 2] -> [B, 4, 4]
    proj1_min, proj1_max = proj1.min(dim=2)[0], proj1.max(dim=2)[0] # [B, 4] [B, 4]
    
    proj2 = torch.einsum('bij,bkj->bik', norm_vec, rect2) # [B, 4, 2] * [B, 4, 2] -> [B, 4, 4]
    proj2_min, proj2_max = proj2.min(dim=2)[0], proj2.max(dim=2)[0] # [B, 4] [B, 4]
    
    overlap = torch.cat([proj1_min - proj2_max, proj2_min - proj1_max], dim=1) # [B, 8]
    
    positive_distance = torch.where(overlap < 0, 1e5, overlap)
    
    is_overlap = (overlap < 0).all(dim=1)

    distance = torch.where(is_overlap, overlap.max(dim=1).values, positive_distance.min(dim=1).values)   
    
    return distance

def batch_exact_rect_distance(rect1, rect2):
    """
    Signed distance between two sets of rectangles.

    rect1/rect2: [B, 4, 2], 4 corners in perimeter order (the convention used by
    compute_corners_torch: local x = length axis, local y = width axis).

    Returns [B]:
      < 0  rectangles intersect -> SAT penetration depth (only the sign is used
            by the existing `dist < 0` collision test, so it keeps working),
      >= 0 the EXACT minimum distance between the two rectangles.

    Why not `batch_signed_distance_rect` (the DiffusionPlanner SAT)?
    That function returns the smallest of the four *projection gaps*, which is a
    lower bound of - and usually much smaller than - the true rectangle
    distance.  Measured on 50k random realistic box pairs it never over-reads
    the clearance, under-reads it in 54% of cases (mean -0.66 m, p05 -3.16 m),
    and therefore flags ~22.6% of *geometrically safe*
    (proposal, agent, timestep) triples as "too close" at the 0.3 m hard
    threshold that was tuned against the shapely reference implementation.
    (Taking the *max* over the SAT axes instead is not equivalent either: for
    two rectangles offset diagonally the true distance is sqrt(2) while every
    SAT axis gap is 1.0.)

    For disjoint convex polygons the minimum distance is attained at a
    vertex/edge pair, so the minimum over the 32 vertex-to-edge distances below
    is exact.  The 32 pairs are evaluated in two broadcast batches
    ([B,4,1,2] x [B,1,4,2] -> [B,4,4]) so the whole thing costs ~6 kernels
    instead of ~200 - it keeps the GPU fast path cheap (no .cpu(), no loop).
    """
    # --- SAT axes: the two edge directions of each rectangle -----------------
    norm_vec = torch.stack([rect1[:, 0] - rect1[:, 1],
                            rect1[:, 1] - rect1[:, 2],
                            rect2[:, 0] - rect2[:, 1],
                            rect2[:, 1] - rect2[:, 2]], dim=1)  # [B, 4, 2]
    norm_vec = norm_vec / torch.norm(norm_vec, dim=2, keepdim=True)

    proj1 = torch.einsum('bij,bkj->bik', norm_vec, rect1)
    proj2 = torch.einsum('bij,bkj->bik', norm_vec, rect2)
    overlap = torch.cat([proj1.min(dim=2)[0] - proj2.max(dim=2)[0],
                         proj2.min(dim=2)[0] - proj1.max(dim=2)[0]], dim=1)  # [B, 8]
    is_overlap = (overlap < 0).all(dim=1)          # rectangles intersect
    penetration = overlap.max(dim=1).values       # < 0

    # --- exact minimum distance (vertex/edge pairs), fully broadcast ---------
    nxt = [1, 2, 3, 0]
    e1a, e1b = rect1, rect1[:, nxt, :]            # [B,4,2] edges of rect1
    e2a, e2b = rect2, rect2[:, nxt, :]            # [B,4,2] edges of rect2

    # rect1 vertices -> rect2 edges : [B,4,4]
    d12 = point_segment_distance(rect1[:, :, None, :],
                                 e2a[:, None, :, :],
                                 e2b[:, None, :, :])
    # rect2 vertices -> rect1 edges : [B,4,4]
    d21 = point_segment_distance(rect2[:, :, None, :],
                                 e1a[:, None, :, :],
                                 e1b[:, None, :, :])
    separated = torch.cat([d12.reshape(d12.shape[0], -1),
                           d21.reshape(d21.shape[0], -1)], dim=1).min(dim=1).values  # [B]

    return torch.where(is_overlap, penetration, separated)


def point_segment_distance(p, a, b):
    """Distance from points p to segments [a, b]; broadcastable to [..., 2].

    Works both for a single [B, 2] x [B, 2] x [B, 2] batch and for the
    [B, 4, 1, 2] x [B, 1, 4, 2] x [B, 1, 4, 2] form used by
    batch_exact_rect_distance, where it evaluates 16 vertex/edge pairs at once.
    """
    ab = b - a
    ap = p - a
    denom = (ab * ab).sum(dim=-1)
    t = torch.where(
        denom > 1e-12,
        (ap * ab).sum(dim=-1) / torch.clamp(denom, min=1e-12),
        torch.zeros_like(denom),
    ).clamp(0.0, 1.0)
    proj = a + t.unsqueeze(-1) * ab
    return torch.norm(p - proj, dim=-1)


def refine_trajs(trajs, current_speed, max_acc=3.75, max_brake=9.0):
    """
    限制轨迹的速度变化，确保不超过车辆的加减速能力

    trajs: [bs, N, T, 2]，偏移形式，每个模态预测的轨迹
    current_speed: [bs, 1]，当前速度（单位：m/s）
    max_acc: 最大加速度（m/s^2）
    max_brake: 最大减速度（m/s^2）
    
    返回：
    refined_trajs: [bs, N, T, 2]，限制速度变化后的轨迹
    """
    dt = 0.2  # 5Hz 频率对应时间间隔

    bs, num_modes, T, _ = trajs.shape

    # 计算每一帧的位移量 norm（速度 * dt）
    displacements = torch.norm(trajs, dim=-1)  # [bs, N, T]
    ideal_speeds = displacements / dt  # [bs, N, T]

    # 初始化真实速度：第一帧为 current_speed，后续逐帧限制加速度变化
    refined_speeds = torch.zeros_like(ideal_speeds)
    refined_speeds[:, :, 0] = current_speed  # 第一帧直接使用当前速度

    for t in range(1, T):
        delta_v = ideal_speeds[:, :, t] - refined_speeds[:, :, t - 1]

        # 限制加速度和减速度
        max_dv = max_acc * dt
        min_dv = -max_brake * dt
        delta_v_clipped = delta_v.clamp(min=min_dv, max=max_dv)

        refined_speeds[:, :, t] = refined_speeds[:, :, t - 1] + delta_v_clipped

    # 重新计算每帧的比例因子 = 限制后的速度 / 原速度，避免除以0
    scale = torch.ones_like(ideal_speeds)
    mask = ideal_speeds > 1e-4
    scale[mask] = refined_speeds[mask] / ideal_speeds[mask]

    # 将每帧的偏移量按比例缩放
    refined_trajs = trajs * scale.unsqueeze(-1)

    return refined_trajs


def socre_collision(collision_matrix, half_time, num_t, device):
    # Find first collision time for each proposal
    first_collision_time = torch.where(
        collision_matrix.any(dim=-1),
        collision_matrix.to(torch.long).argmax(dim=-1),
        num_t  # Set to max time if no collision
    )

    # Calculate time-based scores
    
    # Strict penalty for early collisions (first half)
    early_collision_penalty = torch.where(
        first_collision_time < half_time,
        torch.tensor(0.0, device=device),  # Minimum score for early collisions
        torch.tensor(1.0, device=device)   # Full score if no early collision
    )
    
    # Gradual penalty for late collisions (second half)
    late_collision_penalty = torch.where(
        first_collision_time < num_t,
        (first_collision_time - (half_time-0.1)) / half_time,  # Linear increase from 0 to 1
        torch.tensor(1.0, device=device)  # Full score if no collision at all
    )
    
    # Combine time-based penalties
    time_scores = early_collision_penalty * late_collision_penalty

    set_scores = time_scores  # [bs, num_modal]
    
    #只保留两位小数
    set_scores = torch.round(set_scores * 100) / 100

    return set_scores 


def score_prediction(trajs, agents_corners, current_speed=None, light_state=0):
    # trajs [bs, N, T, 2], Offset形式
    # collision_matrix: whether each proposal collides at each timestep
    # collision_list: which agents each proposal collides with at each timestep
    num_t = trajs.shape[2]
    num_modal = trajs.shape[1]
    bs = trajs.shape[0]
    device = trajs.device

    current_position = torch.zeros_like(trajs[:, :, 0:1, :])  # [bs, N, 1, 2]
    trajs = torch.cat([current_position, trajs], dim=2)  # [bs, N+1, T, 2]
    if current_speed is not None:
        refined_trajs = refine_trajs(trajs, current_speed)
    else:
        refined_trajs = trajs


    # Prepare tensors for scores
    raw_scores = torch.zeros((bs, num_modal), dtype=torch.float32, device=device)
    
    # Reshape trajectories and compute ego vehicle corners
    refined_trajs = rearrange(refined_trajs, 'b m t c -> (b m) t c', t=num_t+1, c=2)
    init_state = torch.zeros((refined_trajs.shape[0], 3), dtype=torch.float32, device=device)
    init_state[:, -1] = 3.14/2  # Default 90 degree heading
    ego_corners = compute_corners_torch(refined_trajs, init_state=init_state)
    ego_corners = rearrange(ego_corners, '(b m) t c d -> b m t c d', b=bs, m=num_modal)


    # Compute collision and distance information
    collision_all_bs = []
    min_distance_matrixs = []

    # 碰撞判定的实现选择，详见 docs/INFERENCE.md：
    #   fast    —— GPU 上的矩形 SAT，论文报告的推理吞吐即由此测得，为默认值；
    #   shapely —— CPU Shapely/STRtree，参考实现，便于与历史指标对齐。
    collision_mode = os.environ.get("ALIGNDRIVE_COLLISION_MODE", "fast").lower()
    if collision_mode not in ("shapely", "fast"):
        raise ValueError(
            "ALIGNDRIVE_COLLISION_MODE must be 'shapely' or 'fast', "
            f"got {collision_mode!r}"
        )

    for i in range(bs):
        if collision_mode == "fast":
            collision_matrix, min_distance_matrix = evaluate_coll_fast(
                agents_corners[i], ego_corners[i]
            )
        else:
            collision_matrix, _, _, _, _, _, min_distance_matrix = evaluate_coll(
                agents_corners[i].cpu(), ego_corners[i].cpu()
            )

        collision_all_bs.append(collision_matrix)
        min_distance_matrixs.append(min_distance_matrix)

    if not isinstance(collision_all_bs[0], torch.Tensor):
        collision_matrix = torch.tensor(
            np.stack(collision_all_bs, axis=0), 
            dtype=torch.bool, 
            device=device
        )  # [bs, num_modal, num_t]
    
        min_distance_matrix = torch.tensor(
            np.stack(min_distance_matrixs, axis=0),
            dtype=torch.float32,
            device=device
        )  # [bs, num_modal, num_t]

    else:
        collision_matrix = torch.stack(collision_all_bs, axis=0).to(torch.bool).to(device)
        min_distance_matrix = torch.stack(min_distance_matrixs, axis=0).to(torch.float32).to(device)


    # 检查每个Batch内是否有当前帧就碰撞的
    current_collision = torch.any(collision_matrix[:, :, 0], dim=1)  # [num_modal]

    collision_matrix = collision_matrix[:, :, 1:]  # 移除第一帧的碰撞信息
    min_distance_matrix = min_distance_matrix[:, :, :, 1:]  # 移除第一帧的最小距离信息

    # Calculate distance-based scores when no collision occurs
    min_distance_thrhold = 0.3              # 用Fast的时候这个参数需要修改
    min_distance_thrhold_soft = 0.05
    min_distances = min_distance_matrix.min(dim=-2)[0]  # [bs, num_modal, T]
    half_time = num_t // 2

    # 只在 t < num_t 时考虑最近距离约束
    time_mask = torch.arange(min_distances.size(-1), device=device) < num_t

    # 生成距离碰撞掩码（True 表示碰撞）
    distance_collision_hard_mask = (min_distances < min_distance_thrhold) & time_mask.unsqueeze(0).unsqueeze(0)
    distance_collision_sorf_mask = (min_distances < (min_distance_thrhold_soft)) & time_mask.unsqueeze(0).unsqueeze(0)
    # 合并到 collision_matrix
    collision_matrix_soft = collision_matrix.clone() | distance_collision_sorf_mask
    collision_matrix_hard = collision_matrix | distance_collision_hard_mask

    set_scores_soft = socre_collision(collision_matrix_soft, half_time, num_t, device)
 
    set_scores_hard = socre_collision(collision_matrix_hard, half_time, num_t, device)

    # set_scores, collision_matrix, collision_all_bs = socre_collision(collision_matrix_hard, half_time)

    return set_scores_hard, set_scores_soft, current_collision
