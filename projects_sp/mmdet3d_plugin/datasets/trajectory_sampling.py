"""Trajectory sampling shared by training datasets and data preprocessing."""

import numpy as np


class Bench2DriveTrajectoryMixin:
    """Sample masked ego and agent trajectories from ordered Bench2Drive metadata."""

    def get_fut_agent(self, idx, sample_rate, frames):
        """Return agent displacement sequences and masks in the current LiDAR frame."""
        adj_idx_list = range(idx, idx + (frames + 1) * sample_rate, sample_rate)
        cur_frame = self.data_infos[idx]

        box_ids = cur_frame['gt_ids']

        future_track = np.zeros((len(box_ids), frames + 1, 2))
        future_mask = np.zeros((len(box_ids), frames + 1))
        world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for i in range(len(box_ids)):
            box_id = box_ids[i]

            for j in range(len(adj_idx_list)):
                adj_idx = adj_idx_list[j]
                if adj_idx < 0 or adj_idx >= len(self.data_infos):
                    break
                adj_frame = self.data_infos[adj_idx]
                if adj_frame['folder'] != cur_frame['folder']:
                    break
                if len(np.where(adj_frame['gt_ids'] == box_id)[0]) == 0:
                    break
                assert len(np.where(adj_frame['gt_ids'] == box_id)[0]) == 1, np.where(
                    adj_frame['gt_ids'] == box_id
                )[0]
                adj_idx = np.where(adj_frame['gt_ids'] == box_id)[0][0]
                adj_box2lidar = world2lidar_lidar_cur @ adj_frame['npc2world'][adj_idx]
                adj_xy = adj_box2lidar[0:2, 3]
                if j > 0:
                    last_xy = future_track[i, j - 1, :]
                    distance = np.linalg.norm(last_xy - adj_xy)
                    if distance > 10:
                        break
                future_track[i, j, :] = adj_xy
                future_mask[i, j] = 1

        future_track_offset = future_track[:, 1:, :] - future_track[:, :-1, :]
        future_mask_offset = future_mask[:, 1:]
        future_track_offset[future_mask_offset == 0] = 0

        return future_track_offset.astype(np.float32), future_mask_offset.astype(np.float32)

    def get_fut_agent_dp(self, idx, end_idx, interval_distance=5, num_points=6):

        """Sample agent drive paths by distance and return offsets with validity masks."""
        info = self.data_infos[idx]
        mask = info['num_points'] != 0

        future_frames = end_idx - idx
        gt_agent_fut_trajs, gt_agent_fut_masks = self.get_fut_agent(idx, 1, future_frames)
        gt_agent_fut_trajs = np.cumsum(gt_agent_fut_trajs, axis=1)
        interval_func = self._parse_interval_distance(interval_distance, num_points)

        num_agents = gt_agent_fut_trajs.shape[0]
        agents_dp = []
        agents_mask = []
        for i in range(num_agents):
            future_track = np.zeros((num_points, 2))
            agents_positions = gt_agent_fut_trajs[i]
            sampled_points = self._linear_interpolate_dynamic_distance(
                agents_positions, interval_func, num_points
            )
            valid_mask = np.zeros(num_points, dtype=bool)

            num_valid_points = min(len(sampled_points), num_points)
            if num_valid_points == 0:
                agents_dp.append(np.zeros((num_points, 2)))
                agents_mask.append(valid_mask)
                continue

            future_track[:num_valid_points] = sampled_points[:num_valid_points]
            valid_mask[:num_valid_points] = True

            current_posi = np.zeros(shape=(1, 2))
            future_track = np.concatenate((current_posi, future_track), axis=0)

            offset_track = future_track[1:] - future_track[:-1]

            agents_dp.append(offset_track)
            agents_mask.append(valid_mask)

        agents_dp = np.array(agents_dp)[mask]
        agents_mask = np.array(agents_mask)[mask]

        return agents_dp.copy(), agents_mask.copy()

    def get_ego_trajs(self, idx, sample_rate, past_frames, future_frames):
        """Return past/future ego offsets, future validity, and the navigation command."""
        adj_idx_list = range(
            idx - past_frames * sample_rate, idx + (future_frames + 1) * sample_rate, sample_rate
        )
        cur_frame = self.data_infos[idx]
        full_adj_track = np.zeros((past_frames + future_frames + 1, 2))
        full_adj_adj_mask = np.zeros(past_frames + future_frames + 1)
        world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for j in range(len(adj_idx_list)):
            adj_idx = adj_idx_list[j]
            if adj_idx < 0 or adj_idx >= len(self.data_infos):
                break

            adj_frame = self.data_infos[adj_idx]
            if adj_frame['folder'] != cur_frame['folder']:
                break

            world2lidar_ego_adj = adj_frame['sensors']['LIDAR_TOP']['world2lidar']
            adj2cur_lidar = world2lidar_lidar_cur @ np.linalg.inv(world2lidar_ego_adj)
            xy = adj2cur_lidar[0:2, 3]
            full_adj_track[j, 0:2] = xy
            full_adj_adj_mask[j] = 1
        offset_track = full_adj_track[1:] - full_adj_track[:-1]
        for j in range(past_frames - 1, -1, -1):
            if full_adj_adj_mask[j] == 0:
                offset_track[j] = offset_track[j + 1]
        for j in range(past_frames, past_frames + future_frames, 1):

            if full_adj_adj_mask[j + 1] == 0:
                offset_track[j] = 0
        if self.use_cmd:
            command = self.command2hot(cur_frame['command_near'])
        else:
            command = np.array(
                [
                    0,
                ]
            )
        offset_track = offset_track.astype(np.float32)
        return (
            offset_track[:past_frames].copy(),
            offset_track[past_frames:].copy(),
            full_adj_adj_mask[-future_frames:].copy(),
            command,
        )

    def get_future_trajs_by_distance(self, idx, end_idx, interval_distance, num_points=6):
        """Sample future ego positions using scalar, per-point, or callable distances."""

        future_track = np.zeros((num_points, 2))
        valid_mask = np.zeros(num_points, dtype=bool)

        cur_frame = self.data_infos[idx]
        world2lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']

        positions = self._collect_future_positions(
            idx, end_idx, world2lidar_cur, cur_frame['folder']
        )

        if len(positions) < 2:
            return np.zeros((num_points, 2)), valid_mask, self._get_command(cur_frame)

        interval_func = self._parse_interval_distance(interval_distance, num_points)

        sampled_points = self._linear_interpolate_dynamic_distance(
            positions, interval_func, num_points
        )

        num_valid_points = min(len(sampled_points), num_points)
        if num_valid_points == 0:
            return np.zeros((num_points, 2)), valid_mask, self._get_command(cur_frame)
        future_track[:num_valid_points] = sampled_points[:num_valid_points]
        valid_mask[:num_valid_points] = True

        current_posi = np.zeros(shape=(1, 2))
        future_track = np.concatenate((current_posi, future_track), axis=0)

        offset_track = future_track[1:] - future_track[:-1]

        return offset_track.copy(), valid_mask.copy(), self._get_command(cur_frame)

    def _parse_interval_distance(self, interval_distance, num_points):
        """Normalize the spacing specification to a callable indexed by sample number."""
        if callable(interval_distance):

            return interval_distance
        elif isinstance(interval_distance, (list, np.ndarray)):

            if len(interval_distance) != num_points:
                raise ValueError(
                    f"Expected {num_points} interval distances, got {len(interval_distance)}"
                )
            return lambda i: interval_distance[i]
        else:

            return lambda i: interval_distance

    def _linear_interpolate_dynamic_distance(self, positions, interval_func, max_points):
        """Interpolate an (N, 2) polyline using the requested consecutive distances."""
        if len(positions) < 2:
            return []

        distances = np.zeros(len(positions))
        for i in range(1, len(positions)):
            distances[i] = distances[i - 1] + np.linalg.norm(positions[i] - positions[i - 1])

        total_distance = distances[-1]
        sampled_points = []
        current_distance = 0

        for point_idx in range(max_points):

            interval_dist = interval_func(point_idx)

            target_distance = current_distance + interval_dist

            if target_distance > total_distance:
                break
            elif target_distance <= 1e-6 and total_distance < 1e-6:
                break

            idx_high = np.searchsorted(distances, target_distance)
            idx_low = idx_high - 1

            if idx_high >= len(distances):
                break

            ratio = (target_distance - distances[idx_low]) / (
                distances[idx_high] - distances[idx_low]
            )

            interpolated_point = positions[idx_low] + ratio * (
                positions[idx_high] - positions[idx_low]
            )
            sampled_points.append(interpolated_point)

            current_distance = target_distance

        return np.array(sampled_points)

    def _collect_future_positions(self, start_idx, end_idx, world2lidar_cur, current_folder):
        """Collect future LiDAR origins in the current frame, stopping at route boundaries."""
        positions = []
        current_idx = start_idx + 1

        while current_idx < end_idx:
            frame = self.data_infos[current_idx]

            if frame['folder'] != current_folder:
                break

            world2lidar = frame['sensors']['LIDAR_TOP']['world2lidar']
            adj2cur = world2lidar_cur @ np.linalg.inv(world2lidar)
            xy = adj2cur[0:2, 3]
            positions.append(xy)
            current_idx += 1

        return np.array(positions)

    def _get_command(self, frame):
        """Return the configured navigation command representation."""
        if self.use_cmd:
            return self.command2hot(frame['command_near'])
        return np.array(
            [
                0,
            ]
        )

    def get_future_trajs_by_time_interval(self, idx, end_idx, time_interval, num_points=6):
        """Return ego offsets sampled every time_interval dataset frames.

        end_idx is retained for compatibility with the shared converter interface.
        Route boundaries and unavailable future frames are masked by get_ego_trajs.
        """
        _, future_offsets, valid_mask, command = self.get_ego_trajs(
            idx, time_interval, 0, num_points
        )
        return future_offsets, valid_mask, command

    def command2hot(self, command, max_dim=6):
        if command < 0:
            command = 4
        command -= 1
        cmd_one_hot = np.zeros(max_dim)
        cmd_one_hot[command] = 1
        return cmd_one_hot

    def invert_pose(self, pose):
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = np.transpose(pose[:3, :3])
        inv_pose[:3, -1] = -inv_pose[:3, :3] @ pose[:3, -1]
        return inv_pose
