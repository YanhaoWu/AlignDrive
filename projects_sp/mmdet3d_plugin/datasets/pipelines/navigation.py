import torch

import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from PIL import Image





@PIPELINES.register_module()
class Process_EgoStatus(object):

    def __init__(self, noise_pro, num_bins=1024, v_max=20,  used_attri=['speed_0']):
        self.noise_pro = noise_pro
        self.num_bins = num_bins
        self.v_max = v_max
        self.used_attri = used_attri
        self.all_attri = ['accel_0', 'accel_1', 'accel_2', 'rotation_0', 'rotation_1', 'rotation_2', 'speed_0', 'speed_1', 'speed_2', 'direction']

    def __call__(self, input_dict):
        noise_ego = random.normal(0, self.noise_pro, size=(1, 1))
        # 获得需要的属性
        ego_status = input_dict["ego_status"].copy()

        key_staus = []
        for attri in self.used_attri:
            index = self.all_attri.index(attri)
            key_staus.append(ego_status[index] + noise_ego[0])  # 添加噪声
            ego_status[index] = ego_status[index] + noise_ego[0]  # 更新原始ego_status
        input_dict["ego_speed"] = torch.tensor(ego_status[self.all_attri.index('speed_0')]).reshape(1, 1 )
        # 归一化
        key_staus = np.array(key_staus)
        key_staus = key_staus / np.array([self.v_max])
        # 量化
        keystatus = np.clip(key_staus, 0, 1)
        # 离散化
        bins = np.linspace(0, 1, self.num_bins + 1)
        digitized = np.digitize(keystatus, bins) - 1
        # 更新
        input_dict["ego_status_digit"] = torch.tensor(digitized)

        return input_dict





@PIPELINES.register_module()
class Build_Navaigation(object):

    def __init__(self, gloalpoint_extend, noise_pro=0.0, noise_scale=0.0):
        self.gloalpoint_extend = gloalpoint_extend
        self.noise_pro = noise_pro
        self.noise_scale = noise_scale

    def _add_noise(self, points):
        """
        给导航点添加随机噪声
        
        参数:
            points: 导航点坐标 (N, 2)
            
        返回:
            添加噪声后的导航点
        """
        if self.noise_pro <= 0:
            return points
            
        # 随机选择要添加噪声的点
        mask = np.random.random(len(points)) < self.noise_pro
        if not np.any(mask):
            return points
            
        # 生成高斯噪声
        noise = np.random.randn(*points[mask].shape) * self.noise_scale
        points[mask] += noise
        
        return points

    def __call__(self, input_dict):
        goal_points = []
        tp_near = input_dict["tp_near"].reshape(-1, 2)
        tp_far = input_dict["tp_far"].reshape(-1, 2)

        if self.noise_pro > 0:
            tp_near = self._add_noise(tp_near)
            tp_far = self._add_noise(tp_far)
        

        goal_points.append(tp_near)
        goal_points.append(tp_far)

        if self.gloalpoint_extend:
            # 确保我们有两个目标点
            assert len(goal_points) == 2, "Need exactly two points for extension"
            
            # 提取两个目标点
            p_start = goal_points[0][0]  # 第一个目标点
            p_end = goal_points[1][0]    # 第二个目标点
            
            # 计算5个插值点（总共6个点）
            t = np.linspace(0, 1, 6)  # 均匀采样6个点
            interpolated_points = []
            for ti in t:
                # 线性插值
                x = p_start[0] + ti * (p_end[0] - p_start[0])
                y = p_start[1] + ti * (p_end[1] - p_start[1])
                interpolated_points.append([x, y])
            
            meraged_line = np.array(interpolated_points).reshape(-1, 2)

        else:
            meraged_line = goal_points[-1]
        
        input_dict['tp_line'] = meraged_line
        return input_dict