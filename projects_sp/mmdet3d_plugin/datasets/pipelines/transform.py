import numpy as np
import mmcv
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

@PIPELINES.register_module()
class MultiScaleDepthMapGenerator(object):
    def __init__(self, downsample=1, max_depth=60):
        if not isinstance(downsample, (list, tuple)):
            downsample = [downsample]
        self.downsample = downsample
        self.max_depth = max_depth

    def __call__(self, input_dict):
        points = input_dict["points"][..., :3, None]
        gt_depth = []
        for i, lidar2img in enumerate(input_dict["lidar2img"]):
            H, W = input_dict["img_shape"][i][:2]

            pts_2d = (
                np.squeeze(lidar2img[:3, :3] @ points, axis=-1)
                + lidar2img[:3, 3]
            )
            pts_2d[:, :2] /= pts_2d[:, 2:3]
            U = np.round(pts_2d[:, 0]).astype(np.int32)
            V = np.round(pts_2d[:, 1]).astype(np.int32)
            depths = pts_2d[:, 2]
            mask = np.logical_and.reduce(
                [
                    V >= 0,
                    V < H,
                    U >= 0,
                    U < W,
                    depths >= 0.1,
                    # depths <= self.max_depth,
                ]
            )
            V, U, depths = V[mask], U[mask], depths[mask]
            sort_idx = np.argsort(depths)[::-1]
            V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]
            depths = np.clip(depths, 0.1, self.max_depth)
            for j, downsample in enumerate(self.downsample):
                if len(gt_depth) < j + 1:
                    gt_depth.append([])
                h, w = (int(H / downsample), int(W / downsample))
                u = np.floor(U / downsample).astype(np.int32)
                v = np.floor(V / downsample).astype(np.int32)
                depth_map = np.ones([h, w], dtype=np.float32) * -1
                depth_map[v, u] = depths
                gt_depth[j].append(depth_map)

        input_dict["gt_depth"] = [np.stack(x) for x in gt_depth]
        return input_dict



@PIPELINES.register_module()
class DenseDepthProbLabelGenerator(object):
    def __init__(
        self,
        max_depth=60,
        min_depth=0.25,
        num_depth=64,
        origin_stride=4,
        strides=[4, 8, 16, 32],
        depth_mode=0,
        image_hw=None,
    ):
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.num_depth = num_depth
        self.origin_stride = origin_stride
        self.strides = [stride // origin_stride for stride in strides]
        self.depth_mode = depth_mode
        # self.image_hw = np.array(image_hw)

    def __call__(self, input_dict):
        aug_config = input_dict.get("aug_config")
        filename = input_dict["depth_filename"]
        depths = [cv2.imread(name, cv2.IMREAD_UNCHANGED) for name in filename]
        N = len(depths)
        new_depths = []
        for i in range(N):
            depth = self._img_transform(
                depths[i] , aug_config,
            )
            new_depths.append(depth)
        
        depth = np.stack(new_depths)
        if self.origin_stride != 1:
            H, W = depth.shape[-2:]
            depth = np.transpose(depth, (1, 2, 0))
            depth = mmcv.imresize(
                depth, (W//self.origin_stride, H//self.origin_stride),
                interpolation="nearest",
            )
            depth = np.transpose(depth, (2, 0, 1))
        input_dict["gt_depth"] = depth
 
        if self.depth_mode == 0:
            return self.get_depth_gt_prob(input_dict)
        elif self.depth_mode == 1:
            return self.get_depth_gt_onehot(input_dict)
        else:
            assert False
  
    def _img_transform(self, img, aug_configs):
        H, W = img.shape[:2]
        resize = aug_configs.get("resize", 1)
        resize_dims = (int(W * resize), int(H * resize))
        crop = aug_configs.get("crop", [0, 0, *resize_dims])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)

        origin_dtype = img.dtype
        if origin_dtype != np.uint8:
            min_value = img.min()
            max_vaule = img.max()
            scale = 255 / (max_vaule - min_value)
            img = (img - min_value) * scale
            img = np.uint8(img)
        img = Image.fromarray(img)
        img = img.resize(resize_dims).crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        img = np.array(img).astype(np.float32)
        if origin_dtype != np.uint8:
            img = img.astype(np.float32)
            img = img / scale + min_value

        return img

    def get_depth_gt_prob(self, input_dict):
        depth = input_dict["gt_depth"]
        depth = depth[:, None]
        depth_valid_mask = depth > 0
        depth = np.clip(
            depth,
            a_min=self.min_depth,
            a_max=self.max_depth,
        )
        depth_anchor = np.linspace(
            self.min_depth, self.max_depth, self.num_depth)[:, None, None]
        distance = np.abs(depth - depth_anchor)
        mask = distance < (depth_anchor[1] - depth_anchor[0])
        depth_gt = np.where(mask, depth_anchor, 0)
        y = depth_gt.sum(axis=1, keepdims=True) - depth_gt
        depth_prob_gt = np.where(
            (depth_gt != 0) & depth_valid_mask,
            (depth - y) / (depth_gt - y),
            0,
        )
        views, _, H, W = depth.shape
        gt = []
        for s in self.strides:
            gt_tmp = np.reshape(
                depth_prob_gt, (views, self.num_depth, H//s, s, W//s, s))
            gt_tmp = gt_tmp.sum(axis=-1).sum(axis=3)
            mask_tmp = depth_valid_mask.reshape(views, 1, H//s, s, W//s, s)
            mask_tmp = mask_tmp.sum(axis=-1).sum(axis=3)
            gt_tmp /= np.clip(mask_tmp, a_min=1, a_max=None)
            gt_tmp = gt_tmp.reshape(views, self.num_depth, -1)
            gt_tmp = np.transpose(gt_tmp, (0, 2, 1))
            gt.append(gt_tmp)
        gt = np.concatenate(gt, axis=1)
        gt = np.clip(gt, a_min=0.0, a_max=1.0)
        input_dict["depth_prob_gt"] = gt
        return input_dict

    def get_depth_gt_onehot(self, input_dict):
        depth = input_dict["dense_depth"]
        depth = np.where(depth > 0, depth, 1e5)
        views, H, W = depth.shape
        depth_interval = (self.max_depth - self.min_depth) / (self.num_depth - 1)
        gt = []
        for s in self.strides:
            gt_tmp = np.reshape(
                depth, (views, H//s, s, W//s, s))
            gt_tmp = gt_tmp.transpose(0, 1, 3, 2, 4).reshape((views, H//s, W//s, -1))
            gt_tmp = gt_tmp.min(axis=-1)
            gt_tmp = (gt_tmp - self.min_depth) / depth_interval + 1
            gt_tmp = np.where(
                (gt_tmp >= 0) & (gt_tmp < self.num_depth + 1),
                gt_tmp, 
                0,
            )
            gt_tmp = F.one_hot(
                torch.tensor(gt_tmp).long(), 
                num_classes=self.num_depth + 1,
            )[..., 1:]
            gt_tmp = gt_tmp.reshape(views, -1, self.num_depth)
            gt.append(gt_tmp)
        gt = np.concatenate(gt, axis=1)
        gt = np.clip(gt, a_min=0.0, a_max=1.0)
        input_dict["depth_prob_gt"] = gt
        return input_dict





@PIPELINES.register_module()
class NuScenesSparse4DAdaptor(object):
    def __init(self):
        pass

    def __call__(self, input_dict):
        input_dict["projection_mat"] = np.float32(
            np.stack(input_dict["lidar2img"])
        )
        input_dict["image_wh"] = np.ascontiguousarray(
            np.array(input_dict["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
        )
        input_dict["T_global_inv"] = np.linalg.inv(input_dict["lidar2global"])
        input_dict["T_global"] = input_dict["lidar2global"]
        if "cam_intrinsic" in input_dict:
            input_dict["cam_intrinsic"] = np.float32(
                np.stack(input_dict["cam_intrinsic"])
            )
            input_dict["focal"] = input_dict["cam_intrinsic"][..., 0, 0]
        if "instance_inds" in input_dict:
            input_dict["instance_id"] = input_dict["instance_inds"]

        if "gt_bboxes_3d" in input_dict:
            input_dict["gt_bboxes_3d"][:, 6] = self.limit_period(
                input_dict["gt_bboxes_3d"][:, 6], offset=0.5, period=2 * np.pi
            )
            input_dict["gt_bboxes_3d"] = DC(
                to_tensor(input_dict["gt_bboxes_3d"]).float()
            )


        if 'fake_bboxes_3d' in input_dict:
            input_dict["fake_bboxes_3d"] = DC(
                to_tensor(input_dict["fake_bboxes_3d"]).float()
            )
            input_dict["fake_agent_fut_trajs"] = DC(
                to_tensor(input_dict["fake_agent_fut_trajs"]).float()
            )
            input_dict["fake_bboxes_3d_5hz"] = DC(
                to_tensor(input_dict["fake_bboxes_3d_5hz"]).float()
            )
            input_dict["fake_agent_fut_trajs_5hz"] = DC(
                to_tensor(input_dict["fake_agent_fut_trajs_5hz"]).float()
            )



            # input_dict["fake_defensive_ego_fut_trajs"] = DC(
            #     to_tensor(input_dict["fake_defensive_ego_fut_trajs"]).float()
            # )
            # input_dict["fake_aggressive_ego_fut_trajs"] = DC(
            #     to_tensor(input_dict["fake_aggressive_ego_fut_trajs"]).float()
            # )
            

        if "gt_labels_3d" in input_dict:
            input_dict["gt_labels_3d"] = DC(
                to_tensor(input_dict["gt_labels_3d"]).long()
            )

        # if "gt_labels_3d" in input_dict:
        #     input_dict["gt_labels_3d"] = DC(
        #         to_tensor(input_dict["gt_labels_3d"]).long()
        #     )

        imgs = [img.transpose(2, 0, 1) for img in input_dict["img"]]
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        input_dict["img"] = DC(to_tensor(imgs), stack=True)

        for key in [
            'gt_map_labels', 
            'gt_map_pts',
            'gt_agent_fut_trajs',
            'gt_agent_fut_masks',
            'agents_spatial_interval_5_traj',
            'agents_spatial_interval_5_mask',
            'traffic_light_state', 
        ]:
            if key not in input_dict:
                continue
            input_dict[key] = DC(to_tensor(input_dict[key]), stack=False, cpu_only=False) 


        ego_status_list = ['gt_ego_fut_trajs', 'gt_ego_fut_masks', 'gt_ego_fut_cmd', 'ego_status',]
        if 'pre_ego_status' in input_dict:
            ego_status_list += ['pre_ego_status']

        for key in ego_status_list:
            if key not in input_dict:
                continue
            input_dict[key] = DC(to_tensor(input_dict[key]), stack=True, cpu_only=False, pad_dims=None)
        
        return input_dict

    def limit_period(
        self, val: np.ndarray, offset: float = 0.5, period: float = np.pi
    ) -> np.ndarray:
        limited_val = val - np.floor(val / period + offset) * period
        return limited_val


@PIPELINES.register_module()
class InstanceNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict["gt_labels_3d"]
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=np.bool_
        )
        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = input_dict["gt_labels_3d"][gt_bboxes_mask]
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][gt_bboxes_mask]
        if "gt_agent_fut_trajs" in input_dict:
            input_dict["gt_agent_fut_trajs"] = input_dict["gt_agent_fut_trajs"][gt_bboxes_mask]
            input_dict["gt_agent_fut_masks"] = input_dict["gt_agent_fut_masks"][gt_bboxes_mask]
        if "agents_spatial_interval_5_traj" in input_dict:
            input_dict["agents_spatial_interval_5_traj"] = input_dict["agents_spatial_interval_5_traj"][gt_bboxes_mask]
            input_dict["agents_spatial_interval_5_mask"] = input_dict["agents_spatial_interval_5_mask"][gt_bboxes_mask]
        if "traffic_light_state" in input_dict:
            input_dict["traffic_light_state"] = input_dict["traffic_light_state"][gt_bboxes_mask]

        return input_dict

    def __repr__(self): 
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(classes={self.classes})"
        return repr_str


@PIPELINES.register_module()
class CircleObjectRangeFilter(object):
    def __init__(
        self, class_dist_thred=[52.5] * 5 + [31.5] + [42] * 3 + [31.5]
    ):
        self.class_dist_thred = class_dist_thred

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        dist = np.sqrt(
            np.sum(gt_bboxes_3d[:, :2] ** 2, axis=-1)
        )
        mask = np.array([False] * len(dist))
        for label_idx, dist_thred in enumerate(self.class_dist_thred):
            mask = np.logical_or(
                mask,
                np.logical_and(gt_labels_3d == label_idx, dist <= dist_thred),
            )

        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask]

        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][mask]
        if "gt_agent_fut_trajs" in input_dict:
            input_dict["gt_agent_fut_trajs"] = input_dict["gt_agent_fut_trajs"][mask]
            input_dict["gt_agent_fut_masks"] = input_dict["gt_agent_fut_masks"][mask]
        if 'agents_spatial_interval_5_traj' in input_dict:
            input_dict["agents_spatial_interval_5_traj"] = input_dict["agents_spatial_interval_5_traj"][mask]
            input_dict["agents_spatial_interval_5_mask"] = input_dict["agents_spatial_interval_5_mask"][mask]

        if "gt_names" in input_dict:
            input_dict["gt_names"] = input_dict["gt_names"][mask]

        if "traffic_light_state" in input_dict:
            input_dict["traffic_light_state"] = input_dict["traffic_light_state"][mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(class_dist_thred={self.class_dist_thred})"
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str
