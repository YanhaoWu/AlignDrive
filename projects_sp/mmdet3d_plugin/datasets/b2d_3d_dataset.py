import math
import os
from os import path as osp

import tempfile
import copy

import pickle
import json


import numpy as np

import torch
from torch.utils.data import Dataset

from pyquaternion import Quaternion
from shapely.geometry import LineString
from .evaluation.detection.nuscenes_styled_eval_utils import (
    DetectionMetrics,
    EvalBoxes,
    DetectionBox,
    center_distance,
    accumulate,
    DetectionMetricDataList,
    calc_ap,
    calc_tp,
)

import mmcv
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from .trajectory_sampling import Bench2DriveTrajectoryMixin
from .utils import generate_fake_bbox

CAM_SORT = [
    'rgb_front',
    'rgb_front_right',
    'rgb_front_left',
    'rgb_back',
    'rgb_back_left',
    'rgb_back_right',
]


@DATASETS.register_module()
class B2D3DDataset(Bench2DriveTrajectoryMixin, Dataset):
    """Bench2Drive camera, detection, map, and aligned-planning dataset."""

    CLASSES = [
        'car',
        'van',
        'truck',
        'bicycle',
        'traffic_sign',
        'traffic_cone',
        'traffic_light',
        'pedestrian',
        'others',
    ]
    MAP_CLASSES = [
        'Broken',
        'Solid',
        'SolidSolid',
    ]
    ID_COLOR_MAP = [
        (59, 59, 238),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 255),
        (0, 127, 255),
        (71, 130, 255),
        (127, 127, 0),
    ]

    def __init__(
        self,
        ann_file,
        waypoints_file=None,
        agents_dp_file=None,
        pipeline=None,
        data_root=None,
        classes=None,
        map_classes=None,
        name_mapping=None,
        load_interval=1,
        modality=None,
        sample_interval=5,
        past_frames=2,
        future_frames=6,
        test_mode=False,
        vis_score_threshold=0.25,
        data_aug_conf=None,
        sequences_split_num=1,
        with_seq_flag=False,
        keep_consistent_seq_aug=True,
        work_dir=None,
        eval_config=None,
        use_cmd=True,
        use_generated_img=False,
        generated_img_root=None,
        debug_mode=False,
        replace_img=False,
        resample_split=None,
        num_map_classes=3,
        gloalpoint_extend=True,
        augment_bbox=False,
        find_interactive_agents=False,
    ):
        self.load_interval = load_interval
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        print("Anotation file: ", self.ann_file)
        self.waypoints_file = waypoints_file
        self.test_mode = test_mode
        self.modality = modality
        self.box_mode_3d = 0
        self.sample_interval = sample_interval
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.debug_mode = debug_mode
        self.num_map_classes = num_map_classes
        self.gloalpoint_extend = gloalpoint_extend
        self.augment_bbox = augment_bbox
        self.find_interactive_agents = find_interactive_agents

        if classes is not None:
            self.CLASSES = classes
        if map_classes is not None:
            self.MAP_CLASSES = map_classes

        self.num_map_classes = len(self.MAP_CLASSES)
        print("Number of map classes: ", self.num_map_classes)

        self.NameMapping = name_mapping
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}

        self.data_infos = self.load_annotations(self.ann_file)
        if self.waypoints_file is not None:
            self.waypoints_info = self.load_annotations(self.waypoints_file)
            assert len(self.data_infos) == len(
                self.waypoints_info
            ), "The length of data_infos and waypoints_info should be the same."
        else:
            self.waypoints_info = None

        if agents_dp_file is not None:
            self.agents_dp_info = self.load_annotations(agents_dp_file)
            assert len(self.data_infos) == len(
                self.agents_dp_info
            ), "The length of data_infos and agents_dp_info should be the same."
        else:
            self.agents_dp_info = None

        self.data_obtain = False
        self.clip_info = None

        self.replace_img = replace_img
        if self.replace_img:
            print("----------------WANING: Replace image with selected image---------------------")
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=False,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.vis_score_threshold = vis_score_threshold

        self.data_aug_conf = data_aug_conf
        self.sequences_split_num = sequences_split_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug
        if with_seq_flag:
            self._set_sequence_group_flag()

        self.work_dir = work_dir
        self.eval_config = eval_config
        self.use_cmd = use_cmd

        self.use_generated_img = use_generated_img
        self.generated_img_root = generated_img_root

        self.eval_cfg = {
            "dist_ths": [0.5, 1.0, 2.0, 4.0],
            "dist_th_tp": 2.0,
            "min_recall": 0.1,
            "min_precision": 0.1,
            "mean_ap_weight": 5,
            "class_names": [
                'car',
                'van',
                'truck',
                'bicycle',
                'traffic_sign',
                'traffic_cone',
                'traffic_light',
                'pedestrian',
            ],
            "tp_metrics": ['trans_err', 'scale_err', 'orient_err', 'vel_err'],
            "err_name_maping": {
                'trans_err': 'mATE',
                'scale_err': 'mASE',
                'orient_err': 'mAOE',
                'vel_err': 'mAVE',
                'attr_err': 'mAAE',
            },
            "class_range": {
                'car': (50, 50),
                'van': (50, 50),
                'truck': (50, 50),
                'bicycle': (40, 40),
                'traffic_sign': (30, 30),
                'traffic_cone': (30, 30),
                'traffic_light': (30, 30),
                'pedestrian': (40, 40),
            },
        }

        self.resample_split = resample_split
        self.version = 'v1.0-trainval'
        if 'val' in self.ann_file:
            self.mode = 'val'
        else:
            self.mode = 'train'

        if self.resample_split is not None and self.mode == 'train':

            split_path = self.resample_split['train_split']
            self.resample_dataset(split_path)
            print("Resampled dataset, there are {} frames in total.".format(len(self.data_infos)))

    def __len__(self):
        return len(self.data_infos)

    def resample_dataset(self, train_split_path):

        with open(train_split_path, 'rb') as f:
            train_split = pickle.load(f)

        new_data_infos = []
        for i in range(len(train_split)):
            new_data_infos.append(self.data_infos[train_split[i]])
        self.data_infos = new_data_infos

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        if self.sequences_split_num == -1:
            self.flag = np.arange(len(self.data_infos))
            return

        res = []

        curr_folder = self.data_infos[0]["folder"]
        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and self.data_infos[idx]["folder"] != curr_folder:

                curr_sequence += 1
                curr_folder = self.data_infos[idx]["folder"]
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == "all":
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(bin_counts[curr_flag] / self.sequences_split_num),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )

                    for sub_seq_idx in curr_sequence_length[1:] - curr_sequence_length[:-1]:
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert (
                    len(np.bincount(new_flags))
                    == len(np.bincount(self.flag)) * self.sequences_split_num
                )
                self.flag = np.array(new_flags, dtype=np.int64)

    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
            rotate_3d = np.random.uniform(*self.data_aug_conf["rot3d_range"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
            rotate_3d = 0
        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_config

    def __getitem__(self, idx):
        if self.data_obtain:
            for clip in self.clip_info:

                start_index = clip['start_index']
                end_index = clip['end_index']
                if idx >= start_index and idx < end_index:
                    break
            agent_dp, valid_mask = self.get_fut_agent_dp(
                idx=idx, end_idx=end_index, interval_distance=5, num_points=6
            )
            return agent_dp, valid_mask

        else:
            if isinstance(idx, dict):
                aug_config = idx["aug_config"]
                idx = idx["idx"]
            else:
                aug_config = self.get_augmentation()
            data = self.get_data_info(idx)
            data["aug_config"] = aug_config

            data = self.pipeline(data)
            data['idx'] = idx
            return data

    def match_traffic_lights(self, data):
        """Return aligned box states: 0 affects ego, 1 does not, 2 is not a light."""
        light_states = {}
        for traffic_light in data['traffic_light']:
            light_states.setdefault(
                int(traffic_light['id']), 0 if traffic_light['affects_ego'] else 1
            )
        return [light_states.get(int(box_id), 2) for box_id in data['gt_ids']]

    def replace_input_img(self, data, base_path='./output/b2d_test'):
        """Replace camera paths with a fixed six-camera debug sample."""
        data['img_filename'] = [osp.join(base_path, camera + '.png') for camera in CAM_SORT]
        return data

    def load_annotations(self, ann_file):
        """Load an ordered annotation list with the configured subsampling interval."""
        data = mmcv.load(ann_file, file_format="pkl")
        data_infos = data[:: self.load_interval]
        return data_infos

    def anno2geom(self, annos):
        """Convert map polylines to Shapely geometries grouped by class."""
        map_geoms = {}
        for label, anno_list in annos.items():
            map_geoms[label] = []
            for anno in anno_list:
                geom = LineString(anno)
                map_geoms[label].append(geom)
        return map_geoms

    def get_data_info(self, index):
        """Assemble camera, perception, waypoint, and augmentation fields for one sample."""
        info = self.data_infos[index]
        map_class_count = 3 if self.num_map_classes == 3 else 6
        map_annotations = {label: info['map_annos'][label] for label in range(map_class_count)}
        sample = dict(
            token=info['token'],
            timestamp=info['timestamp'] / 1e6,
            map_infos=map_annotations,
            map_geoms=self.anno2geom(map_annotations),
        )
        if self.modality['use_camera']:
            sample.update(self._get_camera_info(info))
        sample.update(self.get_ann_info(index))
        if self.replace_img:
            sample = self.replace_input_img(sample)
        sample.update(self._get_waypoint_info(index))
        if self.augment_bbox:
            sample.update(self._get_planning_augmentation(sample))
        return sample

    def _get_camera_info(self, info):
        """Resolve six-camera files and calibration matrices in dataset sensor order."""
        image_paths, depth_paths = [], []
        lidar_to_image, lidar_to_camera, intrinsics = [], [], []
        lidar_to_ego = info['sensors']['LIDAR_TOP']['lidar2ego']
        lidar_to_global = self.invert_pose(info['sensors']['LIDAR_TOP']['world2lidar'])
        for sensor_name, camera in info['sensors'].items():
            if 'CAM' not in sensor_name:
                continue
            camera_path = camera['data_path']
            image_path = osp.join(self.data_root, camera_path)
            if not osp.isfile(image_path) and not osp.isabs(camera_path):
                legacy_path = osp.join(self.data_root, 'v1', camera_path)
                if osp.isfile(legacy_path):
                    image_path = legacy_path
            image_paths.append(image_path)
            depth_paths.append(image_path.replace('rgb_', 'depth_').replace('.jpg', '.png'))
            intrinsic = copy.deepcopy(camera['intrinsic'])
            intrinsics.append(intrinsic)
            projection = np.eye(4)
            projection[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
            transform = (self.invert_pose(camera['cam2ego']) @ lidar_to_ego).T
            lidar_to_camera.append(transform)
            lidar_to_image.append(projection @ transform.T)
        return dict(
            img_filename=image_paths,
            depth_filename=depth_paths,
            lidar2img=lidar_to_image,
            lidar2cam=lidar_to_camera,
            cam_intrinsic=intrinsics,
            lidar2global=lidar_to_global,
        )

    def _get_waypoint_info(self, index):
        """Copy aligned waypoint annotations without changing the cached pkl contents."""
        annotations = {}
        if self.waypoints_info is not None:
            for key, value in self.waypoints_info[index].items():
                annotations[key] = value.astype(np.bool_) if 'mask' in key else copy.deepcopy(value)
        if self.agents_dp_info is not None:
            for key, value in self.agents_dp_info[index].items():
                if 'mask' in key:
                    value = value.astype(np.bool_)
                annotations['agents_' + key] = value.astype(np.float32)
        return annotations

    def _get_planning_augmentation(self, sample):
        """Generate synthetic agents and aligned defensive/aggressive planning targets."""
        augmented = generate_fake_bbox(
            sample['spatial_interval_2_traj'].copy(),
            sample['gt_ego_fut_trajs'].copy(),
            sample['temporal_interval_2_traj'].copy(),
            sample['gt_bboxes_3d'].copy(),
            sample['gt_agent_fut_trajs'].copy(),
            safety_threshold=5.0,
        )
        output_keys = (
            'fake_bboxes_3d',
            'fake_agent_fut_trajs',
            'fake_bboxes_3d_5hz',
            'fake_agent_fut_trajs_5hz',
            'fake_defensive_ego_fut_trajs',
            'fake_aggressive_ego_fut_trajs',
            'fake_defensive_ego_fut_trajs_5hz',
            'fake_aggressive_ego_fut_trajs_5hz',
            'valid_flag',
            'aggressive_valid',
        )
        return dict(zip(output_keys, augmented))

    def obtain_goal_points(
        self, info, keys_list=('command_near_xy', 'command_far_xy'), extend=False
    ):
        """Transform navigation targets into LiDAR coordinates, optionally sampling their segment."""
        ego_position = np.asarray(info['ego_translation'])[:2]
        heading = -(info['ego_yaw'] - np.pi / 2)
        rotation = np.array(
            [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
        )
        goal_points = [
            (rotation @ (np.asarray(info[key])[:2] - ego_position)).reshape(1, 2)
            for key in keys_list
        ]
        if extend:
            if len(goal_points) != 2:
                raise ValueError('Exactly two navigation targets are required for extension')
            merged_line = np.linspace(goal_points[0][0], goal_points[1][0], 6)
        else:
            merged_line = goal_points[-1]
        return goal_points[0], goal_points[1], merged_line

    def get_ann_info(self, index):
        """Collect detection labels, agent motion, ego waypoints, and navigation targets."""
        info = self.data_infos[index]
        mask = info['num_points'] != 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        for i in range(len(gt_names_3d)):
            if gt_names_3d[i] in self.NameMapping.keys():
                gt_names_3d[i] = self.NameMapping[gt_names_3d[i]]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            scenario_name=info['folder'].split('_Town')[0],
        )

        if "gt_ids" in info:
            instance_inds = np.array(info["gt_ids"], dtype=np.int64)[mask]
            anns_results["instance_inds"] = instance_inds

        gt_agent_fut_trajs, gt_agent_fut_masks = self.get_fut_agent(
            index, self.sample_interval, self.future_frames
        )
        anns_results['gt_agent_fut_trajs'] = gt_agent_fut_trajs[mask]
        anns_results['gt_agent_fut_masks'] = gt_agent_fut_masks[mask]

        (ego_his_trajs, ego_fut_trajs, ego_fut_masks, command) = self.get_ego_trajs(
            index, self.sample_interval, self.past_frames, self.future_frames
        )
        anns_results['gt_ego_fut_trajs'] = ego_fut_trajs
        anns_results['gt_ego_fut_masks'] = ego_fut_masks
        anns_results['gt_ego_fut_cmd'] = command
        anns_results['ego_his_trajs'] = ego_his_trajs

        (ego_his_trajs, ego_fut_trajs, ego_fut_masks, command) = self.get_ego_trajs(
            index, 2, self.past_frames, 15
        )
        anns_results['gt_ego_fut_trajs_5hz'] = ego_fut_trajs
        anns_results['gt_ego_fut_masks_5hz'] = ego_fut_masks

        tp_near, tp_far, tp_mearge = self.obtain_goal_points(
            info, keys_list=['command_near_xy', 'command_far_xy'], extend=self.gloalpoint_extend
        )
        anns_results["tp_near"] = tp_near
        anns_results["tp_far"] = tp_far
        anns_results["tp_line"] = tp_mearge

        if 'traffic_light' in info:
            traffic_light_state = self.match_traffic_lights(info)
            anns_results['traffic_light_state'] = np.array(traffic_light_state)[mask]

        ego_status = np.zeros(10)
        ego_status[:3] = info["ego_accel"]
        ego_status[3:6] = info["ego_rotation_rate"]
        ego_status[6:9] = info["ego_vel"]
        anns_results["ego_status"] = ego_status.astype(np.float32)
        anns_results["route_name"] = info['folder'].split('/')[-1]

        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None, tracking=False):
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            box3d = det['boxes_3d']
            scores = det['scores_3d']
            labels = det['labels_3d']
            box_gravity_center = box3d[:, :3]
            box_dims = box3d[:, 3:6]
            box_yaw = box3d[:, 6]
            sample_token = self.data_infos[sample_id]['token']

            for i in range(len(box3d)):
                quat = list(Quaternion(axis=[0, 0, 1], radians=box_yaw[i]))
                velocity = [box3d[i, 7].item(), box3d[i, 8].item()]
                name = mapped_class_names[labels[i]]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box_gravity_center[i].tolist(),
                    size=box_dims[i].tolist(),
                    rotation=quat,
                    velocity=velocity,
                    detection_name=name,
                    detection_score=scores[i].item(),
                    attribute_name=name,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(self, result_path, logger=None, result_name="img_bbox", tracking=False):

        with open(result_path) as f:
            result_data = json.load(f)
        pred_boxes = EvalBoxes.deserialize(result_data['results'], DetectionBox)
        meta = result_data['meta']

        gt_boxes = self.load_gt()

        metric_data_list = DetectionMetricDataList()
        for class_name in self.eval_cfg['class_names']:
            for dist_th in self.eval_cfg['dist_ths']:
                md = accumulate(gt_boxes, pred_boxes, class_name, center_distance, dist_th)
                metric_data_list.set(class_name, dist_th, md)
                metrics = DetectionMetrics(self.eval_cfg)

        for class_name in self.eval_cfg['class_names']:

            for dist_th in self.eval_cfg['dist_ths']:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(
                    metric_data, self.eval_cfg['min_recall'], self.eval_cfg['min_precision']
                )
                metrics.add_label_ap(class_name, dist_th, ap)

            for metric_name in self.eval_cfg['tp_metrics']:
                metric_data = metric_data_list[(class_name, self.eval_cfg['dist_th_tp'])]
                tp = calc_tp(metric_data, self.eval_cfg['min_recall'], metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = meta.copy()
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))

        print()
        print('Per-class results:')
        print('Object Class\tAP\tATE\tASE\tAOE\tAVE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print(
                '%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                % (
                    class_name,
                    class_aps[class_name],
                    class_tps[class_name]['trans_err'],
                    class_tps[class_name]['scale_err'],
                    class_tps[class_name]['orient_err'],
                    class_tps[class_name]['vel_err'],
                )
            )

        detail = dict()
        metric_prefix = 'bbox_NuScenes'
        for name in self.eval_cfg['class_names']:
            for k, v in metrics_summary['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics_summary['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics_summary['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix, self.eval_cfg['err_name_maping'][k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics_summary['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics_summary['mean_ap']

        return detail

    def load_gt(self):
        all_annotations = EvalBoxes()
        for i in range(len(self.data_infos)):
            sample_boxes = []
            sample_data = self.data_infos[i]

            gt_boxes = sample_data['gt_boxes']

            for j in range(gt_boxes.shape[0]):
                class_name = self.NameMapping[sample_data['gt_names'][j]]
                if not class_name in self.eval_cfg['class_range'].keys():
                    continue
                range_x, range_y = self.eval_cfg['class_range'][class_name]
                if abs(gt_boxes[j, 0]) > range_x or abs(gt_boxes[j, 1]) > range_y:
                    continue
                sample_boxes.append(
                    DetectionBox(
                        sample_token=sample_data['token'],
                        translation=gt_boxes[j, 0:3],
                        size=gt_boxes[j, 3:6],
                        rotation=list(Quaternion(axis=[0, 0, 1], radians=gt_boxes[j, 6])),
                        velocity=gt_boxes[j, 7:9],
                        num_pts=int(sample_data['num_points'][j]),
                        detection_name=self.NameMapping[sample_data['gt_names'][j]],
                        detection_score=-1.0,
                        attribute_name=self.NameMapping[sample_data['gt_names'][j]],
                    )
                )
            all_annotations.add_boxes(sample_data['token'], sample_boxes)
        return all_annotations

    def format_results(self, results, jsonfile_prefix=None, tracking=False):
        assert isinstance(results, list), "results must be a list"

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        if not ("pts_bbox" in results[0] or "img_bbox" in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix, tracking=tracking)
        else:
            result_files = dict()
            for name in results[0]:
                print(f"\nFormating bboxes of {name}")
                results_ = [out[name] for out in results]
                tmp_file_ = jsonfile_prefix
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_, tracking=tracking)}
                )
        return result_files, tmp_dir

    def format_map_results(self, results, prefix=None):
        submissions = {
            'results': {},
        }

        for j, pred in enumerate(results):
            '''
            For each case, the result should be formatted as Dict{'vectors': [], 'scores': [], 'labels': []}
            'vectors': List of vector, each vector is a array([[x1, y1], [x2, y2] ...]),
                contain all vectors predicted in this sample.
            'scores: List of score(float),
                contain scores of all instances in this sample.
            'labels': List of label(int),
                contain labels of all instances in this sample.
            '''
            if pred is None:
                continue
            pred = pred['img_bbox']

            single_case = {'vectors': [], 'scores': [], 'labels': []}
            token = self.data_infos[j]['token']
            for i in range(len(pred['scores'])):
                score = pred['scores'][i]
                label = pred['labels'][i]
                vector = pred['vectors'][i]

                if len(vector) < 2:
                    continue

                single_case['vectors'].append(vector)
                single_case['scores'].append(score)
                single_case['labels'].append(label)

            submissions['results'][token] = single_case

        out_path = osp.join(prefix, 'submission_vector.json')
        print(f'saving submissions results to {out_path}')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mmcv.dump(submissions, out_path)
        return out_path

    def _evaluate_single_motion(
        self, results, result_path, logger=None, metric='bbox', result_name='pts_bbox'
    ):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from scipy.optimize import linear_sum_assignment
        from .evaluation.planning.planning_eval import PlanningMetric, write_to_table

        motion_metric = PlanningMetric()

        for index in mmcv.track_iter_progress(range(len(results))):
            result = results[index]
            if result is None:
                continue
            GT_data = self.data_infos[index]
            GT_boxes = GT_data['gt_boxes']
            detect_boxes = []

            detect_boxes = result['img_bbox']['boxes_3d'].numpy()
            trajs = result['img_bbox']['trajs_3d'].numpy()
            mask = result['img_bbox']["cls_scores"].numpy() >= 0.3
            detect_boxes = detect_boxes[mask]
            trajs = trajs[mask]

            if len(GT_boxes) == 0 or len(detect_boxes) == 0:
                continue
            cost_matrix = np.zeros((len(GT_boxes), len(detect_boxes)))
            for i in range(len(GT_boxes)):
                for j in range(len(detect_boxes)):
                    cost_matrix[i][j] = np.linalg.norm(GT_boxes[i][:2] - detect_boxes[j][:2])
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            gt_agent_fut_trajs, gt_agent_fut_masks = self.get_fut_agent(
                index, self.sample_interval, self.future_frames
            )
            gt_agent_fut_masks_sum = gt_agent_fut_masks.sum(axis=1)

            gt_agent_fut_masks[gt_agent_fut_masks_sum <= 1, :] = 0

            gt_agent_fut_trajs = gt_agent_fut_trajs[row_ind]
            gt_agent_fut_masks = gt_agent_fut_masks[row_ind]
            gt_agent_fut_masks = np.repeat(gt_agent_fut_masks[:, :, np.newaxis], 2, axis=2)
            pred_box_trajs = trajs[col_ind]

            for j in range(len(row_ind)):
                motion_metric.update(
                    torch.tensor(pred_box_trajs[j : j + 1, :6, :2]),
                    torch.tensor(gt_agent_fut_trajs[j : j + 1, :6, :2]),
                    torch.tensor(gt_agent_fut_masks[j : j + 1, :6, :2]),
                    fut_boxes=None,
                    last_final_planning=None,
                )

        motion_results = motion_metric.compute()
        motion_metric.reset()

        metric_dict = {}
        motion_tab, metric_dict = write_to_table(
            motion_results, prefix='motion', metric_dict=metric_dict
        )

        if logger is not None:
            print_log('\n' + str(motion_tab), logger=logger)
        else:
            print('\n' + str(motion_tab))

        new_dict = {}
        for keys in motion_results.keys():
            new_keys = 'motion_' + keys
            new_dict[new_keys] = motion_results[keys]
        motion_results = new_dict
        return motion_results

    def evaluate(
        self,
        results,
        eval_mode,
        metric=None,
        logger=None,
        jsonfile_prefix=None,
        result_names=["img_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        res_path = "results.pkl"
        res_path = osp.join(self.work_dir, res_path)
        print('All Results write to', res_path)
        mmcv.dump(results, res_path)

        results_dict = dict()
        if eval_mode['with_det']:
            self.tracking = eval_mode["with_tracking"]
            self.tracking_threshold = eval_mode["tracking_threshold"]
            for metric in ["detection", "tracking"]:
                tracking = metric == "tracking"
                if tracking and not self.tracking:
                    continue
                result_files, tmp_dir = self.format_results(
                    results, jsonfile_prefix=self.work_dir, tracking=tracking
                )

                if isinstance(result_files, dict):
                    for name in result_names:
                        ret_dict = self._evaluate_single(result_files[name], tracking=tracking)
                    results_dict.update(ret_dict)
                elif isinstance(result_files, str):
                    ret_dict = self._evaluate_single(result_files, tracking=tracking)
                    results_dict.update(ret_dict)

                if tmp_dir is not None:
                    tmp_dir.cleanup()

        if eval_mode['with_map']:
            from .evaluation.map.vector_eval import VectorEvaluate

            self.map_evaluator = VectorEvaluate(self.eval_config)
            result_path = self.format_map_results(results, prefix=self.work_dir)
            map_results_dict = self.map_evaluator.evaluate(result_path, logger=logger)
            results_dict.update(map_results_dict)

        if eval_mode['with_motion']:

            motion_results_dict = self._evaluate_single_motion(
                results, self.work_dir, logger=logger
            )

            try:
                results_dict.update(motion_results_dict)
            except:
                print("Error in motion eval")

        if eval_mode['with_planning']:
            from .evaluation.planning.planning_eval import planning_eval

            planning_results_dict = planning_eval(results, self.eval_config, logger=logger)
            results_dict.update(planning_results_dict)

        if show or out_dir:
            self.show(results, save_dir=out_dir, show=show, pipeline=pipeline)

        metric_str = '\n'
        if "img_bbox_NuScenes/NDS" in results_dict:
            metric_str += f'mAP: {results_dict.get("img_bbox_NuScenes/mAP"):.4f}\n'
            metric_str += f'mATE: {results_dict.get("img_bbox_NuScenes/mATE"):.4f}\n'
            metric_str += f'mASE: {results_dict.get("img_bbox_NuScenes/mASE"):.4f}\n'
            metric_str += f'mAOE: {results_dict.get("img_bbox_NuScenes/mAOE"):.4f}\n'
            metric_str += f'mAVE: {results_dict.get("img_bbox_NuScenes/mAVE"):.4f}\n'
            metric_str += f'mAAE: {results_dict.get("img_bbox_NuScenes/mAAE"):.4f}\n'
            metric_str += f'NDS: {results_dict.get("img_bbox_NuScenes/NDS"):.4f}\n\n'

        if "img_bbox_NuScenes/amota" in results_dict:
            metric_str += f'AMOTA: {results_dict["img_bbox_NuScenes/amota"]:.4f}\n'
            metric_str += f'AMOTP: {results_dict["img_bbox_NuScenes/amotp"]:.4f}\n'
            metric_str += f'RECALL: {results_dict["img_bbox_NuScenes/recall"]:.4f}\n'
            metric_str += f'MOTAR: {results_dict["img_bbox_NuScenes/motar"]:.4f}\n'
            metric_str += f'MOTA: {results_dict["img_bbox_NuScenes/mota"]:.4f}\n'
            metric_str += f'MOTP: {results_dict["img_bbox_NuScenes/motp"]:.4f}\n'
            metric_str += f'IDS: {results_dict["img_bbox_NuScenes/ids"]}\n\n'

        if "mAP_normal" in results_dict:

            metric_str += f'mAP_normal= {results_dict["mAP_normal"]:.4f}\n\n'

        if "car_EPA" in results_dict:
            metric_str += 'Car / Ped\n'
            metric_str += (
                f'epa= {results_dict["car_EPA"]:.4f} / {results_dict["pedestrian_EPA"]:.4f}\n'
            )
            metric_str += f'ade= {results_dict["car_min_ade_err"]:.4f} / {results_dict["pedestrian_min_ade_err"]:.4f}\n'
            metric_str += f'fde= {results_dict["car_min_fde_err"]:.4f} / {results_dict["pedestrian_min_fde_err"]:.4f}\n'
            metric_str += f'mr= {results_dict["car_miss_rate_err"]:.4f} / {results_dict["pedestrian_miss_rate_err"]:.4f}\n\n'

        if "L2" in results_dict:
            metric_str += f'obj_box_col: {(results_dict["obj_box_col"]*100):.3f}%\n'
            metric_str += f'L2: {results_dict["L2"]:.4f}\n\n'

        print_log(metric_str, logger=logger)
        return results_dict

    def show(self, results, save_dir=None, show=False, pipeline=None):
        """Render perception debug results using the shared visualization tools."""
        from tools.visualization.perception import render_predictions

        return render_predictions(self, results, save_dir=save_dir, pipeline=pipeline)
