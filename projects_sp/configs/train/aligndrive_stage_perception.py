"""Perception-stage settings; shared modules are inherited from _base_."""

_base_ = [
    '../_base_/perception.py', '../_base_/bench2drive.py', '../_base_/runtime.py',
    '../_base_/training.py'
]

work_dir = './work_dirs/aligndrive_stage_perception'

load_from = None

evaluation = {
    'interval': None,
    'eval_mode': {
        'with_det': True,
        'with_tracking': True,
        'with_map': True,
        'with_motion': False,
        'with_planning': False,
        'tracking_threshold': 0.2,
        'motion_threshhold': 0.2
    }
}

num_epochs = 15

eval_epoch_interval = 5

optimizer = {'paramwise_cfg': {'custom_keys': {'img_backbone': {'lr_mult': 0.5}}}}

class_names = {{_base_.perception_class_names}}

NameMapping = {{_base_.perception_name_mapping}}

model = {
    'use_GAN': False,
    'head': {
        'img_feature_res': True,
        'det_head': {
            'instance_bank': {
                'queue_mode': 'temporal',
                'train_at_2hz': True
            }
        },
        'map_head': {
            'instance_bank': {
                'queue_mode': 'temporal',
                'train_at_2hz': True
            }
        },
        'motion_plan_head': None
    }
}

train_pipeline = [
    {{_base_.pipeline_steps.load_multi_view_image_from_files}}, {{_base_.pipeline_steps.resize_crop_flip_image}}, {
        'type': 'BBoxRotation'
    }, {{_base_.pipeline_steps.photo_metric_distortion_multi_view_image}}, {{_base_.pipeline_steps.normalize_multiview_image}}, {
        'type': 'CircleObjectRangeFilter',
        'class_dist_thred': [55, 55, 55, 55, 55, 55, 55, 55, 55, 55]
    }, {
        'type': 'InstanceNameFilter',
        'classes': {{_base_.perception_class_names}}
    }, {
        'type': 'VectorizeMap',
        'roi_size': (30, 60),
        'simplify': False,
        'normalize': False,
        'sample_num': 20,
        'permute': True
    }, {{_base_.pipeline_steps.nu_scenes_sparse4d_adaptor}}, {
        'type': 'Collect',
        'keys': [
            'img', 'timestamp', 'projection_mat', 'image_wh', 'focal', 'gt_bboxes_3d',
            'gt_labels_3d', 'gt_map_labels', 'gt_map_pts', 'gt_agent_fut_trajs',
            'gt_agent_fut_masks', 'gt_ego_fut_trajs', 'gt_ego_fut_masks', 'gt_ego_fut_cmd',
            'ego_status', 'tp_near', 'tp_far'
        ],
        'meta_keys': [
            'scenario_name', 'T_global', 'T_global_inv', 'timestamp', 'instance_id', 'lidar2img'
        ]
    }
]

test_pipeline = [
    {{_base_.pipeline_steps.load_multi_view_image_from_files}}, {{_base_.pipeline_steps.resize_crop_flip_image}}, {{_base_.pipeline_steps.normalize_multiview_image}},
    {{_base_.pipeline_steps.nu_scenes_sparse4d_adaptor}}, {
        'type': 'Collect',
        'keys': [
            'img', 'timestamp', 'projection_mat', 'image_wh', 'ego_status', 'gt_ego_fut_cmd',
            'gt_ego_fut_trajs'
        ],
        'meta_keys': ['scenario_name', 'T_global', 'T_global_inv', 'timestamp', 'lidar2img']
    }
]

eval_config = {
    'type': 'B2D3DDataset',
    'data_root': {{_base_.data_root}},
    'classes': {{_base_.perception_class_names}},
    'map_classes': {{_base_.map_class_names}},
    'name_mapping': {{_base_.perception_name_mapping}},
    'modality': {{_base_.input_modality}},
    'sample_interval': 5,
    'past_frames': 2,
    'future_frames': 6,
    'use_cmd': True,
    'ann_file': {{_base_.val_ann_file}},
    'pipeline': [{
        'type': 'CircleObjectRangeFilter',
        'class_dist_thred': [55, 55, 55, 55, 55, 55, 55, 55, 55, 55]
    }, {
        'type': 'InstanceNameFilter',
        'classes': {{_base_.perception_class_names}}
    }, {
        'type': 'VectorizeMap',
        'roi_size': (30, 60),
        'simplify': True,
        'normalize': False
    }, {
        'type': 'Collect',
        'keys': [
            'vectors', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_agent_fut_trajs', 'gt_agent_fut_masks',
            'gt_ego_fut_trajs', 'gt_ego_fut_masks', 'gt_ego_fut_cmd'
        ],
        'meta_keys': ['scenario_name', 'token', 'timestamp', 'lidar2img']
    }],
    'test_mode': True,
    'debug_mode': False
}

data = {
    'samples_per_gpu': {{_base_.batch_size}},
    'workers_per_gpu': 8,
    'train': {
        'type': 'B2D3DDataset',
        'data_root': {{_base_.data_root}},
        'classes': {{_base_.perception_class_names}},
        'map_classes': {{_base_.map_class_names}},
        'name_mapping': {{_base_.perception_name_mapping}},
        'modality': {{_base_.input_modality}},
        'sample_interval': 5,
        'past_frames': 2,
        'future_frames': 6,
        'use_cmd': True,
        'ann_file': {{_base_.train_ann_file}},
        'pipeline': train_pipeline,
        'test_mode': False,
        'data_aug_conf': {{_base_.data_aug_conf}},
        'with_seq_flag': True,
        'sequences_split_num': 2,
        'keep_consistent_seq_aug': True,
        'debug_mode': False
    },
    'val': {
        'type': 'B2D3DDataset',
        'data_root': {{_base_.data_root}},
        'classes': {{_base_.perception_class_names}},
        'map_classes': {{_base_.map_class_names}},
        'name_mapping': {{_base_.perception_name_mapping}},
        'modality': {{_base_.input_modality}},
        'sample_interval': 5,
        'past_frames': 2,
        'future_frames': 6,
        'use_cmd': True,
        'ann_file': {{_base_.val_ann_file}},
        'pipeline': test_pipeline,
        'data_aug_conf': {{_base_.data_aug_conf}},
        'test_mode': True,
        'eval_config': eval_config,
        'debug_mode': False
    },
    'test': {
        'type': 'B2D3DDataset',
        'data_root': {{_base_.data_root}},
        'classes': {{_base_.perception_class_names}},
        'map_classes': {{_base_.map_class_names}},
        'name_mapping': {{_base_.perception_name_mapping}},
        'modality': {{_base_.input_modality}},
        'sample_interval': 5,
        'past_frames': 2,
        'future_frames': 6,
        'use_cmd': True,
        'ann_file': {{_base_.val_ann_file}},
        'pipeline': test_pipeline,
        'data_aug_conf': {{_base_.data_aug_conf}},
        'test_mode': True,
        'eval_config': eval_config,
        'debug_mode': False
    }
}
