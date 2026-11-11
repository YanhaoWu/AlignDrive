"""Final training-stage settings; architecture is shared with inference."""

import os

_base_ = [
    '../_base_/aligndrive.py', '../_base_/bench2drive.py', '../_base_/runtime.py',
    '../_base_/training.py'
]

work_dir = './work_dirs/aligndrive_stage_lon'

load_from = os.environ.get('ALIGNDRIVE_SPATIAL_CHECKPOINT', './ckpt/aligndrive_spatial_last.pth')

load_from_modules = None

find_unused_parameters = False

evaluation = {
    'interval': None,
    'eval_mode': {
        'with_det': False,
        'with_tracking': False,
        'with_map': False,
        'with_motion': False,
        'with_planning': True,
        'tracking_threshold': 0.2,
        'motion_threshhold': 0.2
    },
    'metric': 'bbox',
    'save_best': 'L2_final',
    'rule': 'less'
}

enable_online_evaluation = True

eval_epoch_interval = 2

model = {
    'head': {
        'longitudinal_head': {{_base_.model.head.longitudinal_head}},
        'motion_plan_head': {
            'final_used_head': 'longitudinal',
            'other_output_pretext': ['longitudinal'],
            'collision_loss': {
                'reduction': 'sample_mean'
            },
            'conditional_loss_weighting': {
                'spatial': {
                    'multiplier': 1.2,
                    'scenarios': ['ParkingExit', 'VehicleOpensDoorTwoWays']
                },
                'longitudinal': {
                    'multiplier': 1.2,
                    'scenarios': [
                        'SignalizedJunctionRightTurn', 'NonSignalizedJunctionRightTurn',
                        'SignalizedJunctionLeftTurn', 'NonSignalizedJunctionLeftTurn'
                    ],
                    'command_ids': [0, 1]
                },
                'collision': {
                    'multiplier': 1.0,
                    'scenarios': [],
                    'command_ids': [0, 1]
                }
            }
        }
    }
}

train_pipeline = [
    {{_base_.pipeline_steps.load_multi_view_image_from_files}}, {{_base_.pipeline_steps.resize_crop_flip_image}}, {{_base_.pipeline_steps.photo_metric_distortion_multi_view_image}}, {{_base_.pipeline_steps.normalize_multiview_image}},
    {
        'type': 'CircleObjectRangeFilter',
        'class_dist_thred': [55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55]
    }, {
        'type': 'InstanceNameFilter',
        'classes': {{_base_.class_names}}
    }, {
        'type': 'VectorizeMap',
        'roi_size': (30, 60),
        'simplify': False,
        'normalize': False,
        'sample_num': 20,
        'permute': True
    }, {
        'type': 'CommandNoise',
        'noise_pro': 0.5,
        'keys': 'gt_ego_fut_cmd',
        'num_cmd': 6,
        'noise_command': [0, 1, 2, 3, 4, 5]
    }, {
        'type': 'Process_EgoStatus',
        'noise_pro': 2,
        'num_bins': 1024,
        'v_max': 20,
        'used_attri': ['speed_0']
    }, {
        'type': 'Build_Navaigation',
        'gloalpoint_extend': True,
        'noise_pro': 1.0,
        'noise_scale': 0.25
    }, {{_base_.pipeline_steps.nu_scenes_sparse4d_adaptor}}, {
        'type': 'Collect',
        'keys': [
            'img', 'timestamp', 'projection_mat', 'image_wh', 'focal', 'gt_bboxes_3d',
            'gt_labels_3d', 'gt_map_labels', 'gt_map_pts', 'gt_agent_fut_trajs',
            'gt_agent_fut_masks', 'gt_ego_fut_trajs', 'gt_ego_fut_masks', 'gt_ego_fut_cmd',
            'ego_status', 'ego_status_digit', 'ego_speed', 'tp_near', 'tp_far', 'tp_line',
            'spatial_interval_2_traj', 'spatial_interval_5_traj', 'temporal_interval_2_traj',
            'temporal_interval_5_traj', 'spatial_interval_2_mask', 'spatial_interval_5_mask',
            'temporal_interval_2_mask', 'temporal_interval_5_mask', 'fake_bboxes_3d',
            'fake_agent_fut_trajs', 'fake_bboxes_3d_5hz', 'fake_agent_fut_trajs_5hz',
            'fake_defensive_ego_fut_trajs', 'fake_aggressive_ego_fut_trajs',
            'fake_defensive_ego_fut_trajs_5hz', 'fake_aggressive_ego_fut_trajs_5hz', 'valid_flag',
            'aggressive_valid'
        ],
        'meta_keys': [
            'scenario_name', 'T_global', 'T_global_inv', 'timestamp', 'instance_id', 'lidar2img'
        ]
    }
]

test_pipeline = [
    {{_base_.pipeline_steps.load_multi_view_image_from_files}}, {{_base_.pipeline_steps.resize_crop_flip_image}}, {{_base_.pipeline_steps.normalize_multiview_image}}, {
        'type': 'CommandNoise',
        'noise_pro': 0,
        'keys': 'gt_ego_fut_cmd',
        'num_cmd': 6
    }, {
        'type': 'Process_EgoStatus',
        'noise_pro': 0,
        'num_bins': 1024,
        'v_max': 20,
        'used_attri': ['speed_0']
    }, {
        'type': 'Build_Navaigation',
        'gloalpoint_extend': True,
        'noise_pro': 0.0,
        'noise_scale': 0.0
    }, {{_base_.pipeline_steps.nu_scenes_sparse4d_adaptor}}, {
        'type': 'Collect',
        'keys': [
            'img', 'timestamp', 'projection_mat', 'image_wh', 'ego_status', 'ego_status_digit',
            'ego_speed', 'gt_ego_fut_cmd', 'gt_ego_fut_trajs', 'tp_near', 'tp_far', 'tp_line',
            'spatial_interval_2_traj', 'spatial_interval_5_traj', 'temporal_interval_2_traj',
            'temporal_interval_5_traj', 'spatial_interval_2_mask', 'spatial_interval_5_mask',
            'temporal_interval_2_mask', 'temporal_interval_5_mask'
        ],
        'meta_keys': ['scenario_name', 'T_global', 'T_global_inv', 'timestamp', 'lidar2img']
    }
]

eval_config = {
    'type': 'B2D3DDataset',
    'data_root': {{_base_.data_root}},
    'classes': {{_base_.class_names}},
    'map_classes': {{_base_.map_class_names}},
    'name_mapping': {{_base_.NameMapping}},
    'modality': {{_base_.input_modality}},
    'sample_interval': 5,
    'past_frames': 2,
    'future_frames': 6,
    'use_cmd': True,
    'gloalpoint_extend': True,
    'ann_file': {{_base_.val_ann_file}},
    'pipeline': [{
        'type': 'CircleObjectRangeFilter',
        'class_dist_thred': [55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55]
    }, {
        'type': 'InstanceNameFilter',
        'classes': {{_base_.class_names}}
    }, {
        'type': 'VectorizeMap',
        'roi_size': (30, 60),
        'simplify': True,
        'normalize': False
    }, {
        'type': 'CommandNoise',
        'noise_pro': 0,
        'keys': 'gt_ego_fut_cmd',
        'num_cmd': 6
    }, {
        'type': 'Process_EgoStatus',
        'noise_pro': 0,
        'num_bins': 1024,
        'v_max': 20,
        'used_attri': ['speed_0']
    }, {
        'type': 'Build_Navaigation',
        'gloalpoint_extend': True,
        'noise_pro': 0.0,
        'noise_scale': 0.0
    }, {
        'type': 'Collect',
        'keys': [
            'vectors', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_agent_fut_trajs', 'gt_agent_fut_masks',
            'gt_ego_fut_trajs', 'gt_ego_fut_masks', 'gt_ego_fut_cmd', 'ego_status_digit',
            'ego_speed', 'ego_status', 'tp_near', 'tp_far', 'tp_line', 'spatial_interval_2_traj',
            'spatial_interval_5_traj', 'temporal_interval_2_traj', 'temporal_interval_5_traj',
            'spatial_interval_2_mask', 'spatial_interval_5_mask', 'temporal_interval_2_mask',
            'temporal_interval_5_mask'
        ],
        'meta_keys': ['scenario_name', 'token', 'timestamp', 'lidar2img']
    }],
    'test_mode': True,
    'debug_mode': False,
    'waypoints_file': {{_base_.val_waypoints_file}}
}

data = {
    'samples_per_gpu': {{_base_.batch_size}},
    'workers_per_gpu': 8,
    'train': {
        'type': 'B2D3DDataset',
        'data_root': {{_base_.data_root}},
        'classes': {{_base_.class_names}},
        'map_classes': {{_base_.map_class_names}},
        'name_mapping': {{_base_.NameMapping}},
        'modality': {{_base_.input_modality}},
        'sample_interval': 5,
        'past_frames': 2,
        'future_frames': 6,
        'use_cmd': True,
        'gloalpoint_extend': True,
        'ann_file': {{_base_.train_ann_file}},
        'waypoints_file': {{_base_.train_waypoints_file}},
        'pipeline': train_pipeline,
        'test_mode': False,
        'data_aug_conf': {{_base_.data_aug_conf}},
        'with_seq_flag': True,
        'sequences_split_num': 2,
        'keep_consistent_seq_aug': True,
        'debug_mode': False,
        'resample_split': None,
        'augment_bbox': True
    },
    'val': {
        'type': 'B2D3DDataset',
        'data_root': {{_base_.data_root}},
        'classes': {{_base_.class_names}},
        'map_classes': {{_base_.map_class_names}},
        'name_mapping': {{_base_.NameMapping}},
        'modality': {{_base_.input_modality}},
        'sample_interval': 5,
        'past_frames': 2,
        'future_frames': 6,
        'use_cmd': True,
        'gloalpoint_extend': True,
        'ann_file': {{_base_.val_ann_file}},
        'waypoints_file': {{_base_.val_waypoints_file}},
        'pipeline': test_pipeline,
        'data_aug_conf': {{_base_.data_aug_conf}},
        'test_mode': True,
        'eval_config': eval_config,
        'debug_mode': False,
        'augment_bbox': False
    },
    'test': {
        'type': 'B2D3DDataset',
        'data_root': {{_base_.data_root}},
        'classes': {{_base_.class_names}},
        'map_classes': {{_base_.map_class_names}},
        'name_mapping': {{_base_.NameMapping}},
        'modality': {{_base_.input_modality}},
        'sample_interval': 5,
        'past_frames': 2,
        'future_frames': 6,
        'use_cmd': True,
        'gloalpoint_extend': True,
        'ann_file': {{_base_.val_ann_file}},
        'waypoints_file': {{_base_.val_waypoints_file}},
        'pipeline': test_pipeline,
        'data_aug_conf': {{_base_.data_aug_conf}},
        'test_mode': True,
        'eval_config': eval_config,
        'debug_mode': False
    }
}
