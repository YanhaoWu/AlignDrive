"""Evaluation settings; architecture is shared with final training."""

_base_ = ['../_base_/aligndrive.py', '../_base_/bench2drive.py', '../_base_/runtime.py']

work_dir = './work_dirs/aligndrive_inference'

num_cmd = 6

evaluation = {
    'interval': 1834,
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

custom_imports = {'imports': [], 'allow_failed_imports': False}

control_setting = {
    'spatial': 'spatial_interval_2_traj',
    'temporal': 'temporal_interval_2_traj',
    'post_process_score': True,
    'perferred_anchor': -1,
    'light_state_enhancement': 3,
    'valid_waypoints': 6
}

strict_checkpoint = True

require_longitudinal_output = True

model = {
    'head': {
        'longitudinal_head': {
            'longitudinal_train_use_grand': {
                'start_iter': 5502
            }
        },
        'motion_plan_head': {
            'auxiliary_loss': {
                'start_iter': 5502
            },
            'plan_control': 'close_loop',
            'longitudinal_train_use_grand': {
                'start_iter': 5502
            },
            'plan_loss_reg': {
                'balance_setting': {
                    'command_scale_iter_map': 5502,
                    'command_scale_iter_plan': 5502
                }
            },
            'control_setting': {
                'spatial': 'spatial_interval_2_traj',
                'temporal': 'temporal_interval_2_traj',
                'post_process_score': True,
                'perferred_anchor': -1,
                'light_state_enhancement': 3,
                'valid_waypoints': 6
            }
        }
    },
    'pretrained': None
}

test_pipeline = [
    {{_base_.pipeline_steps.load_multi_view_image_from_files}}, {{_base_.pipeline_steps.resize_crop_flip_image}}, {{_base_.pipeline_steps.normalize_multiview_image}}, {
        'type': 'Process_EgoStatus',
        'noise_pro': 0,
        'num_bins': 1024,
        'v_max': 20,
        'used_attri': ['speed_0']
    }, {{_base_.pipeline_steps.nu_scenes_sparse4d_adaptor}}, {
        'type': 'Build_Navaigation',
        'gloalpoint_extend': True
    }, {
        'type': 'Collect',
        'keys': [
            'img', 'timestamp', 'projection_mat', 'image_wh', 'ego_status', 'gt_ego_fut_cmd',
            'tp_near', 'tp_far', 'tp_line', 'ego_speed'
        ],
        'meta_keys': ['scenario_name', 'T_global', 'T_global_inv', 'timestamp', 'lidar2img']
    }
]

open_loop_pipeline = [
    {{_base_.pipeline_steps.load_multi_view_image_from_files}}, {{_base_.pipeline_steps.resize_crop_flip_image}}, {{_base_.pipeline_steps.normalize_multiview_image}}, {
        'keys': 'gt_ego_fut_cmd',
        'noise_pro': 0,
        'num_cmd': 6,
        'type': 'CommandNoise'
    }, {
        'noise_pro': 0,
        'num_bins': 1024,
        'type': 'Process_EgoStatus',
        'used_attri': ['speed_0'],
        'v_max': 20
    }, {
        'gloalpoint_extend': True,
        'noise_pro': 0.0,
        'noise_scale': 0.0,
        'type': 'Build_Navaigation'
    }, {{_base_.pipeline_steps.nu_scenes_sparse4d_adaptor}}, {
        'keys': [
            'img', 'timestamp', 'projection_mat', 'image_wh', 'ego_status', 'ego_status_digit',
            'ego_speed', 'gt_ego_fut_cmd', 'gt_ego_fut_trajs', 'tp_near', 'tp_far', 'tp_line',
            'spatial_interval_2_traj', 'spatial_interval_5_traj', 'temporal_interval_2_traj',
            'temporal_interval_5_traj', 'spatial_interval_2_mask', 'spatial_interval_5_mask',
            'temporal_interval_2_mask', 'temporal_interval_5_mask'
        ],
        'meta_keys': ['T_global', 'T_global_inv', 'timestamp', 'lidar2img', 'scenario_name'],
        'type': 'Collect'
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
        'class_dist_thred': [55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55],
        'type': 'CircleObjectRangeFilter'
    }, {
        'classes': {{_base_.class_names}},
        'type': 'InstanceNameFilter'
    }, {
        'normalize': False,
        'roi_size': (30, 60),
        'simplify': True,
        'type': 'VectorizeMap'
    }, {
        'keys': 'gt_ego_fut_cmd',
        'noise_pro': 0,
        'num_cmd': 6,
        'type': 'CommandNoise'
    }, {
        'noise_pro': 0,
        'num_bins': 1024,
        'type': 'Process_EgoStatus',
        'used_attri': ['speed_0'],
        'v_max': 20
    }, {
        'gloalpoint_extend': True,
        'noise_pro': 0.0,
        'noise_scale': 0.0,
        'type': 'Build_Navaigation'
    }, {
        'keys': [
            'vectors', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_agent_fut_trajs', 'gt_agent_fut_masks',
            'gt_ego_fut_trajs', 'gt_ego_fut_masks', 'gt_ego_fut_cmd', 'ego_status_digit',
            'ego_speed', 'ego_status', 'tp_near', 'tp_far', 'tp_line', 'spatial_interval_2_traj',
            'spatial_interval_5_traj', 'temporal_interval_2_traj', 'temporal_interval_5_traj',
            'spatial_interval_2_mask', 'spatial_interval_5_mask', 'temporal_interval_2_mask',
            'temporal_interval_5_mask'
        ],
        'meta_keys': ['token', 'timestamp', 'lidar2img', 'scenario_name'],
        'type': 'Collect'
    }],
    'test_mode': True,
    'debug_mode': False,
    'waypoints_file': {{_base_.val_waypoints_file}}
}

data = {
    'workers_per_gpu': 8,
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
        'pipeline': open_loop_pipeline,
        'data_aug_conf': {{_base_.data_aug_conf}},
        'test_mode': True,
        'eval_config': eval_config,
        'debug_mode': False
    }
}
