"""Complete AlignDrive planning model, extending the shared perception modules."""

_base_ = ['perception.py']

model = {
    'training_stage': 'normal',
    'head': {
        'type': 'UnifiedHead',
        'task_config': {
            'with_motion_plan': True,
            'with_motion': True
        },
        'temporal_queue': {
            'temporal_img': -1,
            'temporal_gap': 3
        },
        'num_decoder': 6,
        'dual_plan_head': None,
        'motion_plan_head': {
            'type': 'LongitudinalPlanningModule',
            'cls_threshold_to_reg': 0.05,
            'auxiliary_loss': {
                'dynamic_loss': False,
                'dynamic_loss_longitudinal': True,
                'dynamic_horzion': 10,
                'dynamic_loss_scale': 0.1,
                'dynamic_dead_range': 0.5,
                'dynamic_all_trajs': False,
                'dynamic_loss_other_trajs_scale': 0.2,
                'lateral_loss': False,
                'lateral_loss_longitudinal': False,
                'lateral_loss_scale': 0.2,
                'start_iter': 'auto_half_training'
            },
            'command_res': False,
            'plan_control': 'open_loop',
            'anchor_supervise': False,
            'decouple_attn': False,
            'gloalpoint_extend': True,
            'force_check_timeinterval': True,
            'gt_modal': 'distance',
            'final_used_head': 'longitudinal',
            'longitudinal_train_use_grand': {
                'temporal': 'temporal_interval_2_traj',
                'spatial': 'spatial_interval_2_traj',
                'augmentation': 'fake_defensive_ego_fut_trajs_5hz',
                'augmented_bbox': 'fake_bboxes_3d_5hz',
                'augmented_bbox_trajs': 'fake_agent_fut_trajs_5hz',
                'interval': 2,
                'num_horzion': 15,
                'instance_feature': True,
                'confidence_thrhold': 0.3,
                'start_iter': 'auto_half_training'
            },
            'augment_bbox': False,
            'augment_bbox_longitudinal': True,
            'instance_bank': {
                'type': 'InstanceBankPlan',
                'num_anchor': 24,
                'embed_dims': 256,
                'anchor': {{_base_.plan_anchor_path}},
                'anchor_mearge': True,
                'anchor_handler': {
                    'type': 'SparsePoint3DKeyPointsGenerator'
                },
                'num_temp_instances': 24,
                'confidence_decay': 0.6,
                'feat_grad': True,
                'project_temporal_anchor': False,
                'queue_length': 50,
                'temporal_gap': 0.0,
                'used_grand': [
                    'spatial_interval_2_traj', 'spatial_interval_5_traj',
                    'temporal_interval_2_traj', 'temporal_interval_5_traj'
                ]
            },
            'anchor_encoder': {
                'type': 'SparsePoint3DEncoder',
                'embed_dims': 256,
                'num_sample': 15
            },
            'anchor_encoder_low_frency': {
                'type': 'SparsePoint3DEncoder',
                'embed_dims': 256,
                'num_sample': 6
            },
            'num_single_frame_decoder': 1,
            'temp_graph_model': {
                'type': 'MultiheadFlashAttention',
                'embed_dims': 256,
                'num_heads': 8,
                'batch_first': True,
                'dropout': 0.1
            },
            'graph_model': {
                'type': 'MultiheadFlashAttention',
                'embed_dims': 256,
                'num_heads': 8,
                'batch_first': True,
                'dropout': 0.1
            },
            'norm_layer': {
                'type': 'LN',
                'normalized_shape': 256
            },
            'img_feature_extrctor': {
                'type': 'GlobalFeatureExtractor',
                'embed_dims': 256,
                'feature_map_index': [-1],
                'add_3dpositional_encoding': True
            },
            'deformable_model': {
                'type': 'DeformableFeatureAggregation',
                'embed_dims': 256,
                'num_groups': 8,
                'num_levels': 4,
                'num_cams': 6,
                'attn_drop': 0.15,
                'use_deformable_func': True,
                'use_camera_embed': True,
                'residual_mode': 'add',
                'kps_generator': {
                    'type': 'SparsePoint3DKeyPointsGenerator',
                    'embed_dims': 256,
                    'num_sample': 15,
                    'num_learnable_pts': 3,
                    'fix_height': (0, 0.5, -0.5, 1, -1),
                    'ground_height': -1.84023
                }
            },
            'spdeformable_model': None,
            'objectdeformable_model': {
                'type': 'ObjectDFA',
                'embed_dims': 256,
                'num_groups': 8,
                'num_levels': 4,
                'num_cams': 6,
                'attn_drop': 0.15,
                'use_deformable_func': False,
                'use_camera_embed': True,
                'residual_mode': 'add',
                'kps_generator': None,
                'num_pts': 195
            },
            'sampler': {
                'type': 'MultiGranularityPlanningTarget',
                'ego_fut_ts': 15,
                'ego_fut_mode': 6,
                'plan_vocab_path': {{_base_.plan_anchor_path}},
                'used_grand': [
                    'spatial_interval_2_traj', 'spatial_interval_5_traj',
                    'temporal_interval_2_traj', 'temporal_interval_5_traj'
                ],
                'SpeedAwareRange': [[0.1, 0.4], [0.4, 3.0], [3.0, 5.0], [5.0, 7.0], [7.0, 10.0]],
                'rand_grand_pro': 0,
                'num_drivestyle': 0
            },
            'navigation_module': {
                'type': 'NavigationAware',
                'mode': 'sum'
            },
            'decoder': {
                'type': 'MaplikePlanDecoder'
            },
            'reg_weights': [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
            ],
            'gt_cls_key': 'gt_ego_fut_cmd',
            'gt_reg_key': 'gt_ego_fut_trajs',
            'gt_id_key': 'map_instance_id',
            'with_instance_id': False,
            'task_prefix': 'plan',
            'plan_loss_cls': {
                'type': 'FocalLoss',
                'use_sigmoid': True,
                'gamma': 2.0,
                'alpha': 0.25,
                'loss_weight': 2.0
            },
            'plan_loss_reg': {
                'type': 'PlanningRegLoss',
                'loss_weight': 1.0,
                'base_loss': {
                    'type': 'L1Loss',
                    'loss_weight': 1.0
                },
                'temporal_weight': [
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4
                ],
                'balance_setting': {
                    'spatial_interval_2_traj': {
                        'temporal_weighted': [
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4,
                            0.4
                        ],
                        'total_weighted': 3.0
                    },
                    'spatial_interval_5_traj': {
                        'temporal_weighted': [
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0
                        ],
                        'total_weighted': 1.0
                    },
                    'temporal_interval_2_traj': {
                        'temporal_weighted': [
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4,
                            0.4
                        ],
                        'total_weighted': 3.0
                    },
                    'temporal_interval_5_traj': {
                        'temporal_weighted': [
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0
                        ],
                        'total_weighted': 1.0
                    },
                    'command': {
                        '0': 1.0,
                        '1': 1.0,
                        '2': 1.0,
                        '3': 1.0,
                        '4': 1.0,
                        '5': 1.0,
                        '6': 1.0
                    },
                    'command_scale_iter_map': 'auto_half_training',
                    'command_scale_iter_plan': 'auto_half_training'
                }
            },
            'collision_loss': {
                'type': 'PlanningCollisionLoss',
                'weights': 1.0
            },
            'plan_loss_score': {
                'type': 'L1Loss',
                'loss_weight': 1.0
            },
            'keypoint_loss': {
                'type': 'MSELoss',
                'loss_weight': 1.0
            },
            'plan_loss_status': {
                'type': 'L1Loss',
                'loss_weight': 0.2
            },
            'longitudinal_head_scale': 1.0,
            'num_cmd': 7,
            'num_sample_waypoints': 15,
            'temporal_img': -1,
            'use_GAN': False,
            'temporal_gap': 3,
            'use_perception_query': True,
            'use_tp': True,
            'use_command': True,
            'collision_check': False,
            'anchor_mearge': True,
            'num_temporal_plan': 50,
            'operation_order': [
                ['add_navigation', 'norm', 'deformable', 'ffn', 'norm', 'refine'],
                ['temp_gnn', 'add_navigation', 'norm', 'deformable', 'ffn', 'norm', 'refine'],
                ['temp_gnn', 'add_navigation', 'norm', 'deformable', 'ffn', 'norm', 'refine'],
                ['temp_gnn', 'add_navigation', 'norm', 'deformable', 'ffn', 'norm', 'refine'],
                ['temp_gnn', 'add_navigation', 'norm', 'deformable', 'ffn', 'norm', 'refine'],
                ['temp_gnn', 'add_navigation', 'norm', 'deformable', 'ffn', 'norm', 'refine']
            ],
            'ffn': {
                'type': 'AsymmetricFFN',
                'in_channels': 256,
                'pre_norm': {
                    'type': 'LN'
                },
                'embed_dims': 256,
                'feedforward_channels': 1024,
                'num_fcs': 2,
                'ffn_drop': 0.1,
                'act_cfg': {
                    'type': 'ReLU',
                    'inplace': True
                }
            },
            'use_ego_status': False,
            'tp_res': False,
            'refine_layer': {
                'type': 'MaplikePlanQueryDecoder',
                'embed_dims': 256,
                'num_sample': 15,
                'num_cls': 1,
                'offset_learn': True,
                'with_status_branch': True,
                'num_drivestyle': 0,
                'num_granularity': 4,
                'num_modal': 6,
                'mutil_head': True
            },
            'other_output_pretext': ['longitudinal'],
            'conditional_loss_weighting': {}
        },
        'num_key_object': 15,
        'query_drop_rate': 0.0,
        'fusion_model': {
            'type': 'QueryFusion',
            'distance_attn_model': {
                'type': 'MultiheadFlashAttention',
                'embed_dims': 256,
                'num_heads': 8,
                'batch_first': True,
                'dropout': 0.1,
                'distance_attention': True
            },
            'graph_model': {
                'type': 'MultiheadFlashAttention',
                'embed_dims': 256,
                'num_heads': 8,
                'batch_first': True,
                'dropout': 0.1
            },
            'plan_mearge_model': {
                'type': 'MaplikePlanQueryDecoder',
                'embed_dims': 256,
                'num_sample': 15,
                'num_cls': 1,
                'offset_learn': True,
                'with_status_branch': False
            },
            'agent_anchor_encoder': {
                'type': 'SparsePoint3DEncoder',
                'embed_dims': 256,
                'num_sample': 1,
                'coords_dim': 11
            },
            'map_anchor_encoder': {
                'type': 'SparsePoint3DEncoder',
                'embed_dims': 256,
                'num_sample': 20,
                'coords_dim': 2
            },
            'norm_layer': {
                'type': 'LN',
                'normalized_shape': 256
            },
            'operation_order': [[],
                                [
                                    'temp_cross_gnn', 'temp_cross_norm', 'cross_gnn', 'cross_norm',
                                    'agent_cross_map', 'agent_cross_map_norm'
                                ],
                                [
                                    'temp_cross_gnn', 'temp_cross_norm', 'cross_gnn', 'cross_norm',
                                    'agent_cross_map', 'agent_cross_map_norm'
                                ],
                                [
                                    'temp_cross_gnn', 'temp_cross_norm', 'cross_gnn', 'cross_norm',
                                    'agent_cross_map', 'agent_cross_map_norm'
                                ],
                                [
                                    'temp_cross_gnn', 'temp_cross_norm', 'cross_gnn', 'cross_norm',
                                    'agent_cross_map', 'agent_cross_map_norm'
                                ],
                                [
                                    'temp_cross_gnn', 'temp_cross_norm', 'cross_gnn', 'cross_norm',
                                    'agent_cross_map', 'agent_cross_map_norm'
                                ]],
            'merage_operation_order': None,
            'num_query': 1024,
            'embed_dims': 256,
            'motion_queue': None,
            'decode_perception_query': False,
            'augment_bbox_pro': 0.0,
            'plan_num_det': 50,
            'plan_num_map': 25
        },
        'det_head': {
            'num_decoder': 6,
            'instance_bank': {
                'queue_length': 1,
                'num_queue_instance': 50,
                'temp_gap': 1
            },
            'operation_order': [['deformable', 'ffn', 'norm', 'refine'],
                                ['temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine'],
                                ['temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine'],
                                ['temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine'],
                                ['temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine'],
                                ['temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine']],
            'refine_layer': {
                'num_cls': 12,
                'with_traj': True,
                'with_dp': False
            },
            'motion_sampler': {
                'type': 'MotionTarget'
            },
            'motion_loss_reg': {
                'type': 'L1Loss',
                'loss_weight': 0.2
            },
            'class_weight': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.2, 1.0, 1.0, 1.0]
        },
        'map_head': {
            'instance_bank': {
                'queue_length': 1,
                'num_queue_instance': 25,
                'temp_gap': 1
            },
            'operation_order': [['gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine'],
                                ['temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine'],
                                ['temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine'],
                                ['temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine'],
                                ['temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine'],
                                ['temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine']],
            'loss_reg': {
                'loss_line': {
                    'loss_weight': 25.0
                }
            },
            'temp_graph_model': {
                'no_flash_attn': True
            },
            'graph_model': {
                'no_flash_attn': True
            }
        },
        'longitudinal_head': {
            'type': 'DrivePathPredictor',
            'use_ego_status': False,
            'use_previous_trajectory': True,
            'num_det': 30,
            'num_map': 25,
            'augment_bbox_pro': 0.1,
            'img_encoder': None,
            'img_graph_model': None,
            'use_gt_anchor': False,
            'longitudinal_train_use_grand': {
                'temporal': 'temporal_interval_2_traj',
                'spatial': 'spatial_interval_2_traj',
                'instance_feature': True,
                'augmentation': 'fake_defensive_ego_fut_trajs_5hz',
                'augmented_bbox': 'fake_bboxes_3d_5hz',
                'augmented_bbox_trajs': 'fake_agent_fut_trajs_5hz',
                'interval': 2,
                'num_horzion': 15,
                'confidence_thrhold': 0.3,
                'start_iter': 'auto_half_training'
            },
            'class_names': [
                'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone', 'traffic_light_0',
                'traffic_light_1', 'traffic_light_2', 'pedestrian', 'others', 'traffic.stop'
            ],
            'merage_traffic_light': 'none',
            'instance_bank': {
                'type': 'DrivePathBank',
                'plan_grand': [
                    'spatial_interval_2_traj', 'spatial_interval_5_traj',
                    'temporal_interval_2_traj', 'temporal_interval_5_traj'
                ],
                'num_modal': 6,
                'embed_dims': 256,
                'anchor_grand': None,
                'mutil_step_query': 16,
                'speed_range': [[0.1, 0.4], [0.4, 3.0], [3.0, 5.0], [5.0, 7.0], [7.0, 10.0]],
                'num_ds': 5,
                'prediction_mode': 'distance',
                'timeinterval': 2,
                'drivepath_grand': 'spatial_interval_2_traj'
            },
            'HistoryPlanBank': {
                'type': 'HistoryPlanBank',
                'embed_dims': 256,
                'num_temp_instances': 30,
                'confidence_decay': 0.6,
                'project_temporal_anchor': False,
                'queue_length': 50,
                'temporal_gap': 0.0
            },
            'anchor_encoder': {
                'type': 'SparsePoint3DEncoder',
                'embed_dims': 256,
                'num_sample': 15,
                'coords_dim': 2
            },
            'graph_model': {
                'type': 'MultiheadFlashAttention',
                'embed_dims': 256,
                'num_heads': 8,
                'batch_first': True,
                'dropout': 0.1,
                'distance_attention': True
            },
            'graph_model_no_distance': {
                'type': 'MultiheadFlashAttention',
                'embed_dims': 256,
                'num_heads': 8,
                'batch_first': True,
                'dropout': 0.1,
                'distance_attention': False
            },
            'temp_graph_model': {
                'type': 'BlockAR',
                'embed_dims': 256,
                'num_heads': 8,
                'batch_first': True,
                'dropout': 0.0
            },
            'TemporalMixer': {
                'type': 'TemporalMixer'
            },
            'ffn': {
                'type': 'AsymmetricFFN',
                'in_channels': 256,
                'pre_norm': {
                    'type': 'LN'
                },
                'embed_dims': 256,
                'feedforward_channels': 1024,
                'num_fcs': 2,
                'ffn_drop': 0.1,
                'act_cfg': {
                    'type': 'ReLU',
                    'inplace': True
                }
            },
            'refine_layer': {
                'type': 'SpeedAwarePlanQueryDecoder',
                'embed_dims': 256,
                'num_sample': 15,
                'num_cls': 1,
                'with_status_branch': False,
                'offset_learn': True,
                'num_drivestyle': 0,
                'num_granularity': 5,
                'num_modal': 6,
                'mutil_head': False,
                'mutil_step_query': 16,
                'prediction_mode': 'distance'
            },
            'norm_layer': {
                'type': 'LN',
                'normalized_shape': 256
            },
            'operation_order': [[
                'reference_cross_gnn', 'reference_cross_norm', 'cross_gnn', 'cross_norm',
                'map_cross_gnn', 'cross_norm', 'mutil_step_fusion', 'cross_norm', 'ffn', 'refine'
            ],
                                [
                                    'reference_cross_gnn', 'reference_cross_norm', 'cross_gnn',
                                    'cross_norm', 'map_cross_gnn', 'cross_norm',
                                    'mutil_step_fusion', 'cross_norm', 'ffn', 'refine'
                                ],
                                [
                                    'reference_cross_gnn', 'reference_cross_norm', 'cross_gnn',
                                    'cross_norm', 'map_cross_gnn', 'cross_norm',
                                    'mutil_step_fusion', 'cross_norm', 'ffn', 'refine'
                                ],
                                [
                                    'reference_cross_gnn', 'reference_cross_norm', 'cross_gnn',
                                    'cross_norm', 'map_cross_gnn', 'cross_norm',
                                    'mutil_step_fusion', 'cross_norm', 'ffn', 'refine'
                                ],
                                [
                                    'reference_cross_gnn', 'reference_cross_norm', 'cross_gnn',
                                    'cross_norm', 'map_cross_gnn', 'cross_norm',
                                    'mutil_step_fusion', 'cross_norm', 'ffn', 'refine'
                                ],
                                [
                                    'reference_cross_gnn', 'reference_cross_norm', 'cross_gnn',
                                    'cross_norm', 'map_cross_gnn', 'cross_norm',
                                    'mutil_step_fusion', 'cross_norm', 'ffn', 'refine'
                                ]],
            'embed_dims': 256,
            'p_topk': 1,
            'longitudinal_layer_range': [0, 1, 2, 3, 4, 5],
            'decode_perception_query': True
        }
    }
}
