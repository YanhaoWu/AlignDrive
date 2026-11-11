"""Shared perception architecture used by perception training and AlignDrive."""

import os

map_anchor_path = os.path.join(os.environ.get('ALIGNDRIVE_ANCHOR_ROOT', './kmeans/b2d'),
                               'kmeans_map_100.npy')

box_anchor_path = os.path.join(os.environ.get('ALIGNDRIVE_ANCHOR_ROOT', './kmeans/b2d'),
                               'kmeans_det_900.npy')

plan_anchor_path = os.path.join(os.environ.get('ALIGNDRIVE_ANCHOR_ROOT', './kmeans/b2d'),
                                'train_mutil_granular_waypoint_cluster_refine_1.pkl')

resnet_checkpoint = os.environ.get('ALIGNDRIVE_RESNET_CHECKPOINT', './ckpt/resnet50-19c8e357.pth')

model = {
    'type': 'AlignDrive',
    'use_grid_mask': True,
    'use_deformable_func': True,
    'img_backbone': {
        'type': 'ResNet',
        'depth': 50,
        'num_stages': 4,
        'frozen_stages': -1,
        'norm_eval': False,
        'style': 'pytorch',
        'with_cp': True,
        'out_indices': (0, 1, 2, 3),
        'norm_cfg': {
            'type': 'BN',
            'requires_grad': True
        },
        'pretrained': resnet_checkpoint
    },
    'img_neck': {
        'type': 'PAFPN',
        'num_outs': 4,
        'start_level': 0,
        'out_channels': 256,
        'add_extra_convs': 'on_output',
        'relu_before_extra_convs': True,
        'in_channels': [256, 512, 1024, 2048]
    },
    'head': {
        'type': 'AlignDriveHead',
        'task_config': {
            'with_det': True,
            'with_map': True,
            'with_motion_plan': False
        },
        'det_head': {
            'type': 'Sparse4DHead',
            'cls_threshold_to_reg': 0.05,
            'decouple_attn': True,
            'instance_bank': {
                'type': 'InstanceBank',
                'num_anchor': 900,
                'embed_dims': 256,
                'anchor': box_anchor_path,
                'anchor_handler': {
                    'type': 'SparseBox3DKeyPointsGenerator'
                },
                'num_temp_instances': 600,
                'confidence_decay': 0.6,
                'feat_grad': False,
                'queue_length': 5
            },
            'anchor_encoder': {
                'type': 'SparseBox3DEncoder',
                'vel_dims': 3,
                'embed_dims': [128, 32, 32, 64],
                'mode': 'cat',
                'output_fc': False,
                'in_loops': 1,
                'out_loops': 4
            },
            'num_single_frame_decoder': 1,
            'operation_order': [
                'deformable', 'ffn', 'norm', 'refine', 'temp_gnn', 'gnn', 'norm', 'deformable',
                'ffn', 'norm', 'refine', 'temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm',
                'refine', 'temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine',
                'temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine', 'temp_gnn', 'gnn',
                'norm', 'deformable', 'ffn', 'norm', 'refine'
            ],
            'temp_graph_model': {
                'type': 'MultiheadFlashAttention',
                'embed_dims': 512,
                'num_heads': 8,
                'batch_first': True,
                'dropout': 0.1
            },
            'graph_model': {
                'type': 'MultiheadFlashAttention',
                'embed_dims': 512,
                'num_heads': 8,
                'batch_first': True,
                'dropout': 0.1
            },
            'norm_layer': {
                'type': 'LN',
                'normalized_shape': 256
            },
            'ffn': {
                'type': 'AsymmetricFFN',
                'in_channels': 512,
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
            'deformable_model': {
                'type': 'DeformableFeatureAggregation',
                'embed_dims': 256,
                'num_groups': 8,
                'num_levels': 4,
                'num_cams': 6,
                'attn_drop': 0.15,
                'use_deformable_func': True,
                'use_camera_embed': True,
                'residual_mode': 'cat',
                'kps_generator': {
                    'type': 'SparseBox3DKeyPointsGenerator',
                    'num_learnable_pts': 6,
                    'fix_scale': [[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0], [0, 0.45, 0],
                                  [0, -0.45, 0], [0, 0, 0.45], [0, 0, -0.45]]
                }
            },
            'refine_layer': {
                'type': 'SparseBox3DRefinementModule',
                'embed_dims': 256,
                'num_cls': 10,
                'refine_yaw': True,
                'with_quality_estimation': True
            },
            'sampler': {
                'type': 'SparseBox3DTarget',
                'num_dn_groups': 0,
                'num_temp_dn_groups': 0,
                'dn_noise_scale': [2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                'max_dn_gt': 32,
                'add_neg_dn': True,
                'cls_weight': 2.0,
                'box_weight': 0.25,
                'reg_weights': [2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                'cls_wise_reg_weights': {
                    5: [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
                }
            },
            'loss_cls': {
                'type': 'FocalLoss',
                'use_sigmoid': True,
                'gamma': 2.0,
                'alpha': 0.25,
                'loss_weight': 2.0
            },
            'loss_reg': {
                'type': 'SparseBox3DLoss',
                'loss_box': {
                    'type': 'L1Loss',
                    'loss_weight': 0.25
                },
                'loss_centerness': {
                    'type': 'CrossEntropyLoss',
                    'use_sigmoid': True
                },
                'loss_yawness': {
                    'type': 'GaussianFocalLoss'
                }
            },
            'decoder': {
                'type': 'SparseBox3DDecoder'
            },
            'reg_weights': [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        },
        'map_head': {
            'type': 'Sparse4DHead',
            'cls_threshold_to_reg': 0.05,
            'decouple_attn': False,
            'instance_bank': {
                'type': 'InstanceBank',
                'num_anchor': 100,
                'embed_dims': 256,
                'anchor': map_anchor_path,
                'anchor_handler': {
                    'type': 'SparsePoint3DKeyPointsGenerator'
                },
                'num_temp_instances': 0,
                'confidence_decay': 0.9,
                'feat_grad': True,
                'queue_length': 5
            },
            'anchor_encoder': {
                'type': 'SparsePoint3DEncoder',
                'embed_dims': 256,
                'num_sample': 20
            },
            'num_single_frame_decoder': 1,
            'operation_order': [
                'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine', 'temp_gnn', 'gnn', 'norm',
                'deformable', 'ffn', 'norm', 'refine', 'temp_gnn', 'gnn', 'norm', 'deformable',
                'ffn', 'norm', 'refine', 'temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm',
                'refine', 'temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine',
                'temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine'
            ],
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
            'ffn': {
                'type': 'AsymmetricFFN',
                'in_channels': 512,
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
            'deformable_model': {
                'type': 'DeformableFeatureAggregation',
                'embed_dims': 256,
                'num_groups': 8,
                'num_levels': 4,
                'num_cams': 6,
                'attn_drop': 0.15,
                'use_deformable_func': True,
                'use_camera_embed': True,
                'residual_mode': 'cat',
                'kps_generator': {
                    'type': 'SparsePoint3DKeyPointsGenerator',
                    'embed_dims': 256,
                    'num_sample': 20,
                    'num_learnable_pts': 3,
                    'fix_height': (0, 0.5, -0.5, 1, -1),
                    'ground_height': -1.84023
                }
            },
            'refine_layer': {
                'type': 'SparsePoint3DRefinementModule',
                'embed_dims': 256,
                'num_sample': 20,
                'num_cls': 6
            },
            'sampler': {
                'type': 'SparsePoint3DTarget',
                'assigner': {
                    'type': 'HungarianLinesAssigner',
                    'cost': {
                        'type': 'MapQueriesCost',
                        'cls_cost': {
                            'type': 'FocalLossCost',
                            'weight': 1.0
                        },
                        'reg_cost': {
                            'type': 'LinesL1Cost',
                            'weight': 10.0,
                            'beta': 0.01,
                            'permute': True
                        }
                    }
                },
                'num_cls': 6,
                'num_sample': 20,
                'roi_size': (30, 60)
            },
            'loss_cls': {
                'type': 'FocalLoss',
                'use_sigmoid': True,
                'gamma': 2.0,
                'alpha': 0.25,
                'loss_weight': 1.0
            },
            'loss_reg': {
                'type': 'SparseLineLoss',
                'loss_line': {
                    'type': 'LinesL1Loss',
                    'loss_weight': 10.0,
                    'beta': 0.01
                },
                'num_sample': 20,
                'roi_size': (30, 60)
            },
            'decoder': {
                'type': 'SparsePoint3DDecoder'
            },
            'reg_weights': [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0
            ],
            'gt_cls_key': 'gt_map_labels',
            'gt_reg_key': 'gt_map_pts',
            'gt_id_key': 'map_instance_id',
            'with_instance_id': False,
            'task_prefix': 'map'
        }
    }
}
