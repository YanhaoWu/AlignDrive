map_anchor_path = './kmeans/b2d/kmeans_map_100.npy'
box_anchor_path = './kmeans/b2d/kmeans_det_900.npy'
plan_anchor_path = './kmeans/b2d/train_mutil_granular_waypoint_cluster_refine_1.pkl'
resnet_checkpoint = './ckpt/resnet50-19c8e357.pth'
model = dict(
    type='AlignDrive',
    use_grid_mask=True,
    use_deformable_func=True,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style='pytorch',
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        pretrained='./ckpt/resnet50-19c8e357.pth'),
    img_neck=dict(
        type='FPN',
        num_outs=4,
        start_level=0,
        out_channels=256,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048]),
    head=dict(
        type='UnifiedHead',
        task_config=dict(
            with_det=True,
            with_map=True,
            with_motion_plan=True,
            with_motion=True),
        det_head=dict(
            type='Sparse4DHead',
            cls_threshold_to_reg=0.05,
            decouple_attn=True,
            instance_bank=dict(
                type='InstanceBank',
                num_anchor=900,
                embed_dims=256,
                anchor='./kmeans/b2d/kmeans_det_900.npy',
                anchor_handler=dict(type='SparseBox3DKeyPointsGenerator'),
                num_temp_instances=600,
                confidence_decay=0.6,
                feat_grad=False,
                queue_length=1,
                num_queue_instance=50,
                temp_gap=1),
            anchor_encoder=dict(
                type='SparseBox3DEncoder',
                vel_dims=3,
                embed_dims=[128, 32, 32, 64],
                mode='cat',
                output_fc=False,
                in_loops=1,
                out_loops=4),
            num_single_frame_decoder=1,
            operation_order=[['deformable', 'ffn', 'norm', 'refine'],
                             [
                                 'temp_gnn', 'gnn', 'norm', 'deformable',
                                 'ffn', 'norm', 'refine'
                             ],
                             [
                                 'temp_gnn', 'gnn', 'norm', 'deformable',
                                 'ffn', 'norm', 'refine'
                             ],
                             [
                                 'temp_gnn', 'gnn', 'norm', 'deformable',
                                 'ffn', 'norm', 'refine'
                             ],
                             [
                                 'temp_gnn', 'gnn', 'norm', 'deformable',
                                 'ffn', 'norm', 'refine'
                             ],
                             [
                                 'temp_gnn', 'gnn', 'norm', 'deformable',
                                 'ffn', 'norm', 'refine'
                             ]],
            temp_graph_model=dict(
                type='MultiheadFlashAttention',
                embed_dims=512,
                num_heads=8,
                batch_first=True,
                dropout=0.1),
            graph_model=dict(
                type='MultiheadFlashAttention',
                embed_dims=512,
                num_heads=8,
                batch_first=True,
                dropout=0.1),
            norm_layer=dict(type='LN', normalized_shape=256),
            ffn=dict(
                type='AsymmetricFFN',
                in_channels=512,
                pre_norm=dict(type='LN'),
                embed_dims=256,
                feedforward_channels=1024,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)),
            deformable_model=dict(
                type='DeformableFeatureAggregation',
                embed_dims=256,
                num_groups=8,
                num_levels=4,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=True,
                use_camera_embed=True,
                residual_mode='cat',
                kps_generator=dict(
                    type='SparseBox3DKeyPointsGenerator',
                    num_learnable_pts=6,
                    fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0],
                               [0, 0.45, 0], [0, -0.45, 0], [0, 0, 0.45],
                               [0, 0, -0.45]])),
            refine_layer=dict(
                type='SparseBox3DRefinementModule',
                embed_dims=256,
                num_cls=12,
                refine_yaw=True,
                with_quality_estimation=True,
                with_traj=True,
                with_dp=False),
            sampler=dict(
                type='SparseBox3DTarget',
                num_dn_groups=0,
                num_temp_dn_groups=0,
                dn_noise_scale=[
                    2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                ],
                max_dn_gt=32,
                add_neg_dn=True,
                cls_weight=2.0,
                box_weight=0.25,
                reg_weights=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                cls_wise_reg_weights=dict(
                    {5: [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]})),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            loss_reg=dict(
                type='SparseBox3DLoss',
                loss_box=dict(type='L1Loss', loss_weight=0.25),
                loss_centerness=dict(
                    type='CrossEntropyLoss', use_sigmoid=True),
                loss_yawness=dict(type='GaussianFocalLoss')),
            decoder=dict(type='SparseBox3DDecoder'),
            reg_weights=[2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            num_decoder=6,
            motion_sampler=dict(type='MotionTarget'),
            motion_loss_reg=dict(type='L1Loss', loss_weight=0.2),
            class_weight=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.2, 1.0, 1.0, 1.0
            ]),
        map_head=dict(
            type='Sparse4DHead',
            cls_threshold_to_reg=0.05,
            decouple_attn=False,
            instance_bank=dict(
                type='InstanceBank',
                num_anchor=100,
                embed_dims=256,
                anchor='./kmeans/b2d/kmeans_map_100.npy',
                anchor_handler=dict(type='SparsePoint3DKeyPointsGenerator'),
                num_temp_instances=0,
                confidence_decay=0.9,
                feat_grad=True,
                queue_length=1,
                num_queue_instance=25,
                temp_gap=1),
            anchor_encoder=dict(
                type='SparsePoint3DEncoder', embed_dims=256, num_sample=20),
            num_single_frame_decoder=1,
            operation_order=[[
                'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine'
            ],
                             [
                                 'temp_gnn', 'gnn', 'norm', 'deformable',
                                 'ffn', 'norm', 'refine'
                             ],
                             [
                                 'temp_gnn', 'gnn', 'norm', 'deformable',
                                 'ffn', 'norm', 'refine'
                             ],
                             [
                                 'temp_gnn', 'gnn', 'norm', 'deformable',
                                 'ffn', 'norm', 'refine'
                             ],
                             [
                                 'temp_gnn', 'gnn', 'norm', 'deformable',
                                 'ffn', 'norm', 'refine'
                             ],
                             [
                                 'temp_gnn', 'gnn', 'norm', 'deformable',
                                 'ffn', 'norm', 'refine'
                             ]],
            temp_graph_model=None,
            graph_model=dict(
                type='MultiheadFlashAttention',
                embed_dims=256,
                num_heads=8,
                batch_first=True,
                dropout=0.1),
            norm_layer=dict(type='LN', normalized_shape=256),
            ffn=dict(
                type='AsymmetricFFN',
                in_channels=512,
                pre_norm=dict(type='LN'),
                embed_dims=256,
                feedforward_channels=1024,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)),
            deformable_model=dict(
                type='DeformableFeatureAggregation',
                embed_dims=256,
                num_groups=8,
                num_levels=4,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=True,
                use_camera_embed=True,
                residual_mode='cat',
                kps_generator=dict(
                    type='SparsePoint3DKeyPointsGenerator',
                    embed_dims=256,
                    num_sample=20,
                    num_learnable_pts=3,
                    fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=-1.84023)),
            refine_layer=dict(
                type='SparsePoint3DRefinementModule',
                embed_dims=256,
                num_sample=20,
                num_cls=6),
            sampler=dict(
                type='SparsePoint3DTarget',
                assigner=dict(
                    type='HungarianLinesAssigner',
                    cost=dict(
                        type='MapQueriesCost',
                        cls_cost=dict(type='FocalLossCost', weight=1.0),
                        reg_cost=dict(
                            type='LinesL1Cost',
                            weight=10.0,
                            beta=0.01,
                            permute=True))),
                num_cls=6,
                num_sample=20,
                roi_size=(30, 60)),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_reg=dict(
                type='SparseLineLoss',
                loss_line=dict(
                    type='LinesL1Loss', loss_weight=25.0, beta=0.01),
                num_sample=20,
                roi_size=(30, 60)),
            decoder=dict(type='SparsePoint3DDecoder'),
            reg_weights=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0
            ],
            gt_cls_key='gt_map_labels',
            gt_reg_key='gt_map_pts',
            gt_id_key='map_instance_id',
            with_instance_id=False,
            task_prefix='map'),
        temporal_queue=dict(temporal_img=-1, temporal_gap=3),
        num_decoder=6,
        dual_plan_head=None,
        motion_plan_head=dict(
            type='LongitudinalPlanningModule',
            cls_threshold_to_reg=0.05,
            auxiliary_loss=dict(
                dynamic_loss=False,
                dynamic_loss_longitudinal=True,
                dynamic_horzion=10,
                dynamic_loss_scale=0.1,
                dynamic_dead_range=0.5,
                dynamic_all_trajs=False,
                dynamic_loss_other_trajs_scale=0.2,
                lateral_loss=False,
                lateral_loss_longitudinal=False,
                lateral_loss_scale=0.2,
                start_iter=5502),
            command_res=False,
            plan_control='close_loop',
            anchor_supervise=False,
            decouple_attn=False,
            gloalpoint_extend=True,
            force_check_timeinterval=True,
            gt_modal='distance',
            final_used_head='longitudinal',
            longitudinal_train_use_grand=dict(
                temporal='temporal_interval_2_traj',
                spatial='spatial_interval_2_traj',
                augmentation='fake_defensive_ego_fut_trajs_5hz',
                augmented_bbox='fake_bboxes_3d_5hz',
                augmented_bbox_trajs='fake_agent_fut_trajs_5hz',
                interval=2,
                num_horzion=15,
                instance_feature=True,
                confidence_thrhold=0.3,
                start_iter=5502),
            augment_bbox=False,
            augment_bbox_longitudinal=True,
            instance_bank=dict(
                type='InstanceBankPlan',
                num_anchor=24,
                embed_dims=256,
                anchor=
                './kmeans/b2d/train_mutil_granular_waypoint_cluster_refine_1.pkl',
                anchor_mearge=True,
                anchor_handler=dict(type='SparsePoint3DKeyPointsGenerator'),
                num_temp_instances=24,
                confidence_decay=0.6,
                feat_grad=True,
                project_temporal_anchor=False,
                queue_length=50,
                temporal_gap=0.0,
                used_grand=[
                    'spatial_interval_2_traj', 'spatial_interval_5_traj',
                    'temporal_interval_2_traj', 'temporal_interval_5_traj'
                ]),
            anchor_encoder=dict(
                type='SparsePoint3DEncoder', embed_dims=256, num_sample=15),
            anchor_encoder_low_frency=dict(
                type='SparsePoint3DEncoder', embed_dims=256, num_sample=6),
            num_single_frame_decoder=1,
            temp_graph_model=dict(
                type='MultiheadFlashAttention',
                embed_dims=256,
                num_heads=8,
                batch_first=True,
                dropout=0.1),
            graph_model=dict(
                type='MultiheadFlashAttention',
                embed_dims=256,
                num_heads=8,
                batch_first=True,
                dropout=0.1),
            norm_layer=dict(type='LN', normalized_shape=256),
            img_feature_extrctor=dict(
                type='GlobalFeatureExtractor',
                embed_dims=256,
                feature_map_index=[-1],
                add_3dpositional_encoding=True),
            deformable_model=dict(
                type='DeformableFeatureAggregation',
                embed_dims=256,
                num_groups=8,
                num_levels=4,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=True,
                use_camera_embed=True,
                residual_mode='add',
                kps_generator=dict(
                    type='SparsePoint3DKeyPointsGenerator',
                    embed_dims=256,
                    num_sample=15,
                    num_learnable_pts=3,
                    fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=-1.84023)),
            spdeformable_model=None,
            objectdeformable_model=dict(
                type='ObjectDFA',
                embed_dims=256,
                num_groups=8,
                num_levels=4,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=False,
                use_camera_embed=True,
                residual_mode='add',
                kps_generator=None,
                num_pts=195),
            sampler=dict(
                type='MultiGranularityPlanningTarget',
                ego_fut_ts=15,
                ego_fut_mode=6,
                plan_vocab_path=
                './kmeans/b2d/train_mutil_granular_waypoint_cluster_refine_1.pkl',
                used_grand=[
                    'spatial_interval_2_traj', 'spatial_interval_5_traj',
                    'temporal_interval_2_traj', 'temporal_interval_5_traj'
                ],
                SpeedAwareRange=[[0.1, 0.4], [0.4, 3.0], [3.0, 5.0],
                                 [5.0, 7.0], [7.0, 10.0]],
                rand_grand_pro=0,
                num_drivestyle=0),
            navigation_module=dict(type='NavigationAware', mode='sum'),
            decoder=dict(type='MaplikePlanDecoder'),
            reg_weights=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
            ],
            gt_cls_key='gt_ego_fut_cmd',
            gt_reg_key='gt_ego_fut_trajs',
            gt_id_key='map_instance_id',
            with_instance_id=False,
            task_prefix='plan',
            plan_loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            plan_loss_reg=dict(
                type='PlanningRegLoss',
                loss_weight=1.0,
                base_loss=dict(type='L1Loss', loss_weight=1.0),
                temporal_weight=[
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                    0.4, 0.4, 0.4
                ],
                balance_setting=dict(
                    spatial_interval_2_traj=dict(
                        temporal_weighted=[
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 0.6, 0.6,
                            0.6, 0.6, 0.4, 0.4, 0.4
                        ],
                        total_weighted=3.0),
                    spatial_interval_5_traj=dict(
                        temporal_weighted=[
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0
                        ],
                        total_weighted=1.0),
                    temporal_interval_2_traj=dict(
                        temporal_weighted=[
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 0.6, 0.6,
                            0.6, 0.6, 0.4, 0.4, 0.4
                        ],
                        total_weighted=3.0),
                    temporal_interval_5_traj=dict(
                        temporal_weighted=[
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0
                        ],
                        total_weighted=1.0),
                    command=dict({
                        '0': 1.0,
                        '1': 1.0,
                        '2': 1.0,
                        '3': 1.0,
                        '4': 1.0,
                        '5': 1.0,
                        '6': 1.0
                    }),
                    command_scale_iter_map=5502,
                    command_scale_iter_plan=5502)),
            collision_loss=dict(type='PlanningCollisionLoss', weights=1.0),
            plan_loss_score=dict(type='L1Loss', loss_weight=1.0),
            keypoint_loss=dict(type='MSELoss', loss_weight=1.0),
            plan_loss_status=dict(type='L1Loss', loss_weight=0.2),
            longitudinal_head_scale=1.0,
            num_cmd=7,
            num_sample_waypoints=15,
            temporal_img=-1,
            use_GAN=False,
            temporal_gap=3,
            use_perception_query=True,
            use_tp=True,
            use_command=True,
            collision_check=False,
            anchor_mearge=True,
            num_temporal_plan=50,
            operation_order=[[
                'add_navigation', 'norm', 'deformable', 'ffn', 'norm', 'refine'
            ],
                             [
                                 'temp_gnn', 'add_navigation', 'norm',
                                 'deformable', 'ffn', 'norm', 'refine'
                             ],
                             [
                                 'temp_gnn', 'add_navigation', 'norm',
                                 'deformable', 'ffn', 'norm', 'refine'
                             ],
                             [
                                 'temp_gnn', 'add_navigation', 'norm',
                                 'deformable', 'ffn', 'norm', 'refine'
                             ],
                             [
                                 'temp_gnn', 'add_navigation', 'norm',
                                 'deformable', 'ffn', 'norm', 'refine'
                             ],
                             [
                                 'temp_gnn', 'add_navigation', 'norm',
                                 'deformable', 'ffn', 'norm', 'refine'
                             ]],
            ffn=dict(
                type='AsymmetricFFN',
                in_channels=256,
                pre_norm=dict(type='LN'),
                embed_dims=256,
                feedforward_channels=1024,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)),
            use_ego_status=False,
            tp_res=False,
            refine_layer=dict(
                type='MaplikePlanQueryDecoder',
                embed_dims=256,
                num_sample=15,
                num_cls=1,
                offset_learn=True,
                with_status_branch=True,
                num_drivestyle=0,
                num_granularity=4,
                num_modal=6,
                mutil_head=True),
            other_output_pretext=['longitudinal'],
            conditional_loss_weighting=dict(),
            control_setting=dict(
                spatial='spatial_interval_2_traj',
                temporal='temporal_interval_2_traj',
                post_process_score=True,
                perferred_anchor=-1,
                light_state_enhancement=3,
                valid_waypoints=6)),
        num_key_object=15,
        query_drop_rate=0.0,
        fusion_model=dict(
            type='QueryFusion',
            distance_attn_model=dict(
                type='MultiheadFlashAttention',
                embed_dims=256,
                num_heads=8,
                batch_first=True,
                dropout=0.1,
                distance_attention=True),
            graph_model=dict(
                type='MultiheadFlashAttention',
                embed_dims=256,
                num_heads=8,
                batch_first=True,
                dropout=0.1),
            plan_mearge_model=dict(
                type='MaplikePlanQueryDecoder',
                embed_dims=256,
                num_sample=15,
                num_cls=1,
                offset_learn=True,
                with_status_branch=False),
            agent_anchor_encoder=dict(
                type='SparsePoint3DEncoder',
                embed_dims=256,
                num_sample=1,
                coords_dim=11),
            map_anchor_encoder=dict(
                type='SparsePoint3DEncoder',
                embed_dims=256,
                num_sample=20,
                coords_dim=2),
            norm_layer=dict(type='LN', normalized_shape=256),
            operation_order=[[],
                             [
                                 'temp_cross_gnn', 'temp_cross_norm',
                                 'cross_gnn', 'cross_norm', 'agent_cross_map',
                                 'agent_cross_map_norm'
                             ],
                             [
                                 'temp_cross_gnn', 'temp_cross_norm',
                                 'cross_gnn', 'cross_norm', 'agent_cross_map',
                                 'agent_cross_map_norm'
                             ],
                             [
                                 'temp_cross_gnn', 'temp_cross_norm',
                                 'cross_gnn', 'cross_norm', 'agent_cross_map',
                                 'agent_cross_map_norm'
                             ],
                             [
                                 'temp_cross_gnn', 'temp_cross_norm',
                                 'cross_gnn', 'cross_norm', 'agent_cross_map',
                                 'agent_cross_map_norm'
                             ],
                             [
                                 'temp_cross_gnn', 'temp_cross_norm',
                                 'cross_gnn', 'cross_norm', 'agent_cross_map',
                                 'agent_cross_map_norm'
                             ]],
            merage_operation_order=None,
            num_query=1024,
            embed_dims=256,
            motion_queue=None,
            decode_perception_query=False,
            augment_bbox_pro=0.0,
            plan_num_det=50,
            plan_num_map=25),
        longitudinal_head=dict(
            type='DrivePathPredictor',
            use_ego_status=False,
            use_previous_trajectory=True,
            num_det=30,
            num_map=25,
            augment_bbox_pro=0.1,
            img_encoder=None,
            img_graph_model=None,
            use_gt_anchor=False,
            longitudinal_train_use_grand=dict(
                temporal='temporal_interval_2_traj',
                spatial='spatial_interval_2_traj',
                instance_feature=True,
                augmentation='fake_defensive_ego_fut_trajs_5hz',
                augmented_bbox='fake_bboxes_3d_5hz',
                augmented_bbox_trajs='fake_agent_fut_trajs_5hz',
                interval=2,
                num_horzion=15,
                confidence_thrhold=0.3,
                start_iter=5502),
            class_names=[
                'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                'traffic_cone', 'traffic_light_0', 'traffic_light_1',
                'traffic_light_2', 'pedestrian', 'others', 'traffic.stop'
            ],
            merage_traffic_light='none',
            instance_bank=dict(
                type='DrivePathBank',
                plan_grand=[
                    'spatial_interval_2_traj', 'spatial_interval_5_traj',
                    'temporal_interval_2_traj', 'temporal_interval_5_traj'
                ],
                num_modal=6,
                embed_dims=256,
                anchor_grand=None,
                mutil_step_query=16,
                speed_range=[[0.1, 0.4], [0.4, 3.0], [3.0, 5.0], [5.0, 7.0],
                             [7.0, 10.0]],
                num_ds=5,
                prediction_mode='distance',
                timeinterval=2,
                drivepath_grand='spatial_interval_2_traj'),
            HistoryPlanBank=dict(
                type='HistoryPlanBank',
                embed_dims=256,
                num_temp_instances=30,
                confidence_decay=0.6,
                project_temporal_anchor=False,
                queue_length=50,
                temporal_gap=0.0),
            anchor_encoder=dict(
                type='SparsePoint3DEncoder',
                embed_dims=256,
                num_sample=15,
                coords_dim=2),
            graph_model=dict(
                type='MultiheadFlashAttention',
                embed_dims=256,
                num_heads=8,
                batch_first=True,
                dropout=0.1,
                distance_attention=True),
            graph_model_no_distance=dict(
                type='MultiheadFlashAttention',
                embed_dims=256,
                num_heads=8,
                batch_first=True,
                dropout=0.1,
                distance_attention=False),
            temp_graph_model=dict(
                type='BlockAR',
                embed_dims=256,
                num_heads=8,
                batch_first=True,
                dropout=0.0),
            TemporalMixer=dict(type='TemporalMixer'),
            ffn=dict(
                type='AsymmetricFFN',
                in_channels=256,
                pre_norm=dict(type='LN'),
                embed_dims=256,
                feedforward_channels=1024,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)),
            refine_layer=dict(
                type='SpeedAwarePlanQueryDecoder',
                embed_dims=256,
                num_sample=15,
                num_cls=1,
                with_status_branch=False,
                offset_learn=True,
                num_drivestyle=0,
                num_granularity=5,
                num_modal=6,
                mutil_head=False,
                mutil_step_query=16,
                prediction_mode='distance'),
            norm_layer=dict(type='LN', normalized_shape=256),
            operation_order=[
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
                ],
                [
                    'reference_cross_gnn', 'reference_cross_norm', 'cross_gnn',
                    'cross_norm', 'map_cross_gnn', 'cross_norm',
                    'mutil_step_fusion', 'cross_norm', 'ffn', 'refine'
                ]
            ],
            embed_dims=256,
            p_topk=1,
            longitudinal_layer_range=[0, 1, 2, 3, 4, 5],
            decode_perception_query=True)),
    training_stage='normal',
    pretrained=None)
data_root = './data/Bench2Drive'
anno_root = './data/bench2drive_meta/infos_b2d_full'
class_names = [
    'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
    'traffic_light_0', 'traffic_light_1', 'traffic_light_2', 'pedestrian',
    'others', 'traffic.stop'
]
map_class_names = [
    'Broken', 'Solid', 'SolidSolid', 'Center', 'TrafficLight', 'StopSign'
]
NameMapping = dict({
    'vehicle.bh.crossbike':
    'bicycle',
    'vehicle.diamondback.century':
    'bicycle',
    'vehicle.gazelle.omafiets':
    'bicycle',
    'vehicle.chevrolet.impala':
    'car',
    'vehicle.dodge.charger_2020':
    'car',
    'vehicle.dodge.charger_police':
    'car',
    'vehicle.dodge.charger_police_2020':
    'car',
    'vehicle.lincoln.mkz_2017':
    'car',
    'vehicle.lincoln.mkz_2020':
    'car',
    'vehicle.mini.cooper_s_2021':
    'car',
    'vehicle.mercedes.coupe_2020':
    'car',
    'vehicle.ford.mustang':
    'car',
    'vehicle.nissan.patrol_2021':
    'car',
    'vehicle.audi.tt':
    'car',
    'vehicle.audi.etron':
    'car',
    'vehicle.ford.crown':
    'car',
    'vehicle.tesla.model3':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked':
    'van',
    'vehicle.ford.ambulance':
    'van',
    'vehicle.carlamotors.firetruck':
    'truck',
    'traffic.speed_limit.30':
    'traffic_sign',
    'traffic.speed_limit.40':
    'traffic_sign',
    'traffic.speed_limit.50':
    'traffic_sign',
    'traffic.speed_limit.60':
    'traffic_sign',
    'traffic.speed_limit.90':
    'traffic_sign',
    'traffic.speed_limit.120':
    'traffic_sign',
    'traffic.stop':
    'traffic.stop',
    'traffic.yield':
    'traffic_sign',
    'traffic.traffic_light':
    'traffic_light',
    'traffic.traffic_light_0':
    'traffic_light_0',
    'traffic.traffic_light_1':
    'traffic_light_1',
    'traffic.traffic_light_2':
    'traffic_light_2',
    'static.prop.warningconstruction':
    'traffic_cone',
    'static.prop.warningaccident':
    'traffic_cone',
    'static.prop.trafficwarning':
    'traffic_cone',
    'static.prop.constructioncone':
    'traffic_cone',
    'walker.pedestrian.0001':
    'pedestrian',
    'walker.pedestrian.0004':
    'pedestrian',
    'walker.pedestrian.0005':
    'pedestrian',
    'walker.pedestrian.0007':
    'pedestrian',
    'walker.pedestrian.0013':
    'pedestrian',
    'walker.pedestrian.0014':
    'pedestrian',
    'walker.pedestrian.0017':
    'pedestrian',
    'walker.pedestrian.0018':
    'pedestrian',
    'walker.pedestrian.0019':
    'pedestrian',
    'walker.pedestrian.0020':
    'pedestrian',
    'walker.pedestrian.0022':
    'pedestrian',
    'walker.pedestrian.0025':
    'pedestrian',
    'walker.pedestrian.0035':
    'pedestrian',
    'walker.pedestrian.0041':
    'pedestrian',
    'walker.pedestrian.0046':
    'pedestrian',
    'walker.pedestrian.0047':
    'pedestrian',
    'static.prop.dirtdebris01':
    'others',
    'static.prop.dirtdebris02':
    'others'
})
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
data_aug_conf = dict(
    resize_lim=(0.4, 0.47),
    final_dim=(352, 640),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(-5.4, 5.4),
    H=900,
    W=1600,
    rand_flip=True,
    rot3d_range=[0, 0])
perception_class_names = [
    'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
    'traffic_light', 'pedestrian', 'others', 'traffic_stop'
]
perception_name_mapping = dict({
    'vehicle.bh.crossbike':
    'bicycle',
    'vehicle.diamondback.century':
    'bicycle',
    'vehicle.gazelle.omafiets':
    'bicycle',
    'vehicle.chevrolet.impala':
    'car',
    'vehicle.dodge.charger_2020':
    'car',
    'vehicle.dodge.charger_police':
    'car',
    'vehicle.dodge.charger_police_2020':
    'car',
    'vehicle.lincoln.mkz_2017':
    'car',
    'vehicle.lincoln.mkz_2020':
    'car',
    'vehicle.mini.cooper_s_2021':
    'car',
    'vehicle.mercedes.coupe_2020':
    'car',
    'vehicle.ford.mustang':
    'car',
    'vehicle.nissan.patrol_2021':
    'car',
    'vehicle.audi.tt':
    'car',
    'vehicle.audi.etron':
    'car',
    'vehicle.ford.crown':
    'car',
    'vehicle.tesla.model3':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked':
    'van',
    'vehicle.ford.ambulance':
    'van',
    'vehicle.carlamotors.firetruck':
    'truck',
    'traffic.speed_limit.30':
    'traffic_sign',
    'traffic.speed_limit.40':
    'traffic_sign',
    'traffic.speed_limit.50':
    'traffic_sign',
    'traffic.speed_limit.60':
    'traffic_sign',
    'traffic.speed_limit.90':
    'traffic_sign',
    'traffic.speed_limit.120':
    'traffic_sign',
    'traffic.stop':
    'traffic_stop',
    'traffic.yield':
    'traffic_sign',
    'traffic.traffic_light':
    'traffic_light',
    'traffic.traffic_light_0':
    'traffic_light',
    'traffic.traffic_light_1':
    'traffic_light',
    'traffic.traffic_light_2':
    'traffic_light',
    'static.prop.warningconstruction':
    'traffic_cone',
    'static.prop.warningaccident':
    'traffic_cone',
    'static.prop.trafficwarning':
    'traffic_cone',
    'static.prop.constructioncone':
    'traffic_cone',
    'walker.pedestrian.0001':
    'pedestrian',
    'walker.pedestrian.0004':
    'pedestrian',
    'walker.pedestrian.0005':
    'pedestrian',
    'walker.pedestrian.0007':
    'pedestrian',
    'walker.pedestrian.0013':
    'pedestrian',
    'walker.pedestrian.0014':
    'pedestrian',
    'walker.pedestrian.0017':
    'pedestrian',
    'walker.pedestrian.0018':
    'pedestrian',
    'walker.pedestrian.0019':
    'pedestrian',
    'walker.pedestrian.0020':
    'pedestrian',
    'walker.pedestrian.0022':
    'pedestrian',
    'walker.pedestrian.0025':
    'pedestrian',
    'walker.pedestrian.0035':
    'pedestrian',
    'walker.pedestrian.0041':
    'pedestrian',
    'walker.pedestrian.0046':
    'pedestrian',
    'walker.pedestrian.0047':
    'pedestrian',
    'static.prop.dirtdebris01':
    'others',
    'static.prop.dirtdebris02':
    'others'
})
train_ann_file = './data/bench2drive_meta/infos_b2d_full/bench2drive_info_train.pkl'
val_ann_file = './data/bench2drive_meta/infos_b2d_full/bench2drive_info_val.pkl'
train_waypoints_file = './data/bench2drive_meta/infos_b2d_full/bench2drive_aligndrive_waypoints_train.pkl'
val_waypoints_file = './data/bench2drive_meta/infos_b2d_full/bench2drive_aligndrive_waypoints_val.pkl'
pipeline_steps = dict(
    load_multi_view_image_from_files=dict(
        to_float32=True, type='LoadMultiViewImageFromFiles'),
    resize_crop_flip_image=dict(type='ResizeCropFlipImage'),
    normalize_multiview_image=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
        type='NormalizeMultiviewImage'),
    nu_scenes_sparse4d_adaptor=dict(type='NuScenesSparse4DAdaptor'),
    photo_metric_distortion_multi_view_image=dict(
        type='PhotoMetricDistortionMultiViewImage'))
plugin = True
plugin_dir = 'projects_sp/mmdet3d_plugin/'
dist_params = dict(backend='nccl')
fp16 = dict(loss_scale=32.0)
work_dir = './work_dirs/aligndrive_inference'
num_cmd = 6
evaluation = dict(
    interval=1834,
    eval_mode=dict(
        with_det=False,
        with_tracking=False,
        with_map=False,
        with_motion=False,
        with_planning=True,
        tracking_threshold=0.2,
        motion_threshhold=0.2),
    metric='bbox',
    save_best='L2_final',
    rule='less')
custom_imports = dict(imports=[], allow_failed_imports=False)
control_setting = dict(
    spatial='spatial_interval_2_traj',
    temporal='temporal_interval_2_traj',
    post_process_score=True,
    perferred_anchor=-1,
    light_state_enhancement=3,
    valid_waypoints=6)
strict_checkpoint = True
require_longitudinal_output = True
test_pipeline = [
    dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
    dict(type='ResizeCropFlipImage'),
    dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
        type='NormalizeMultiviewImage'),
    dict(
        type='Process_EgoStatus',
        noise_pro=0,
        num_bins=1024,
        v_max=20,
        used_attri=['speed_0']),
    dict(type='NuScenesSparse4DAdaptor'),
    dict(type='Build_Navaigation', gloalpoint_extend=True),
    dict(
        type='Collect',
        keys=[
            'img', 'timestamp', 'projection_mat', 'image_wh', 'ego_status',
            'gt_ego_fut_cmd', 'tp_near', 'tp_far', 'tp_line', 'ego_speed'
        ],
        meta_keys=[
            'scenario_name', 'T_global', 'T_global_inv', 'timestamp',
            'lidar2img'
        ])
]
open_loop_pipeline = [
    dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
    dict(type='ResizeCropFlipImage'),
    dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
        type='NormalizeMultiviewImage'),
    dict(keys='gt_ego_fut_cmd', noise_pro=0, num_cmd=6, type='CommandNoise'),
    dict(
        noise_pro=0,
        num_bins=1024,
        type='Process_EgoStatus',
        used_attri=['speed_0'],
        v_max=20),
    dict(
        gloalpoint_extend=True,
        noise_pro=0.0,
        noise_scale=0.0,
        type='Build_Navaigation'),
    dict(type='NuScenesSparse4DAdaptor'),
    dict(
        keys=[
            'img', 'timestamp', 'projection_mat', 'image_wh', 'ego_status',
            'ego_status_digit', 'ego_speed', 'gt_ego_fut_cmd',
            'gt_ego_fut_trajs', 'tp_near', 'tp_far', 'tp_line',
            'spatial_interval_2_traj', 'spatial_interval_5_traj',
            'temporal_interval_2_traj', 'temporal_interval_5_traj',
            'spatial_interval_2_mask', 'spatial_interval_5_mask',
            'temporal_interval_2_mask', 'temporal_interval_5_mask'
        ],
        meta_keys=[
            'T_global', 'T_global_inv', 'timestamp', 'lidar2img',
            'scenario_name'
        ],
        type='Collect')
]
eval_config = dict(
    type='B2D3DDataset',
    data_root='./data/Bench2Drive',
    classes=[
        'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
        'traffic_light_0', 'traffic_light_1', 'traffic_light_2', 'pedestrian',
        'others', 'traffic.stop'
    ],
    map_classes=[
        'Broken', 'Solid', 'SolidSolid', 'Center', 'TrafficLight', 'StopSign'
    ],
    name_mapping=dict({
        'vehicle.bh.crossbike':
        'bicycle',
        'vehicle.diamondback.century':
        'bicycle',
        'vehicle.gazelle.omafiets':
        'bicycle',
        'vehicle.chevrolet.impala':
        'car',
        'vehicle.dodge.charger_2020':
        'car',
        'vehicle.dodge.charger_police':
        'car',
        'vehicle.dodge.charger_police_2020':
        'car',
        'vehicle.lincoln.mkz_2017':
        'car',
        'vehicle.lincoln.mkz_2020':
        'car',
        'vehicle.mini.cooper_s_2021':
        'car',
        'vehicle.mercedes.coupe_2020':
        'car',
        'vehicle.ford.mustang':
        'car',
        'vehicle.nissan.patrol_2021':
        'car',
        'vehicle.audi.tt':
        'car',
        'vehicle.audi.etron':
        'car',
        'vehicle.ford.crown':
        'car',
        'vehicle.tesla.model3':
        'car',
        '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked':
        'car',
        '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked':
        'car',
        '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked':
        'car',
        '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked':
        'car',
        '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked':
        'car',
        '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked':
        'car',
        '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked':
        'car',
        '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked':
        'van',
        'vehicle.ford.ambulance':
        'van',
        'vehicle.carlamotors.firetruck':
        'truck',
        'traffic.speed_limit.30':
        'traffic_sign',
        'traffic.speed_limit.40':
        'traffic_sign',
        'traffic.speed_limit.50':
        'traffic_sign',
        'traffic.speed_limit.60':
        'traffic_sign',
        'traffic.speed_limit.90':
        'traffic_sign',
        'traffic.speed_limit.120':
        'traffic_sign',
        'traffic.stop':
        'traffic.stop',
        'traffic.yield':
        'traffic_sign',
        'traffic.traffic_light':
        'traffic_light',
        'traffic.traffic_light_0':
        'traffic_light_0',
        'traffic.traffic_light_1':
        'traffic_light_1',
        'traffic.traffic_light_2':
        'traffic_light_2',
        'static.prop.warningconstruction':
        'traffic_cone',
        'static.prop.warningaccident':
        'traffic_cone',
        'static.prop.trafficwarning':
        'traffic_cone',
        'static.prop.constructioncone':
        'traffic_cone',
        'walker.pedestrian.0001':
        'pedestrian',
        'walker.pedestrian.0004':
        'pedestrian',
        'walker.pedestrian.0005':
        'pedestrian',
        'walker.pedestrian.0007':
        'pedestrian',
        'walker.pedestrian.0013':
        'pedestrian',
        'walker.pedestrian.0014':
        'pedestrian',
        'walker.pedestrian.0017':
        'pedestrian',
        'walker.pedestrian.0018':
        'pedestrian',
        'walker.pedestrian.0019':
        'pedestrian',
        'walker.pedestrian.0020':
        'pedestrian',
        'walker.pedestrian.0022':
        'pedestrian',
        'walker.pedestrian.0025':
        'pedestrian',
        'walker.pedestrian.0035':
        'pedestrian',
        'walker.pedestrian.0041':
        'pedestrian',
        'walker.pedestrian.0046':
        'pedestrian',
        'walker.pedestrian.0047':
        'pedestrian',
        'static.prop.dirtdebris01':
        'others',
        'static.prop.dirtdebris02':
        'others'
    }),
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    sample_interval=5,
    past_frames=2,
    future_frames=6,
    use_cmd=True,
    gloalpoint_extend=True,
    ann_file='./data/bench2drive_meta/infos_b2d_full/bench2drive_info_val.pkl',
    pipeline=[
        dict(
            class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55],
            type='CircleObjectRangeFilter'),
        dict(
            classes=[
                'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                'traffic_cone', 'traffic_light_0', 'traffic_light_1',
                'traffic_light_2', 'pedestrian', 'others', 'traffic.stop'
            ],
            type='InstanceNameFilter'),
        dict(
            normalize=False,
            roi_size=(30, 60),
            simplify=True,
            type='VectorizeMap'),
        dict(
            keys='gt_ego_fut_cmd', noise_pro=0, num_cmd=6,
            type='CommandNoise'),
        dict(
            noise_pro=0,
            num_bins=1024,
            type='Process_EgoStatus',
            used_attri=['speed_0'],
            v_max=20),
        dict(
            gloalpoint_extend=True,
            noise_pro=0.0,
            noise_scale=0.0,
            type='Build_Navaigation'),
        dict(
            keys=[
                'vectors', 'gt_bboxes_3d', 'gt_labels_3d',
                'gt_agent_fut_trajs', 'gt_agent_fut_masks', 'gt_ego_fut_trajs',
                'gt_ego_fut_masks', 'gt_ego_fut_cmd', 'ego_status_digit',
                'ego_speed', 'ego_status', 'tp_near', 'tp_far', 'tp_line',
                'spatial_interval_2_traj', 'spatial_interval_5_traj',
                'temporal_interval_2_traj', 'temporal_interval_5_traj',
                'spatial_interval_2_mask', 'spatial_interval_5_mask',
                'temporal_interval_2_mask', 'temporal_interval_5_mask'
            ],
            meta_keys=['token', 'timestamp', 'lidar2img', 'scenario_name'],
            type='Collect')
    ],
    test_mode=True,
    debug_mode=False,
    waypoints_file=
    './data/bench2drive_meta/infos_b2d_full/bench2drive_aligndrive_waypoints_val.pkl'
)
data = dict(
    workers_per_gpu=8,
    test=dict(
        type='B2D3DDataset',
        data_root='./data/Bench2Drive',
        classes=[
            'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
            'traffic_light_0', 'traffic_light_1', 'traffic_light_2',
            'pedestrian', 'others', 'traffic.stop'
        ],
        map_classes=[
            'Broken', 'Solid', 'SolidSolid', 'Center', 'TrafficLight',
            'StopSign'
        ],
        name_mapping=dict({
            'vehicle.bh.crossbike':
            'bicycle',
            'vehicle.diamondback.century':
            'bicycle',
            'vehicle.gazelle.omafiets':
            'bicycle',
            'vehicle.chevrolet.impala':
            'car',
            'vehicle.dodge.charger_2020':
            'car',
            'vehicle.dodge.charger_police':
            'car',
            'vehicle.dodge.charger_police_2020':
            'car',
            'vehicle.lincoln.mkz_2017':
            'car',
            'vehicle.lincoln.mkz_2020':
            'car',
            'vehicle.mini.cooper_s_2021':
            'car',
            'vehicle.mercedes.coupe_2020':
            'car',
            'vehicle.ford.mustang':
            'car',
            'vehicle.nissan.patrol_2021':
            'car',
            'vehicle.audi.tt':
            'car',
            'vehicle.audi.etron':
            'car',
            'vehicle.ford.crown':
            'car',
            'vehicle.tesla.model3':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked':
            'van',
            'vehicle.ford.ambulance':
            'van',
            'vehicle.carlamotors.firetruck':
            'truck',
            'traffic.speed_limit.30':
            'traffic_sign',
            'traffic.speed_limit.40':
            'traffic_sign',
            'traffic.speed_limit.50':
            'traffic_sign',
            'traffic.speed_limit.60':
            'traffic_sign',
            'traffic.speed_limit.90':
            'traffic_sign',
            'traffic.speed_limit.120':
            'traffic_sign',
            'traffic.stop':
            'traffic.stop',
            'traffic.yield':
            'traffic_sign',
            'traffic.traffic_light':
            'traffic_light',
            'traffic.traffic_light_0':
            'traffic_light_0',
            'traffic.traffic_light_1':
            'traffic_light_1',
            'traffic.traffic_light_2':
            'traffic_light_2',
            'static.prop.warningconstruction':
            'traffic_cone',
            'static.prop.warningaccident':
            'traffic_cone',
            'static.prop.trafficwarning':
            'traffic_cone',
            'static.prop.constructioncone':
            'traffic_cone',
            'walker.pedestrian.0001':
            'pedestrian',
            'walker.pedestrian.0004':
            'pedestrian',
            'walker.pedestrian.0005':
            'pedestrian',
            'walker.pedestrian.0007':
            'pedestrian',
            'walker.pedestrian.0013':
            'pedestrian',
            'walker.pedestrian.0014':
            'pedestrian',
            'walker.pedestrian.0017':
            'pedestrian',
            'walker.pedestrian.0018':
            'pedestrian',
            'walker.pedestrian.0019':
            'pedestrian',
            'walker.pedestrian.0020':
            'pedestrian',
            'walker.pedestrian.0022':
            'pedestrian',
            'walker.pedestrian.0025':
            'pedestrian',
            'walker.pedestrian.0035':
            'pedestrian',
            'walker.pedestrian.0041':
            'pedestrian',
            'walker.pedestrian.0046':
            'pedestrian',
            'walker.pedestrian.0047':
            'pedestrian',
            'static.prop.dirtdebris01':
            'others',
            'static.prop.dirtdebris02':
            'others'
        }),
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        sample_interval=5,
        past_frames=2,
        future_frames=6,
        use_cmd=True,
        gloalpoint_extend=True,
        ann_file=
        './data/bench2drive_meta/infos_b2d_full/bench2drive_info_val.pkl',
        waypoints_file=
        './data/bench2drive_meta/infos_b2d_full/bench2drive_aligndrive_waypoints_val.pkl',
        pipeline=[
            dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
            dict(type='ResizeCropFlipImage'),
            dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
                type='NormalizeMultiviewImage'),
            dict(
                keys='gt_ego_fut_cmd',
                noise_pro=0,
                num_cmd=6,
                type='CommandNoise'),
            dict(
                noise_pro=0,
                num_bins=1024,
                type='Process_EgoStatus',
                used_attri=['speed_0'],
                v_max=20),
            dict(
                gloalpoint_extend=True,
                noise_pro=0.0,
                noise_scale=0.0,
                type='Build_Navaigation'),
            dict(type='NuScenesSparse4DAdaptor'),
            dict(
                keys=[
                    'img', 'timestamp', 'projection_mat', 'image_wh',
                    'ego_status', 'ego_status_digit', 'ego_speed',
                    'gt_ego_fut_cmd', 'gt_ego_fut_trajs', 'tp_near', 'tp_far',
                    'tp_line', 'spatial_interval_2_traj',
                    'spatial_interval_5_traj', 'temporal_interval_2_traj',
                    'temporal_interval_5_traj', 'spatial_interval_2_mask',
                    'spatial_interval_5_mask', 'temporal_interval_2_mask',
                    'temporal_interval_5_mask'
                ],
                meta_keys=[
                    'T_global', 'T_global_inv', 'timestamp', 'lidar2img',
                    'scenario_name'
                ],
                type='Collect')
        ],
        data_aug_conf=dict(
            resize_lim=(0.4, 0.47),
            final_dim=(352, 640),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(-5.4, 5.4),
            H=900,
            W=1600,
            rand_flip=True,
            rot3d_range=[0, 0]),
        test_mode=True,
        eval_config=dict(
            type='B2D3DDataset',
            data_root='./data/Bench2Drive',
            classes=[
                'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                'traffic_cone', 'traffic_light_0', 'traffic_light_1',
                'traffic_light_2', 'pedestrian', 'others', 'traffic.stop'
            ],
            map_classes=[
                'Broken', 'Solid', 'SolidSolid', 'Center', 'TrafficLight',
                'StopSign'
            ],
            name_mapping=dict({
                'vehicle.bh.crossbike':
                'bicycle',
                'vehicle.diamondback.century':
                'bicycle',
                'vehicle.gazelle.omafiets':
                'bicycle',
                'vehicle.chevrolet.impala':
                'car',
                'vehicle.dodge.charger_2020':
                'car',
                'vehicle.dodge.charger_police':
                'car',
                'vehicle.dodge.charger_police_2020':
                'car',
                'vehicle.lincoln.mkz_2017':
                'car',
                'vehicle.lincoln.mkz_2020':
                'car',
                'vehicle.mini.cooper_s_2021':
                'car',
                'vehicle.mercedes.coupe_2020':
                'car',
                'vehicle.ford.mustang':
                'car',
                'vehicle.nissan.patrol_2021':
                'car',
                'vehicle.audi.tt':
                'car',
                'vehicle.audi.etron':
                'car',
                'vehicle.ford.crown':
                'car',
                'vehicle.tesla.model3':
                'car',
                '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked':
                'car',
                '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked':
                'car',
                '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked':
                'car',
                '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked':
                'car',
                '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked':
                'car',
                '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked':
                'car',
                '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked':
                'car',
                '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked':
                'van',
                'vehicle.ford.ambulance':
                'van',
                'vehicle.carlamotors.firetruck':
                'truck',
                'traffic.speed_limit.30':
                'traffic_sign',
                'traffic.speed_limit.40':
                'traffic_sign',
                'traffic.speed_limit.50':
                'traffic_sign',
                'traffic.speed_limit.60':
                'traffic_sign',
                'traffic.speed_limit.90':
                'traffic_sign',
                'traffic.speed_limit.120':
                'traffic_sign',
                'traffic.stop':
                'traffic.stop',
                'traffic.yield':
                'traffic_sign',
                'traffic.traffic_light':
                'traffic_light',
                'traffic.traffic_light_0':
                'traffic_light_0',
                'traffic.traffic_light_1':
                'traffic_light_1',
                'traffic.traffic_light_2':
                'traffic_light_2',
                'static.prop.warningconstruction':
                'traffic_cone',
                'static.prop.warningaccident':
                'traffic_cone',
                'static.prop.trafficwarning':
                'traffic_cone',
                'static.prop.constructioncone':
                'traffic_cone',
                'walker.pedestrian.0001':
                'pedestrian',
                'walker.pedestrian.0004':
                'pedestrian',
                'walker.pedestrian.0005':
                'pedestrian',
                'walker.pedestrian.0007':
                'pedestrian',
                'walker.pedestrian.0013':
                'pedestrian',
                'walker.pedestrian.0014':
                'pedestrian',
                'walker.pedestrian.0017':
                'pedestrian',
                'walker.pedestrian.0018':
                'pedestrian',
                'walker.pedestrian.0019':
                'pedestrian',
                'walker.pedestrian.0020':
                'pedestrian',
                'walker.pedestrian.0022':
                'pedestrian',
                'walker.pedestrian.0025':
                'pedestrian',
                'walker.pedestrian.0035':
                'pedestrian',
                'walker.pedestrian.0041':
                'pedestrian',
                'walker.pedestrian.0046':
                'pedestrian',
                'walker.pedestrian.0047':
                'pedestrian',
                'static.prop.dirtdebris01':
                'others',
                'static.prop.dirtdebris02':
                'others'
            }),
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            sample_interval=5,
            past_frames=2,
            future_frames=6,
            use_cmd=True,
            gloalpoint_extend=True,
            ann_file=
            './data/bench2drive_meta/infos_b2d_full/bench2drive_info_val.pkl',
            pipeline=[
                dict(
                    class_dist_thred=[
                        55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55
                    ],
                    type='CircleObjectRangeFilter'),
                dict(
                    classes=[
                        'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                        'traffic_cone', 'traffic_light_0', 'traffic_light_1',
                        'traffic_light_2', 'pedestrian', 'others',
                        'traffic.stop'
                    ],
                    type='InstanceNameFilter'),
                dict(
                    normalize=False,
                    roi_size=(30, 60),
                    simplify=True,
                    type='VectorizeMap'),
                dict(
                    keys='gt_ego_fut_cmd',
                    noise_pro=0,
                    num_cmd=6,
                    type='CommandNoise'),
                dict(
                    noise_pro=0,
                    num_bins=1024,
                    type='Process_EgoStatus',
                    used_attri=['speed_0'],
                    v_max=20),
                dict(
                    gloalpoint_extend=True,
                    noise_pro=0.0,
                    noise_scale=0.0,
                    type='Build_Navaigation'),
                dict(
                    keys=[
                        'vectors', 'gt_bboxes_3d', 'gt_labels_3d',
                        'gt_agent_fut_trajs', 'gt_agent_fut_masks',
                        'gt_ego_fut_trajs', 'gt_ego_fut_masks',
                        'gt_ego_fut_cmd', 'ego_status_digit', 'ego_speed',
                        'ego_status', 'tp_near', 'tp_far', 'tp_line',
                        'spatial_interval_2_traj', 'spatial_interval_5_traj',
                        'temporal_interval_2_traj', 'temporal_interval_5_traj',
                        'spatial_interval_2_mask', 'spatial_interval_5_mask',
                        'temporal_interval_2_mask', 'temporal_interval_5_mask'
                    ],
                    meta_keys=[
                        'token', 'timestamp', 'lidar2img', 'scenario_name'
                    ],
                    type='Collect')
            ],
            test_mode=True,
            debug_mode=False,
            waypoints_file=
            './data/bench2drive_meta/infos_b2d_full/bench2drive_aligndrive_waypoints_val.pkl'
        ),
        debug_mode=False))
