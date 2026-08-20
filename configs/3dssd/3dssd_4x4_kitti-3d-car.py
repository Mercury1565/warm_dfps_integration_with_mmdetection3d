backend_args = None
class_names = [
    'Car',
]
data_root = 'data/kitti/'
dataset_type = 'KittiDataset'
db_sampler = dict(
    backend_args=None,
    classes=[
        'Car',
    ],
    data_root='data/kitti/',
    info_path='data/kitti/kitti_dbinfos_train.pkl',
    points_loader=dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    prepare=dict(
        filter_by_difficulty=[
            -1,
        ], filter_by_min_points=dict(Car=5)),
    rate=1.0,
    sample_groups=dict(Car=15))
default_hooks = dict(
    checkpoint=dict(interval=-1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
eval_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
input_modality = dict(use_camera=False, use_lidar=True)
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 0.002
metainfo = dict(classes=[
    'Car',
])
model = dict(
    backbone=dict(
        aggregation_channels=(
            64,
            128,
            256,
        ),
        fps_mods=(
            'D-FPS',
            'FS',
            (
                'F-FPS',
                'D-FPS',
            ),
        ),
        fps_sample_range_lists=(
            -1,
            -1,
            (
                512,
                -1,
            ),
        ),
        in_channels=4,
        norm_cfg=dict(eps=0.001, momentum=0.1, type='BN2d'),
        num_points=(
            4096,
            512,
            (
                256,
                256,
            ),
        ),
        num_samples=(
            (
                32,
                32,
                64,
            ),
            (
                32,
                32,
                64,
            ),
            (
                32,
                32,
                32,
            ),
        ),
        radii=(
            (
                0.2,
                0.4,
                0.8,
            ),
            (
                0.4,
                0.8,
                1.6,
            ),
            (
                1.6,
                3.2,
                4.8,
            ),
        ),
        sa_cfg=dict(
            normalize_xyz=False,
            pool_mod='max',
            type='PointSAModuleMSG',
            use_xyz=True),
        sa_channels=(
            (
                (
                    16,
                    16,
                    32,
                ),
                (
                    16,
                    16,
                    32,
                ),
                (
                    32,
                    32,
                    64,
                ),
            ),
            (
                (
                    64,
                    64,
                    128,
                ),
                (
                    64,
                    64,
                    128,
                ),
                (
                    64,
                    96,
                    128,
                ),
            ),
            (
                (
                    128,
                    128,
                    256,
                ),
                (
                    128,
                    192,
                    256,
                ),
                (
                    128,
                    256,
                    256,
                ),
            ),
        ),
        type='PointNet2SAMSG'),
    bbox_head=dict(
        bbox_coder=dict(
            num_dir_bins=12, type='AnchorFreeBBoxCoder', with_rot=True),
        center_loss=dict(
            loss_weight=1.0, reduction='sum', type='mmdet.SmoothL1Loss'),
        corner_loss=dict(
            loss_weight=1.0, reduction='sum', type='mmdet.SmoothL1Loss'),
        dir_class_loss=dict(
            loss_weight=1.0, reduction='sum', type='mmdet.CrossEntropyLoss'),
        dir_res_loss=dict(
            loss_weight=1.0, reduction='sum', type='mmdet.SmoothL1Loss'),
        num_classes=1,
        objectness_loss=dict(
            loss_weight=1.0,
            reduction='sum',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        pred_layer_cfg=dict(
            bias=True,
            cls_conv_channels=(128, ),
            conv_cfg=dict(type='Conv1d'),
            in_channels=1536,
            norm_cfg=dict(eps=0.001, momentum=0.1, type='BN1d'),
            reg_conv_channels=(128, ),
            shared_conv_channels=(
                512,
                128,
            )),
        size_res_loss=dict(
            loss_weight=1.0, reduction='sum', type='mmdet.SmoothL1Loss'),
        type='SSD3DHead',
        vote_aggregation_cfg=dict(
            bias=True,
            mlp_channels=(
                (
                    256,
                    256,
                    256,
                    512,
                ),
                (
                    256,
                    256,
                    512,
                    1024,
                ),
            ),
            norm_cfg=dict(eps=0.001, momentum=0.1, type='BN2d'),
            normalize_xyz=False,
            num_point=256,
            radii=(
                4.8,
                6.4,
            ),
            sample_nums=(
                16,
                32,
            ),
            type='PointSAModuleMSG',
            use_xyz=True),
        vote_loss=dict(
            loss_weight=1.0, reduction='sum', type='mmdet.SmoothL1Loss'),
        vote_module_cfg=dict(
            conv_cfg=dict(type='Conv1d'),
            conv_channels=(128, ),
            gt_per_seed=1,
            in_channels=256,
            norm_cfg=dict(eps=0.001, momentum=0.1, type='BN1d'),
            num_points=256,
            vote_xyz_range=(
                3.0,
                3.0,
                2.0,
            ),
            with_res_feat=False)),
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    test_cfg=dict(
        max_output_num=100,
        nms_cfg=dict(iou_thr=0.1, type='nms'),
        per_class_proposal=True,
        sample_mode='spec',
        score_thr=0.0),
    train_cfg=dict(
        expand_dims_length=0.05, pos_distance_thr=10.0, sample_mode='spec'),
    type='SSD3DNet')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.002, type='AdamW', weight_decay=0.0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=80,
        gamma=0.1,
        milestones=[
            45,
            60,
        ],
        type='MultiStepLR'),
]
point_cloud_range = [
    0,
    -40,
    -5,
    70,
    40,
    3,
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='training/velodyne_reduced'),
        data_root='data/kitti/',
        metainfo=dict(classes=[
            'Car',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(
                        rot_range=[
                            0,
                            0,
                        ],
                        scale_ratio_range=[
                            1.0,
                            1.0,
                        ],
                        translation_std=[
                            0,
                            0,
                            0,
                        ],
                        type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(
                        point_cloud_range=[
                            0,
                            -40,
                            -5,
                            70,
                            40,
                            3,
                        ],
                        type='PointsRangeFilter'),
                    dict(num_points=16384, type='PointSample'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/kitti/kitti_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    type='KittiMetric')
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(
        flip=False,
        img_scale=(
            1333,
            800,
        ),
        pts_scale_ratio=1,
        transforms=[
            dict(
                rot_range=[
                    0,
                    0,
                ],
                scale_ratio_range=[
                    1.0,
                    1.0,
                ],
                translation_std=[
                    0,
                    0,
                    0,
                ],
                type='GlobalRotScaleTrans'),
            dict(type='RandomFlip3D'),
            dict(
                point_cloud_range=[
                    0,
                    -40,
                    -5,
                    70,
                    40,
                    3,
                ],
                type='PointsRangeFilter'),
            dict(num_points=16384, type='PointSample'),
        ],
        type='MultiScaleFlipAug3D'),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(max_epochs=80, type='EpochBasedTrainLoop', val_interval=2)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        dataset=dict(
            ann_file='kitti_infos_train.pkl',
            backend_args=None,
            box_type_3d='LiDAR',
            data_prefix=dict(pts='training/velodyne_reduced'),
            data_root='data/kitti/',
            metainfo=dict(classes=[
                'Car',
            ]),
            modality=dict(use_camera=False, use_lidar=True),
            pipeline=[
                dict(
                    backend_args=None,
                    coord_type='LIDAR',
                    load_dim=4,
                    type='LoadPointsFromFile',
                    use_dim=4),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    point_cloud_range=[
                        0,
                        -40,
                        -5,
                        70,
                        40,
                        3,
                    ],
                    type='PointsRangeFilter'),
                dict(
                    point_cloud_range=[
                        0,
                        -40,
                        -5,
                        70,
                        40,
                        3,
                    ],
                    type='ObjectRangeFilter'),
                dict(
                    db_sampler=dict(
                        backend_args=None,
                        classes=[
                            'Car',
                        ],
                        data_root='data/kitti/',
                        info_path='data/kitti/kitti_dbinfos_train.pkl',
                        points_loader=dict(
                            backend_args=None,
                            coord_type='LIDAR',
                            load_dim=4,
                            type='LoadPointsFromFile',
                            use_dim=4),
                        prepare=dict(
                            filter_by_difficulty=[
                                -1,
                            ],
                            filter_by_min_points=dict(Car=5)),
                        rate=1.0,
                        sample_groups=dict(Car=15)),
                    type='ObjectSample'),
                dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
                dict(
                    global_rot_range=[
                        0.0,
                        0.0,
                    ],
                    num_try=100,
                    rot_range=[
                        -1.0471975511965976,
                        1.0471975511965976,
                    ],
                    translation_std=[
                        1.0,
                        1.0,
                        0,
                    ],
                    type='ObjectNoise'),
                dict(
                    rot_range=[
                        -0.78539816,
                        0.78539816,
                    ],
                    scale_ratio_range=[
                        0.9,
                        1.1,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(num_points=16384, type='PointSample'),
                dict(
                    keys=[
                        'points',
                        'gt_bboxes_3d',
                        'gt_labels_3d',
                    ],
                    type='Pack3DDetInputs'),
            ],
            test_mode=False,
            type='KittiDataset'),
        times=2,
        type='RepeatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        point_cloud_range=[
            0,
            -40,
            -5,
            70,
            40,
            3,
        ], type='PointsRangeFilter'),
    dict(
        point_cloud_range=[
            0,
            -40,
            -5,
            70,
            40,
            3,
        ], type='ObjectRangeFilter'),
    dict(
        db_sampler=dict(
            backend_args=None,
            classes=[
                'Car',
            ],
            data_root='data/kitti/',
            info_path='data/kitti/kitti_dbinfos_train.pkl',
            points_loader=dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            prepare=dict(
                filter_by_difficulty=[
                    -1,
                ], filter_by_min_points=dict(Car=5)),
            rate=1.0,
            sample_groups=dict(Car=15)),
        type='ObjectSample'),
    dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
    dict(
        global_rot_range=[
            0.0,
            0.0,
        ],
        num_try=100,
        rot_range=[
            -1.0471975511965976,
            1.0471975511965976,
        ],
        translation_std=[
            1.0,
            1.0,
            0,
        ],
        type='ObjectNoise'),
    dict(
        rot_range=[
            -0.78539816,
            0.78539816,
        ],
        scale_ratio_range=[
            0.9,
            1.1,
        ],
        type='GlobalRotScaleTrans'),
    dict(num_points=16384, type='PointSample'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='training/velodyne_reduced'),
        data_root='data/kitti/',
        metainfo=dict(classes=[
            'Car',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(
                        rot_range=[
                            0,
                            0,
                        ],
                        scale_ratio_range=[
                            1.0,
                            1.0,
                        ],
                        translation_std=[
                            0,
                            0,
                            0,
                        ],
                        type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(
                        point_cloud_range=[
                            0,
                            -40,
                            -5,
                            70,
                            40,
                            3,
                        ],
                        type='PointsRangeFilter'),
                    dict(num_points=16384, type='PointSample'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/kitti/kitti_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    type='KittiMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
