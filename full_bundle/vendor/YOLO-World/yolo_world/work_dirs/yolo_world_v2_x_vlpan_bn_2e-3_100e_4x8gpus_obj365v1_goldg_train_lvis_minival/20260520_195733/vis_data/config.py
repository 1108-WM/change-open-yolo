default_scope = 'mmyolo'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=100),
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,
        save_best='auto',
        max_keep_ckpts=2,
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = '/home/jia/wm_open-yolo/OpenYOLO3D/pretrained/checkpoints/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth'
resume = False
backend_args = None
_backend_args = None
tta_model = dict(
    type='mmdet.DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.65), max_per_img=300))
img_scales = [(640, 640), (320, 320), (960, 960)]
_multiscale_resize_transforms = [
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
            dict(
                type='LetterResize',
                scale=(640, 640),
                allow_scale_up=False,
                pad_val=dict(img=114))
        ]),
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=(320, 320)),
            dict(
                type='LetterResize',
                scale=(320, 320),
                allow_scale_up=False,
                pad_val=dict(img=114))
        ]),
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=(960, 960)),
            dict(
                type='LetterResize',
                scale=(960, 960),
                allow_scale_up=False,
                pad_val=dict(img=114))
        ])
]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type':
            'Compose',
            'transforms': [{
                'type': 'YOLOv5KeepRatioResize',
                'scale': (640, 640)
            }, {
                'type': 'LetterResize',
                'scale': (640, 640),
                'allow_scale_up': False,
                'pad_val': {
                    'img': 114
                }
            }]
        }, {
            'type':
            'Compose',
            'transforms': [{
                'type': 'YOLOv5KeepRatioResize',
                'scale': (320, 320)
            }, {
                'type': 'LetterResize',
                'scale': (320, 320),
                'allow_scale_up': False,
                'pad_val': {
                    'img': 114
                }
            }]
        }, {
            'type':
            'Compose',
            'transforms': [{
                'type': 'YOLOv5KeepRatioResize',
                'scale': (960, 960)
            }, {
                'type': 'LetterResize',
                'scale': (960, 960),
                'allow_scale_up': False,
                'pad_val': {
                    'img': 114
                }
            }]
        }],
                    [{
                        'type': 'mmdet.RandomFlip',
                        'prob': 1.0
                    }, {
                        'type': 'mmdet.RandomFlip',
                        'prob': 0.0
                    }], [{
                        'type': 'mmdet.LoadAnnotations',
                        'with_bbox': True
                    }],
                    [{
                        'type':
                        'mmdet.PackDetInputs',
                        'meta_keys':
                        ('img_id', 'img_path', 'ori_shape', 'img_shape',
                         'scale_factor', 'pad_param', 'flip', 'flip_direction')
                    }]])
]
data_root = 'data/coco/'
train_ann_file = 'annotations/instances_train2017.json'
train_data_prefix = 'train2017/'
val_ann_file = 'annotations/instances_val2017.json'
val_data_prefix = 'val2017/'
num_classes = 1203
train_batch_size_per_gpu = 16
train_num_workers = 8
persistent_workers = True
base_lr = 0.002
max_epochs = 100
close_mosaic_epochs = 2
model_test_cfg = dict(
    multi_label=True,
    nms_pre=30000,
    score_thr=0.001,
    nms=dict(type='nms', iou_threshold=0.7),
    max_per_img=300)
img_scale = (640, 640)
dataset_type = 'YOLOv5CocoDataset'
val_batch_size_per_gpu = 1
val_num_workers = 2
batch_shapes_cfg = None
deepen_factor = 1.0
widen_factor = 1.25
strides = [8, 16, 32]
last_stage_out_channels = 512
num_det_layers = 3
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
affine_scale = 0.9
max_aspect_ratio = 100
tal_topk = 10
tal_alpha = 0.5
tal_beta = 6.0
loss_cls_weight = 0.5
loss_bbox_weight = 7.5
loss_dfl_weight = 0.375
lr_factor = 0.01
weight_decay = 0.025
save_epoch_intervals = 2
val_interval_stage2 = 1
max_keep_ckpts = 2
model = dict(
    type='YOLOWorldDetector',
    data_preprocessor=dict(
        type='YOLOWDetDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True),
    backbone=dict(
        type='MultiModalYOLOBackbone',
        image_model=dict(
            type='YOLOv8CSPDarknet',
            arch='P5',
            last_stage_out_channels=512,
            deepen_factor=1.0,
            widen_factor=1.25,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True)),
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='openai/clip-vit-base-patch32',
            frozen_modules=['all'])),
    neck=dict(
        type='YOLOWorldPAFPN',
        deepen_factor=1.0,
        widen_factor=1.25,
        in_channels=[256, 512, 512],
        out_channels=[256, 512, 512],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
        guide_channels=512,
        embed_channels=[128, 256, 256],
        num_heads=[4, 8, 8],
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',
            num_classes=80,
            in_channels=[256, 512, 512],
            widen_factor=1.25,
            reg_max=16,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=[8, 16, 32],
            use_bn_head=True,
            embed_dims=512),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=0.5),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=7.5,
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=0.375)),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=80,
            use_ciou=True,
            topk=10,
            alpha=0.5,
            beta=6.0,
            eps=1e-09)),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300),
    mm_neck=True,
    num_train_classes=80,
    num_test_classes=1203)
albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]
pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True)
]
last_transform = [
    dict(
        type='mmdet.Albu',
        transforms=[
            dict(type='Blur', p=0.01),
            dict(type='MedianBlur', p=0.01),
            dict(type='ToGray', p=0.01),
            dict(type='CLAHE', p=0.01)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap=dict(img='image', gt_bboxes='bboxes')),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiModalMosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        pre_transform=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(0.09999999999999998, 1.9),
        max_aspect_ratio=100,
        border=(-320, -320),
        border_val=(114, 114, 114)),
    dict(
        type='mmdet.Albu',
        transforms=[
            dict(type='Blur', p=0.01),
            dict(type='MedianBlur', p=0.01),
            dict(type='ToGray', p=0.01),
            dict(type='CLAHE', p=0.01)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap=dict(img='image', gt_bboxes='bboxes')),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='RandomLoadText',
        num_neg_samples=(1203, 1203),
        max_num_samples=80,
        padding_to_max=True,
        padding_value=''),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction', 'texts'))
]
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
    dict(
        type='LetterResize',
        scale=(640, 640),
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(0.09999999999999998, 1.9),
        max_aspect_ratio=100,
        border_val=(114, 114, 114)),
    dict(
        type='mmdet.Albu',
        transforms=[
            dict(type='Blur', p=0.01),
            dict(type='MedianBlur', p=0.01),
            dict(type='ToGray', p=0.01),
            dict(type='CLAHE', p=0.01)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap=dict(img='image', gt_bboxes='bboxes')),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='RandomLoadText',
        num_neg_samples=(1203, 1203),
        max_num_samples=80,
        padding_to_max=True,
        padding_value=''),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction', 'texts'))
]
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolow_collate'),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='MultiModalDataset',
                dataset=dict(
                    type='YOLOv5Objects365V1Dataset',
                    data_root='data/objects365v1/',
                    ann_file='annotations/objects365_train.json',
                    data_prefix=dict(img='train/'),
                    filter_cfg=dict(filter_empty_gt=False, min_size=32)),
                class_text_path='data/texts/obj365v1_class_texts.json',
                pipeline=[
                    dict(type='LoadImageFromFile', backend_args=None),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        type='MultiModalMosaic',
                        img_scale=(640, 640),
                        pad_val=114.0,
                        pre_transform=[
                            dict(type='LoadImageFromFile', backend_args=None),
                            dict(type='LoadAnnotations', with_bbox=True)
                        ]),
                    dict(
                        type='YOLOv5RandomAffine',
                        max_rotate_degree=0.0,
                        max_shear_degree=0.0,
                        scaling_ratio_range=(0.09999999999999998, 1.9),
                        max_aspect_ratio=100,
                        border=(-320, -320),
                        border_val=(114, 114, 114)),
                    dict(
                        type='mmdet.Albu',
                        transforms=[
                            dict(type='Blur', p=0.01),
                            dict(type='MedianBlur', p=0.01),
                            dict(type='ToGray', p=0.01),
                            dict(type='CLAHE', p=0.01)
                        ],
                        bbox_params=dict(
                            type='BboxParams',
                            format='pascal_voc',
                            label_fields=[
                                'gt_bboxes_labels', 'gt_ignore_flags'
                            ]),
                        keymap=dict(img='image', gt_bboxes='bboxes')),
                    dict(type='YOLOv5HSVRandomAug'),
                    dict(type='mmdet.RandomFlip', prob=0.5),
                    dict(
                        type='RandomLoadText',
                        num_neg_samples=(1203, 1203),
                        max_num_samples=80,
                        padding_to_max=True,
                        padding_value=''),
                    dict(
                        type='mmdet.PackDetInputs',
                        meta_keys=('img_id', 'img_path', 'ori_shape',
                                   'img_shape', 'flip', 'flip_direction',
                                   'texts'))
                ]),
            dict(
                type='YOLOv5MixedGroundingDataset',
                data_root='data/flickr/',
                ann_file='annotations/final_flickr_separateGT_train.json',
                data_prefix=dict(img='full_images/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=[
                    dict(type='LoadImageFromFile', backend_args=None),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        type='MultiModalMosaic',
                        img_scale=(640, 640),
                        pad_val=114.0,
                        pre_transform=[
                            dict(type='LoadImageFromFile', backend_args=None),
                            dict(type='LoadAnnotations', with_bbox=True)
                        ]),
                    dict(
                        type='YOLOv5RandomAffine',
                        max_rotate_degree=0.0,
                        max_shear_degree=0.0,
                        scaling_ratio_range=(0.09999999999999998, 1.9),
                        max_aspect_ratio=100,
                        border=(-320, -320),
                        border_val=(114, 114, 114)),
                    dict(
                        type='mmdet.Albu',
                        transforms=[
                            dict(type='Blur', p=0.01),
                            dict(type='MedianBlur', p=0.01),
                            dict(type='ToGray', p=0.01),
                            dict(type='CLAHE', p=0.01)
                        ],
                        bbox_params=dict(
                            type='BboxParams',
                            format='pascal_voc',
                            label_fields=[
                                'gt_bboxes_labels', 'gt_ignore_flags'
                            ]),
                        keymap=dict(img='image', gt_bboxes='bboxes')),
                    dict(type='YOLOv5HSVRandomAug'),
                    dict(type='mmdet.RandomFlip', prob=0.5),
                    dict(
                        type='RandomLoadText',
                        num_neg_samples=(1203, 1203),
                        max_num_samples=80,
                        padding_to_max=True,
                        padding_value=''),
                    dict(
                        type='mmdet.PackDetInputs',
                        meta_keys=('img_id', 'img_path', 'ori_shape',
                                   'img_shape', 'flip', 'flip_direction',
                                   'texts'))
                ]),
            dict(
                type='YOLOv5MixedGroundingDataset',
                data_root='data/mixed_grounding/',
                ann_file='annotations/final_mixed_train_no_coco.json',
                data_prefix=dict(img='gqa/images/'),
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
                pipeline=[
                    dict(type='LoadImageFromFile', backend_args=None),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        type='MultiModalMosaic',
                        img_scale=(640, 640),
                        pad_val=114.0,
                        pre_transform=[
                            dict(type='LoadImageFromFile', backend_args=None),
                            dict(type='LoadAnnotations', with_bbox=True)
                        ]),
                    dict(
                        type='YOLOv5RandomAffine',
                        max_rotate_degree=0.0,
                        max_shear_degree=0.0,
                        scaling_ratio_range=(0.09999999999999998, 1.9),
                        max_aspect_ratio=100,
                        border=(-320, -320),
                        border_val=(114, 114, 114)),
                    dict(
                        type='mmdet.Albu',
                        transforms=[
                            dict(type='Blur', p=0.01),
                            dict(type='MedianBlur', p=0.01),
                            dict(type='ToGray', p=0.01),
                            dict(type='CLAHE', p=0.01)
                        ],
                        bbox_params=dict(
                            type='BboxParams',
                            format='pascal_voc',
                            label_fields=[
                                'gt_bboxes_labels', 'gt_ignore_flags'
                            ]),
                        keymap=dict(img='image', gt_bboxes='bboxes')),
                    dict(type='YOLOv5HSVRandomAug'),
                    dict(type='mmdet.RandomFlip', prob=0.5),
                    dict(
                        type='RandomLoadText',
                        num_neg_samples=(1203, 1203),
                        max_num_samples=80,
                        padding_to_max=True,
                        padding_value=''),
                    dict(
                        type='mmdet.PackDetInputs',
                        meta_keys=('img_id', 'img_path', 'ori_shape',
                                   'img_shape', 'flip', 'flip_direction',
                                   'texts'))
                ])
        ],
        ignore_keys=['classes', 'palette']))
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
    dict(
        type='LetterResize',
        scale=(640, 640),
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param', 'texts'))
]
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MultiModalDataset',
        dataset=dict(
            type='YOLOv5LVISV1Dataset',
            data_root='data/coco/',
            test_mode=True,
            ann_file='lvis/lvis_v1_minival_inserted_image_name.json',
            data_prefix=dict(img=''),
            batch_shapes_cfg=None),
        class_text_path='data/texts/lvis_v1_class_texts.json',
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
            dict(
                type='LetterResize',
                scale=(640, 640),
                allow_scale_up=False,
                pad_val=dict(img=114)),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(type='LoadText'),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'pad_param', 'texts'))
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MultiModalDataset',
        dataset=dict(
            type='YOLOv5LVISV1Dataset',
            data_root='data/coco/',
            test_mode=True,
            ann_file='lvis/lvis_v1_minival_inserted_image_name.json',
            data_prefix=dict(img=''),
            batch_shapes_cfg=None),
        class_text_path='data/texts/lvis_v1_class_texts.json',
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
            dict(
                type='LetterResize',
                scale=(640, 640),
                allow_scale_up=False,
                pad_val=dict(img=114)),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(type='LoadText'),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'pad_param', 'texts'))
        ]))
param_scheduler = None
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='AdamW', lr=0.002, weight_decay=0.025, batch_size_per_gpu=16),
    constructor='YOLOWv5OptimizerConstructor',
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys=dict({
            'backbone.text_model': dict(lr_mult=0.01),
            'logit_scale': dict(weight_decay=0.0)
        })))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=98,
        switch_pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
            dict(
                type='LetterResize',
                scale=(640, 640),
                allow_scale_up=True,
                pad_val=dict(img=114.0)),
            dict(
                type='YOLOv5RandomAffine',
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(0.09999999999999998, 1.9),
                max_aspect_ratio=100,
                border_val=(114, 114, 114)),
            dict(
                type='mmdet.Albu',
                transforms=[
                    dict(type='Blur', p=0.01),
                    dict(type='MedianBlur', p=0.01),
                    dict(type='ToGray', p=0.01),
                    dict(type='CLAHE', p=0.01)
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
                keymap=dict(img='image', gt_bboxes='bboxes')),
            dict(type='YOLOv5HSVRandomAug'),
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(
                type='RandomLoadText',
                num_neg_samples=(1203, 1203),
                max_num_samples=80,
                padding_to_max=True,
                padding_value=''),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'flip', 'flip_direction', 'texts'))
        ])
]
val_evaluator = dict(
    type='mmdet.LVISMetric',
    proposal_nums=(100, 1, 10),
    ann_file='data/coco/lvis/lvis_v1_minival_inserted_image_name.json',
    metric='bbox')
test_evaluator = dict(
    type='mmdet.LVISMetric',
    proposal_nums=(100, 1, 10),
    ann_file='data/coco/lvis/lvis_v1_minival_inserted_image_name.json',
    metric='bbox')
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_interval=10,
    dynamic_intervals=[(98, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
mixup_prob = 0.15
mosaic_affine_transform = [
    dict(
        type='Mosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        pre_transform=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100,
        scaling_ratio_range=(0.09999999999999998, 1.9),
        border=(-320, -320),
        border_val=(114, 114, 114))
]
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)
num_training_classes = 80
text_channels = 512
neck_embed_channels = [128, 256, 256]
neck_num_heads = [4, 8, 8]
text_transform = [
    dict(
        type='RandomLoadText',
        num_neg_samples=(1203, 1203),
        max_num_samples=80,
        padding_to_max=True,
        padding_value=''),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction', 'texts'))
]
obj365v1_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5Objects365V1Dataset',
        data_root='data/objects365v1/',
        ann_file='annotations/objects365_train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/obj365v1_class_texts.json',
    pipeline=[
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='MultiModalMosaic',
            img_scale=(640, 640),
            pad_val=114.0,
            pre_transform=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='LoadAnnotations', with_bbox=True)
            ]),
        dict(
            type='YOLOv5RandomAffine',
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            scaling_ratio_range=(0.09999999999999998, 1.9),
            max_aspect_ratio=100,
            border=(-320, -320),
            border_val=(114, 114, 114)),
        dict(
            type='mmdet.Albu',
            transforms=[
                dict(type='Blur', p=0.01),
                dict(type='MedianBlur', p=0.01),
                dict(type='ToGray', p=0.01),
                dict(type='CLAHE', p=0.01)
            ],
            bbox_params=dict(
                type='BboxParams',
                format='pascal_voc',
                label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
            keymap=dict(img='image', gt_bboxes='bboxes')),
        dict(type='YOLOv5HSVRandomAug'),
        dict(type='mmdet.RandomFlip', prob=0.5),
        dict(
            type='RandomLoadText',
            num_neg_samples=(1203, 1203),
            max_num_samples=80,
            padding_to_max=True,
            padding_value=''),
        dict(
            type='mmdet.PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                       'flip_direction', 'texts'))
    ])
mg_train_dataset = dict(
    type='YOLOv5MixedGroundingDataset',
    data_root='data/mixed_grounding/',
    ann_file='annotations/final_mixed_train_no_coco.json',
    data_prefix=dict(img='gqa/images/'),
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=[
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='MultiModalMosaic',
            img_scale=(640, 640),
            pad_val=114.0,
            pre_transform=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='LoadAnnotations', with_bbox=True)
            ]),
        dict(
            type='YOLOv5RandomAffine',
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            scaling_ratio_range=(0.09999999999999998, 1.9),
            max_aspect_ratio=100,
            border=(-320, -320),
            border_val=(114, 114, 114)),
        dict(
            type='mmdet.Albu',
            transforms=[
                dict(type='Blur', p=0.01),
                dict(type='MedianBlur', p=0.01),
                dict(type='ToGray', p=0.01),
                dict(type='CLAHE', p=0.01)
            ],
            bbox_params=dict(
                type='BboxParams',
                format='pascal_voc',
                label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
            keymap=dict(img='image', gt_bboxes='bboxes')),
        dict(type='YOLOv5HSVRandomAug'),
        dict(type='mmdet.RandomFlip', prob=0.5),
        dict(
            type='RandomLoadText',
            num_neg_samples=(1203, 1203),
            max_num_samples=80,
            padding_to_max=True,
            padding_value=''),
        dict(
            type='mmdet.PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                       'flip_direction', 'texts'))
    ])
flickr_train_dataset = dict(
    type='YOLOv5MixedGroundingDataset',
    data_root='data/flickr/',
    ann_file='annotations/final_flickr_separateGT_train.json',
    data_prefix=dict(img='full_images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=[
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='MultiModalMosaic',
            img_scale=(640, 640),
            pad_val=114.0,
            pre_transform=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='LoadAnnotations', with_bbox=True)
            ]),
        dict(
            type='YOLOv5RandomAffine',
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            scaling_ratio_range=(0.09999999999999998, 1.9),
            max_aspect_ratio=100,
            border=(-320, -320),
            border_val=(114, 114, 114)),
        dict(
            type='mmdet.Albu',
            transforms=[
                dict(type='Blur', p=0.01),
                dict(type='MedianBlur', p=0.01),
                dict(type='ToGray', p=0.01),
                dict(type='CLAHE', p=0.01)
            ],
            bbox_params=dict(
                type='BboxParams',
                format='pascal_voc',
                label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
            keymap=dict(img='image', gt_bboxes='bboxes')),
        dict(type='YOLOv5HSVRandomAug'),
        dict(type='mmdet.RandomFlip', prob=0.5),
        dict(
            type='RandomLoadText',
            num_neg_samples=(1203, 1203),
            max_num_samples=80,
            padding_to_max=True,
            padding_value=''),
        dict(
            type='mmdet.PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                       'flip_direction', 'texts'))
    ])
coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5LVISV1Dataset',
        data_root='data/coco/',
        test_mode=True,
        ann_file='lvis/lvis_v1_minival_inserted_image_name.json',
        data_prefix=dict(img=''),
        batch_shapes_cfg=None),
    class_text_path='data/texts/lvis_v1_class_texts.json',
    pipeline=[
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
        dict(
            type='LetterResize',
            scale=(640, 640),
            allow_scale_up=False,
            pad_val=dict(img=114)),
        dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
        dict(type='LoadText'),
        dict(
            type='mmdet.PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                       'scale_factor', 'pad_param', 'texts'))
    ])
work_dir = '/home/jia/wm_open-yolo/OpenYOLO3D/models/YOLO-World/yolo_world/work_dirs/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival'
