_base_ = [
    '../_base_/models/setr_naive_pup.py',
    '../_base_/datasets/ade20k.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(img_size=512, align_corners=False,
                  pos_embed_interp=True, drop_rate=0., num_classes=150),
    decode_head=dict(img_size=512, align_corners=False, num_conv=2,
                     upsampling_method='bilinear', num_classes=150, conv3x3_conv1x1=False),
    auxiliary_head=[dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=9,
        img_size=512,
        embed_dim=1024,
        num_classes=150,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        align_corners=False,
        conv3x3_conv1x1=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='VisionTransformerUpHead',
            in_channels=1024,
            channels=512,
            in_index=14,
            img_size=512,
            embed_dim=1024,
            num_classes=150,
            norm_cfg=norm_cfg,
            num_conv=2,
            upsampling_method='bilinear',
            align_corners=False,
            conv3x3_conv1x1=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='VisionTransformerUpHead',
            in_channels=1024,
            channels=512,
            in_index=19,
            img_size=512,
            embed_dim=1024,
            num_classes=150,
            norm_cfg=norm_cfg,
            num_conv=2,
            upsampling_method='bilinear',
            align_corners=False,
            conv3x3_conv1x1=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    ])

optimizer = dict(lr=0.01, weight_decay=0.0,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)})
                 )
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341))
find_unused_parameters = True
data = dict(samples_per_gpu=2)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        img_ratios=[0.5, 0.75, 1.0],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True,
                 crop_size=crop_size, setr_multi_scale=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
