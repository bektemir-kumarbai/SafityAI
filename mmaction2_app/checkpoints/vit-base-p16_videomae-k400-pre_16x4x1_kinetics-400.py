ann_file_test = 'data/kinetics400/kinetics400_val_list_videos.txt'
data_root_val = 'data/kinetics400/videos_val'
dataset_type = 'VideoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        depth=12,
        embed_dims=768,
        img_size=224,
        mlp_ratio=4,
        norm_cfg=dict(eps=1e-06, type='LN'),
        num_frames=16,
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        type='VisionTransformer'),
    cls_head=dict(
        average_clips='prob',
        in_channels=768,
        num_classes=400,
        type='TimeSformerHead'),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    type='Recognizer3D')
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='data/kinetics400/kinetics400_val_list_videos.txt',
        data_prefix=dict(video='data/kinetics400/videos_val'),
        pipeline=[
            dict(type='DecordInit'),
            dict(
                clip_len=16,
                frame_interval=4,
                num_clips=5,
                test_mode=True,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                224,
            ), type='Resize'),
            dict(crop_size=224, type='ThreeCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='AccMetric')
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        clip_len=16,
        frame_interval=4,
        num_clips=5,
        test_mode=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        224,
    ), type='Resize'),
    dict(crop_size=224, type='ThreeCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
