_base_ = [
    '../../_base_/dataset/food_dataset.py',
    '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='RecognizerHeadOnly',
    backbone=dict(
        type='Indentity'),
    cls_head=dict(
        type='MLGCNHead',
        num_classes={{ _base_.num_classes }},
        multi_class=True,
        loss_cls=dict(type='BCELossWithLogits', loss_weight=333.),
        in_channels=768,
        num_layers=1,
        tag_embedding_path={{ _base_.tag_embedding_path }},
        graph_path={{ _base_.tag_graph_path }},
        label_map_path={{ _base_.label_map_path }},
        tag_emb_dim=768,
        dropout_ratio=0.4),
    # model training and testing settings
    # train_cfg=dict(aux_info=['vertical']),
    test_cfg=dict(average_clips=None))

# dataset settings
dataset_type = 'FrameFeatureDataset'

train_pipeline = [
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
val_pipeline = [
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=256,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file={{ _base_.ann_file_train }},
        data_prefix={{ _base_.data_root }},
        split='train',
        pipeline=train_pipeline,
        total_clips=8,
        num_classes={{ _base_.num_classes }}),
    val=dict(
        type=dataset_type,
        ann_file={{ _base_.ann_file_val }},
        data_prefix={{ _base_.data_root }},
        split='val',
        pipeline=val_pipeline,
        total_clips=8,
        num_classes={{ _base_.num_classes }}),
    test=dict(
        type=dataset_type,
        ann_file={{ _base_.ann_file_test }},
        data_prefix={{ _base_.data_root }},
        split='test',
        pipeline=test_pipeline,
        total_clips=8,
        num_classes={{ _base_.num_classes }}))

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0025, # 0.01 is used for 8 gpus 32 videos/gpu
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))

lr_config = dict(
    policy='Fixed')
total_epochs = 15

# runtime settings
checkpoint_config = dict(interval=1, by_epoch=True)
log_config = dict(
    interval=40,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])