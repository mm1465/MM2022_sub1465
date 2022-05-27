_base_ = [
    '../../../../_base_/dataset/apparel_dataset.py',
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/schedules/default_schedule.py',
]

# Choose vertical
reload_dataset = False

# model settings
num_gnn_layers = 2
model = dict(
    type='TagGraphRecommender',
    gnn=dict(
        type='OurGNN',
        num_tags={{ _base_.num_classes }},
        video_in_dim=768,
        tag_in_dim=768,
        hidden_dim=768,
        video_emb_layer=2,
        layer_relations=[
            [('video', 'FollowedBy', 'video'),
                ('tag', 'SubTopic', 'tag'),
                ('video', 'HasTag', 'tag')],
            [('video', 'FollowedBy', 'video'),
                ('tag', 'SubTopic', 'tag'),
                ('video', 'HasTag', 'tag')]],
        message_passer=dict(
            type='TransformerMessagePasser',
            in_dim=768,
            out_dim=768,
            num_heads=8),
        aggregator=dict(
            type='StackAggregator'),
        updator=dict(
            type='GatedUpdator',
            hidden_dim=768,
            dropout=0.2),
        reducer=dict(
            type='CommonUniqueReducer',
            hidden_dim=768,
            dropout=0.2)),
    linker=dict(
        type='OurLinker',
        loss_cls=dict(type='BCELossWithLogits', loss_weight=333.),
        num_layer=2,
        dropout_ratio=0.2,
        label_smooth_eps=0,
        lambda0=0.01,
        gamma=20.0,
        total_step=3040))

# dataset settings
dataset_type = 'RADARTagRecoDataset'
train_fanouts_base = [{
    ('tag', 'HasVideo', 'video'): 0,
    ('tag', 'NotHasVideo', 'video'): 0,
    ('video', 'FollowedBy', 'video'): 3,
    ('tag', 'SubTopic', 'tag'): 3,
    ('video', 'HasTag', 'tag'): 3}]
train_fanouts_final = [{
    ('tag', 'HasVideo', 'video'): 0,
    ('tag', 'NotHasVideo', 'video'): 0,
    ('video', 'FollowedBy', 'video'): 3,
    ('tag', 'SubTopic', 'tag'): 3,
    ('video', 'HasTag', 'tag'): 3}]
infer_fanouts_base = [{
    ('tag', 'WhetherHasVideo', 'video'): 0,
    ('tag', 'HasVideo', 'video'): 0,
    ('tag', 'NotHasVideo', 'video'): 0,
    ('video', 'FollowedBy', 'video'): 3,
    ('tag', 'SubTopic', 'tag'): 3,
    ('video', 'HasTag', 'tag'): 3}]
infer_fanouts_final = [{
    ('tag', 'WhetherHasVideo', 'video'): 0,
    ('tag', 'HasVideo', 'video'): 0,
    ('tag', 'NotHasVideo', 'video'): 0,
    ('video', 'FollowedBy', 'video'): 3,
    ('tag', 'SubTopic', 'tag'): 3,
    ('video', 'HasTag', 'tag'): 3}]
train_fanouts = train_fanouts_base*(num_gnn_layers-1) + train_fanouts_final
infer_fanouts = infer_fanouts_base*(num_gnn_layers-1) + infer_fanouts_final

data = dict(
    videos_per_gpu=1024,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        vertical={{ _base_.vertical }},
        split='train',
        dataset_root={{ _base_.dataset_root }},
        video_emb_dir={{ _base_.video_emb_dir }},
        tag_emb_dir={{ _base_.tag_emb_dir }},
        force_reload=reload_dataset),
    train_dataloader=dict(
        dataset_type='graph',
        collator=dict(
            type='RADARNodeCollator'),
        sampler=dict(
            type='MultiLayerNeighborSampler',
            fanouts=train_fanouts)),
    val=dict(
        type=dataset_type,
        vertical={{ _base_.vertical }},
        split='val',
        dataset_root={{ _base_.dataset_root }},
        video_emb_dir={{ _base_.video_emb_dir }},
        tag_emb_dir={{ _base_.tag_emb_dir }},
        force_reload=reload_dataset),
    val_dataloader=dict(
        dataset_type='graph',
        collator=dict(
            type='RADARNodeCollator'),
        sampler=dict(
            type='MultiLayerNeighborSampler',
            fanouts=infer_fanouts)),
    test=dict(
        type=dataset_type,
        vertical={{ _base_.vertical }},
        split='test',
        dataset_root={{ _base_.dataset_root }},
        video_emb_dir={{ _base_.video_emb_dir }},
        tag_emb_dir={{ _base_.tag_emb_dir }},
        force_reload=reload_dataset),
    test_dataloader=dict(
        dataset_type='graph',
        collator=dict(
            type='RADARNodeCollator'),
        sampler=dict(
            type='MultiLayerNeighborSampler',
            fanouts=infer_fanouts)))

total_epochs = 40