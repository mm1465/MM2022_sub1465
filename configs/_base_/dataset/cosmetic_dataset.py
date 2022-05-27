vertical = 'cosmetic'
num_classes = 3114
# Configs for GNN methods
dataset_root = '<Path to dataset>/data/anonymous'
data_root = '<Path to dataset>/data/anonymous/frame_feat'
dataset_root = '<Path to dataset>/data/anonymous'
video_emb_dir = 'video_feat'
tag_emb_dir = 'tag_feat_bert'
# Configs for multi-label classification methods
ann_file_train = '<Path to dataset>/data/anonymous/anonymous_cosmetic_video_train_list.txt'
ann_file_val = '<Path to dataset>/data/anonymous/anonymous_cosmetic_video_val_list.txt'
ann_file_test = '<Path to dataset>/data/anonymous/anonymous_cosmetic_video_test_list.txt'
tag_embedding_path='<Path to dataset>/data/anonymous/tag_feat_bert'
tag_graph_path='<Path to dataset>/data/anonymous/tag_parents_cosmetic.json'
label_map_path='<Path to dataset>/data/anonymous/label_map_anonymous_cosmetic.txt'
class_freq_file = '<Path to dataset>/data/anonymous/anonymous_cosmetic_video_train_class_freq.pkl'