./tools/dist_train.sh configs/tagging/ourgnn/ourgnn_l2_apparel_wo_gated.py 1 --validate --test-best
./tools/dist_train.sh configs/tagging/ourgnn/ourgnn_l2_cosmetic_wo_gated.py 1 --validate --test-best
./tools/dist_train.sh configs/tagging/ourgnn/ourgnn_l2_food_wo_gated.py 1 --validate --test-best
./tools/dist_train.sh configs/tagging/ourgnn/ourgnn_l2_apparel_mutual_attention.py 1 --validate --test-best
./tools/dist_train.sh configs/tagging/ourgnn/ourgnn_l2_cosmetic_mutual_attention.py 1 --validate --test-best
./tools/dist_train.sh configs/tagging/ourgnn/ourgnn_l2_food_mutual_attention.py 1 --validate --test-best