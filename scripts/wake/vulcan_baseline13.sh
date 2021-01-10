
CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
--model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/baseline_tfspecnoclipv713_bs64 \
--train_data_root /vulcanscratch/ilyak/data/alexa/v7.13/train_79840 \
--val_data_root /vulcanscratch/ilyak/data/alexa/v7.13/val_19680 \
--val_shift '[0.33055328296157355]' \
--val_center '[0.20434429029329348]' \
--train_shift '[0.31443345065307565]' \
--train_center '[0.19389327565109943]' \
--feature_height 257 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 64 \
--dropout 0.2 \
--network_name Guo_Li_net
# --pretrain_checkpoint=/vulcanscratch/ilyak/experiments/phones_tf_board/CBHG_skinny_v0.4_gwn_15dBFS_48000/export/best_loss_exporter/1594436033/variables/variables