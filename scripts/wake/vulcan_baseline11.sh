
CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
--model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/baseline_dft_v7.11 \
--train_data_root /vulcanscratch/ilyak/data/alexa/v7.11/train_79840 \
--val_data_root /vulcanscratch/ilyak/data/alexa/v7.11/val_19680 \
--val_shift '[-7.02042586e-05, -4.13042624e-05]' \
--val_center '[0.34443776, 0.31706583]' \
--train_shift '[-7.73664179e-05, -1.30330191e-05]' \
--train_center '[0.32752631, 0.30228321]' \
--feature_height 514 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 32 \
--dropout 0.2 \
--network_name Guo_Li_net
# --pretrain_checkpoint=/vulcanscratch/ilyak/experiments/phones_tf_board/CBHG_skinny_v0.4_gwn_15dBFS_48000/export/best_loss_exporter/1594436033/variables/variables
