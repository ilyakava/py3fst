
CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
--model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/baseline_specv714_bs64 \
--train_data_root /vulcanscratch/ilyak/data/alexa/v7.14/train_79840 \
--val_data_root /vulcanscratch/ilyak/data/alexa/v7.14/val_19680 \
--val_shift '[0.3678513547231636]' \
--val_center '[0.20085507533843427]' \
--train_shift '[0.35283666008897874]' \
--train_center '[0.1941610735887028]' \
--feature_height 257 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 64 \
--dropout 0.2 \
--network_name Guo_Li_net
# --pretrain_checkpoint=/vulcanscratch/ilyak/experiments/phones_tf_board/CBHG_skinny_v0.4_gwn_15dBFS_48000/export/best_loss_exporter/1594436033/variables/variables
