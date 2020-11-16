
CUDA_VISIBLE_DEVICES=1 python spectrogram_NN.py \
--model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/baseline_v717_bs64_trial3 \
--train_data_root /vulcanscratch/ilyak/data/alexa/v7.17/train_79840 \
--val_data_root /vulcanscratch/ilyak/data/alexa/v7.17/val_19680 \
--val_shift '[0.15761846943729804]' \
--val_center '[0.0874684936555642]' \
--train_shift '[0.15263269734976948]' \
--train_center '[0.0845688136921982]' \
--feature_height 128 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 64 \
--dropout 0.2 \
--network_name Guo_Li_net
# --pretrain_checkpoint=/vulcanscratch/ilyak/experiments/phones_tf_board/CBHG_skinny_v0.4_gwn_15dBFS_48000/export/best_loss_exporter/1594436033/variables/variables
