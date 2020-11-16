
CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
--model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/corticalv0res1cmini_drop2_v718_bs64 \
--train_data_root /vulcanscratch/ilyak/data/alexa/v7.18/train_79840 \
--val_data_root /vulcanscratch/ilyak/data/alexa/v7.18/val_19680 \
--val_shift '[0.19373219918357965]' \
--val_center '[0.11091934887483722]' \
--train_shift '[0.18767338919868645]' \
--train_center '[0.10696341974853465]' \
--feature_height 256 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 64 \
--dropout 0.2 \
--network_name cortical_net_v0res1cmini
# --pretrain_checkpoint=/vulcanscratch/ilyak/experiments/phones_tf_board/CBHG_skinny_v0.4_gwn_15dBFS_48000/export/best_loss_exporter/1594436033/variables/variables
