
CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
--model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/baseline_cproj_v7.8 \
--train_data_root /vulcanscratch/ilyak/data/alexa/v7.8/train_79840 \
--val_data_root /vulcanscratch/ilyak/data/alexa/v7.8/val_19680 \
--val_shift -0.00003904 \
--val_center 0.32976197445010447 \
--train_shift -0.00005051 \
--train_center 0.3178966731272077 \
--feature_height 514 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 32 \
--dropout 0.2 \
--network_name Guo_Li_complex_projection_net
# --pretrain_checkpoint=/vulcanscratch/ilyak/experiments/phones_tf_board/CBHG_skinny_v0.4_gwn_15dBFS_48000/export/best_loss_exporter/1594436033/variables/variables
