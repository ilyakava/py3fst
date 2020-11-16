
CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
--model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/corticalv0res1c_v715_bs64_drop903125_trial2 \
--train_data_root /vulcanscratch/ilyak/data/alexa/v7.15/train_79840 \
--val_data_root /vulcanscratch/ilyak/data/alexa/v7.15/val_19680 \
--val_shift '[0.36857295399586554]' \
--val_center '[0.20140133345357206]' \
--train_shift '[0.35344529429466637]' \
--train_center '[0.19433748148262361]' \
--feature_height 257 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 64 \
--dropout 0.903125 \
--network_name cortical_net_v0res1c
