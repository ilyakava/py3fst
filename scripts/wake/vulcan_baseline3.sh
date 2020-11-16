
CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
--model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/baseline_bs64_v7.3 \
--train_data_root /vulcanscratch/ilyak/data/alexa/v7.3/train_79840 \
--val_data_root /vulcanscratch/ilyak/data/alexa/v7.3/val_19680 \
--val_shift 0.12030826327173419 \
--val_center 0.8774376376774593 \
--train_shift 0.10114524341384433 \
--train_center 0.7045985991327786 \
--feature_height 257 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 64 \
--dropout 0.2 \
--network_name Guo_Li_net
