#!/bin/bash

# normal
# python spectrogram_NN.py \
# --network_name Guo_Li_net \
# --model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/normal_bs64_05.02_v7 \
# --train_data_root /scratch1/ilya/locDoc/data/alexa/v7/train_80000 \
# --val_data_root /scratch1/ilya/locDoc/data/alexa/v7/val_19840 \
# --feature_height 256 \
# --lr 0.00005 \
# --tfrecord_train_feature_width 500 \
# --tfrecord_eval_feature_width 124 \
# --batch_size 128 \
# --export_feature_width 31 \
# --export_only_dir /scratch0/ilya/locDoc/adv_audio_exports/len31_byloss_v7data_normal05.02 \
# --warm_start_from /scratch0/ilya/locDoc/adv_audio_tf_board/normal_bs64_05.02_v7/export/best_loss_exporter/*


# export for adv notebook
# CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
# --model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/cortical_bs64_v7.4_trial2 \
# --train_data_root /vulcanscratch/ilyak/data/alexa/v7.4/train_79840 \
# --val_data_root /vulcanscratch/ilyak/data/alexa/v7.4/val_19680 \
# --val_shift 0.3343832678897679 \
# --val_center 0.20798715062082082 \
# --train_shift 0.3246136855279846 \
# --train_center 0.19699797545310258 \
# --feature_height 257 \
# --lr 0.00005 \
# --tfrecord_train_feature_width 500 \
# --tfrecord_eval_feature_width 124 \
# --batch_size 64 \
# --dropout 0.2 \
# --network_name cortical_net_v0 \
# --export_feature_width 31 \
# --export_only_dir=/vulcanscratch/ilyak/experiments/wake_tf_board/cortical_bs64_v7.4_trial2/myexports/len31 \
# --warm_start_from=/vulcanscratch/ilyak/experiments/wake_tf_board/cortical_bs64_v7.4_trial2/export/best_loss_exporter/*


# export for ...
# CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
# --model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/cortical_bs64_v7.4_trial2 \
# --train_data_root /vulcanscratch/ilyak/data/alexa/v7.4/train_79840 \
# --val_data_root /vulcanscratch/ilyak/data/alexa/v7.4/val_19680 \
# --val_shift 0.3343832678897679 \
# --val_center 0.20798715062082082 \
# --train_shift 0.3246136855279846 \
# --train_center 0.19699797545310258 \
# --feature_height 257 \
# --lr 0.00005 \
# --tfrecord_train_feature_width 500 \
# --tfrecord_eval_feature_width 124 \
# --batch_size 64 \
# --dropout 0.2 \
# --network_name cortical_net_v0 \
# --export_feature_width 124 \
# --export_only_dir=/vulcanscratch/ilyak/experiments/wake_tf_board/cortical_bs64_v7.4_trial2/myexports/len124 \
# --warm_start_from=/vulcanscratch/ilyak/experiments/wake_tf_board/cortical_bs64_v7.4_trial2/export/best_loss_exporter/*



# export for ...
CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
--model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/baseline_bs64_v7.4 \
--train_data_root /vulcanscratch/ilyak/data/alexa/v7.4/train_79840 \
--val_data_root /vulcanscratch/ilyak/data/alexa/v7.4/val_19680 \
--val_shift 0.3343832678897679 \
--val_center 0.20798715062082082 \
--train_shift 0.3246136855279846 \
--train_center 0.19699797545310258 \
--feature_height 257 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 64 \
--dropout 0.2 \
--network_name Guo_Li_net \
--export_feature_width 31 \
--export_only_dir=/vulcanscratch/ilyak/experiments/wake_tf_board/baseline_bs64_v7.4/myexports/len31 \
--warm_start_from=/vulcanscratch/ilyak/experiments/wake_tf_board/baseline_bs64_v7.4/export/best_loss_exporter/*
