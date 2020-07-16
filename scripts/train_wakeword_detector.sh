#!/bin/bash
# This is an example script for how to train a wakeword detector.

CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
--model_root=/vulcan/scratch/ilyak/experiments/wake_tf_board/baseline_bs64_v7.4_warm_trial2 \
--train_data_root /vulcan/scratch/ilyak/data/alexa/v7.4/train_79840 \
--val_data_root /vulcan/scratch/ilyak/data/alexa/v7.4/val_19680 \
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
--pretrain_checkpoint=/vulcan/scratch/ilyak/experiments/phones_tf_board/CBHG_skinny_v0.4_gwn_15dBFS_48000/export/best_loss_exporter/1594436033/variables/variables
