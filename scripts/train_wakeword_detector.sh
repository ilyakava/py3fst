#!/bin/bash

# rm -rf /scratch0/ilya/locDoc/adv_audio_tf_board/test;
# mkdir /scratch0/ilya/locDoc/adv_audio_tf_board/test;

# CUDA_VISIBLE_DEVICES=1 python spectrogram_NN.py \
# --model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/logmel \
# --train_data_root /scratch1/ilya/locDoc/data/alexa/v4.logmel/train/ \
# --val_data_root /scratch1/ilya/locDoc/data/alexa/v4.logmel/val/ \
# --feature_height 80 \
# --hop_length 80

CUDA_VISIBLE_DEVICES=1 python spectrogram_NN.py \
--model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/spectrogram03.26 \
--train_data_root /scratch1/ilya/locDoc/data/alexa/v4g/train_80000 \
--val_data_root /scratch1/ilya/locDoc/data/alexa/v4g/val_19840 \
--feature_height 256 \
--hop_length 160
