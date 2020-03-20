#!/bin/bash

# rm -rf /scratch0/ilya/locDoc/adv_audio_tf_board/test;
# mkdir /scratch0/ilya/locDoc/adv_audio_tf_board/test;

# CUDA_VISIBLE_DEVICES=1 python spectrogram_NN.py \
# --model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/logmel \
# --train_data_root /scratch1/ilya/locDoc/data/alexa/v4.logmel/train/ \
# --val_data_root /scratch1/ilya/locDoc/data/alexa/v4.logmel/val/ \
# --feature_height 80 \
# --hop_length 80

CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
--model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/spectrogram03.20.20 \
--train_data_root /scratch1/ilya/locDoc/data/alexa/v4e/train/ \
--val_data_root /scratch1/ilya/locDoc/data/alexa/v4e/val/ \
--feature_height 256 \
--hop_length 160
