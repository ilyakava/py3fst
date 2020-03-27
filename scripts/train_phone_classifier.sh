#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python spectrogram_phoneme_classifier.py \
--model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/phoneme_HWB_03.26 \
--train_data_root /scratch1/ilya/locDoc/data/phones/v0b/train_48000 \
--val_data_root /scratch1/ilya/locDoc/data/phones/v0b/val_19840 \
--feature_height 256 \
--hop_length 160 \
--tfrecord_example_length 48000 \
--network_example_length 19840 \
--batch_size 32
