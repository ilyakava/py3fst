#!/bin/bash

python spectrogram_NN.py \
--model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/simple_prenet_04.15_v6 \
--train_data_root /scratch1/ilya/locDoc/data/alexa/v6b/train_80000 \
--val_data_root /scratch1/ilya/locDoc/data/alexa/v6/val_19840 \
--feature_height 256 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 128
