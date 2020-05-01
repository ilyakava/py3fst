#!/bin/bash

python spectrogram_NN.py \
--model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/cortical_04.30_v6cx2 \
--train_data_root /scratch1/ilya/locDoc/data/alexa/v6cx2/train_80000 \
--val_data_root /scratch1/ilya/locDoc/data/alexa/v6c/val_19840 \
--feature_height 256 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 64
# --eval_only
# --pretrain_checkpoint /scratch0/ilya/locDoc/adv_audio_tf_board/CBHG_pretrain_gwn_15dBFS/export/best_acc_exporter/1585858706/variables/variables
