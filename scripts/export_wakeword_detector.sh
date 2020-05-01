#!/bin/bash

# say you trained via:

# python spectrogram_NN.py \
# --model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/simple_prenet_bs_128_04.16_v6cx2 \
# --train_data_root /scratch1/ilya/locDoc/data/alexa/v6cx2/train_80000 \
# --val_data_root /scratch1/ilya/locDoc/data/alexa/v6c/val_19840 \
# --feature_height 256 \
# --lr 0.00005 \
# --tfrecord_train_feature_width 500 \
# --tfrecord_eval_feature_width 124 \
# --batch_size 128 

# then export a specific model with:

python spectrogram_NN.py \
--model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/simple_prenet_bs_128_04.16_v6cx2 \
--train_data_root /scratch1/ilya/locDoc/data/alexa/v6cx2/train_80000 \
--val_data_root /scratch1/ilya/locDoc/data/alexa/v6c/val_19840 \
--feature_height 256 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 128 \
--export_feature_width 500 \
--export_only_dir '/scratch0/ilya/locDoc/adv_audio_exports/len500_loss' \
--warm_start_from /scratch0/ilya/locDoc/adv_audio_tf_board/simple_prenet_bs_128_04.16_v6cx2/export/best_loss_exporter/1587159942
