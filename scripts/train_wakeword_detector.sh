#!/bin/bash

# rm -rf /scratch0/ilya/locDoc/adv_audio_tf_board/test;
# mkdir /scratch0/ilya/locDoc/adv_audio_tf_board/test;

# CUDA_VISIBLE_DEVICES=1 python spectrogram_NN.py \
# --model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/logmel \
# --train_data_root /scratch1/ilya/locDoc/data/alexa/v4.logmel/train/ \
# --val_data_root /scratch1/ilya/locDoc/data/alexa/v4.logmel/val/ \
# --feature_height 80 \
# --hop_length 80

# CUDA_VISIBLE_DEVICES=1 python spectrogram_NN.py \
# --model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/wake_CBHBH \
# --train_data_root /scratch1/ilya/locDoc/data/alexa/v5/train_80000 \
# --val_data_root /scratch1/ilya/locDoc/data/alexa/v5/val_19840 \
# --feature_height 256 \
# --hop_length 160

# CUDA_VISIBLE_DEVICES=1 python spectrogram_NN.py \
# --model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/test \
# --train_data_root /scratch1/ilya/locDoc/data/alexa/v5b/train_80000 \
# --val_data_root /scratch1/ilya/locDoc/data/alexa/v5b/val_19840 \
# --feature_height 256 \
# --hop_length 160 \
# --pretrain_checkpoint /scratch0/ilya/locDoc/adv_audio_tf_board/CBHG_pretrain_gwn/export/best_acc_exporter/1585776862/variables/variables


# CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
# --model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/spectrogram03.26 \
# --train_data_root /scratch1/ilya/locDoc/data/alexa/v5/train_80000 \
# --val_data_root /scratch1/ilya/locDoc/data/alexa/v5c/val_19840 \
# --feature_height 256 \
# --hop_length 160 \
# --eval_only

# CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
# --model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/CBHBH_v5d_new \
# --train_data_root /scratch1/ilya/locDoc/data/alexa/v5d/train_80000 \
# --val_data_root /scratch1/ilya/locDoc/data/alexa/v5c/val_19840 \
# --feature_height 256 \
# --hop_length 160


# CUDA_VISIBLE_DEVICES=1 python spectrogram_NN.py \
# --model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/amazon_net_04.02_v5c \
# --train_data_root /scratch1/ilya/locDoc/data/alexa/v5c/train_80000 \
# --val_data_root /scratch1/ilya/locDoc/data/alexa/v5c/val_19840 \
# --feature_height 256 \
# --hop_length 160

CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
--model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/amazon_net_04.02_v5c_CBHBH_transfer2 \
--train_data_root /scratch1/ilya/locDoc/data/alexa/v5c/train_80000 \
--val_data_root /scratch1/ilya/locDoc/data/alexa/v5c/val_19840 \
--feature_height 256 \
--hop_length 160 \
--lr 0.00005 \
--pretrain_checkpoint /scratch0/ilya/locDoc/adv_audio_tf_board/CBHG_pretrain_gwn_15dBFS/export/best_acc_exporter/1585858706/variables/variables

# ramawks84

# CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
# --model_root=/scratch2/ilyak/locDoc/adv_audio_tf_board/CBHBH_v5c \
# --train_data_root /scratch2/ilyak/locDoc/data/alexa/v5c/train_80000 \
# --val_data_root /scratch2/ilyak/locDoc/data/alexa/v5c/val_19840 \
# --feature_height 256 \
# --hop_length 160