#!/bin/bash


# CUDA_VISIBLE_DEVICES=1 python spectrogram_phoneme_classifier.py \
# --model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/CBHG_pretrain_gwn_15dBFS \
# --train_data_root /scratch1/ilya/locDoc/data/phones/v0_spec_hop10_gwn_15dBFS/train_48000 \
# --val_data_root /scratch1/ilya/locDoc/data/phones/v0_spec_hop10_gwn_15dBFS/val_19840 \
# --feature_height 256 \
# --tfrecord_feature_width 300 \
# --network_feature_width 124 \
# --hop_length 160 \
# --batch_size 32 \
# --dropout 0.2 \
# --lr 0.0003 \
# --network_type CBHG

# ramawks84

# # /scratch2/ilyak/locDoc/data/phones

# CUDA_VISIBLE_DEVICES=0 python spectrogram_phoneme_classifier.py \
# --model_root=/scratch2/ilyak/locDoc/adv_audio_tf_board/CBHG_gwn_15dBFS_32000 \
# --train_data_root /scratch2/ilyak/locDoc/data/phones/v0_spec_hop10_gwn_15dBFS/train_48000 \
# --val_data_root /scratch2/ilyak/locDoc/data/phones/v0_spec_hop10_gwn_15dBFS/val_32000 \
# --feature_height 256 \
# --tfrecord_feature_width 300 \
# --network_feature_width 200 \
# --hop_length 160 \
# --batch_size 32 \
# --dropout 0.2 \
# --lr 0.0003 \
# --network_type CBHG

CUDA_VISIBLE_DEVICES=1 python spectrogram_phoneme_classifier.py \
--model_root=/scratch2/ilyak/locDoc/adv_audio_tf_board/CBHG_gwn_15dBFS_48000 \
--train_data_root /scratch2/ilyak/locDoc/data/phones/v0_spec_hop10_gwn_15dBFS/train_48000 \
--val_data_root /scratch2/ilyak/locDoc/data/phones/v0_spec_hop10_gwn_15dBFS/val_48000 \
--feature_height 256 \
--tfrecord_feature_width 300 \
--network_feature_width 300 \
--hop_length 160 \
--batch_size 32 \
--dropout 0.2 \
--lr 0.0003 \
--network_type CBHG
