#!/bin/bash
# This is an example script for how to train a phone classifier.

CUDA_VISIBLE_DEVICES=0 python spectrogram_phoneme_classifier.py \
--model_root=/vulcan/scratch/ilyak/experiments/phones_tf_board/CBHG_skinny_v0.4_gwn_15dBFS_48000 \
--train_data_root /vulcan/scratch/ilyak/data/phones/v0.4_spec_hop10_gwn_15dBFS/train_47840 \
--val_data_root /vulcan/scratch/ilyak/data/phones/v0.4_spec_hop10_gwn_15dBFS/val_47840 \
--val_shift 0.28499622849111 \
--val_center 0.18031198767476836 \
--train_shift 0.28605913762866186 \
--train_center 0.18103149871955165 \
--feature_height 257 \
--tfrecord_feature_width 300 \
--network_feature_width 300 \
--hop_length 160 \
--batch_size 32 \
--dropout 0.2 \
--lr 0.0003 \
--network_name CBHG_net