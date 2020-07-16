#!/bin/bash
# This is an example script for how to train a phone classifier


CUDA_VISIBLE_DEVICES=0 python spectrogram_phoneme_classifier.py \
--model_root=/vulcan/scratch/ilyak/experiments/phones_tf_board/CBHG_skinny_v0.3_gwn_15dBFS_48000 \
--train_data_root /vulcan/scratch/ilyak/data/phones/v0.3_spec_hop10_gwn_15dBFS/train_47840 \
--val_data_root /vulcan/scratch/ilyak/data/phones/v0.3_spec_hop10_gwn_15dBFS/val_47840 \
--val_shift 0.058074123157292545 \
--val_center 0.242303276164229 \
--train_shift 0.05836660828353991 \
--train_center 0.2406181628816118 \
--feature_height 257 \
--tfrecord_feature_width 300 \
--network_feature_width 300 \
--hop_length 160 \
--batch_size 32 \
--dropout 0.2 \
--lr 0.0003 \
--network_name CBHG_net