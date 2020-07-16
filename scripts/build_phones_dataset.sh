#!/bin/bash
# This is an example script for how to generate data for phoneme pretraining.

python build_audio_tf_records.py \
--positive_data_dir '/scratch0/ilya/locDoc/data/TIMIT' \
--val_path '/scratch1/ilya/locDoc/data/phones/v0.4_spec_hop10_gwn_15dBFS/val_47840' \
--dataset_type 'TIMIT' \
--example_length 47840 \
--threads 8 \
--max_per_record 650 \
--hop_length 160 \
--noise_type gwn

python build_audio_tf_records.py \
--positive_data_dir '/scratch0/ilya/locDoc/data/TIMIT' \
--train_path '/scratch1/ilya/locDoc/data/phones/v0.4_spec_hop10_gwn_15dBFS/train_47840' \
--dataset_type 'TIMIT' \
--example_length 47840 \
--threads 8 \
--max_per_record 650 \
--hop_length 160 \
--noise_type gwn