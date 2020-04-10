#!/bin/bash

# python build_audio_tf_records.py \
# --positive_data_dir '/scratch0/ilya/locDoc/data/TIMIT' \
# --train_path '/scratch1/ilya/locDoc/data/phones/v0_mfcc_hop5/train_48000' \
# --dataset_type 'TIMIT' \
# --example_length 48000 \
# --threads 8 \
# --max_per_record 500 \
# --hop_length 80

# python build_audio_tf_records.py \
# --positive_data_dir '/scratch0/ilya/locDoc/data/TIMIT' \
# --val_path '/scratch1/ilya/locDoc/data/phones/v0_mfcc_hop5/val_32000' \
# --dataset_type 'TIMIT' \
# --example_length 32000 \
# --threads 8 \
# --max_per_record 750 \
# --hop_length 80

# for spectrogram:
# (625 * 19840 = 100mb)
# (250 * 48000 = 100mb)
# for mfcc

# scp /scratch1/ilya/locDoc/data/phones/v0_mfcc_hop10.tar.gz ilyak@ramawks84:/scratch2/ilyak/locDoc/data/phones/

# python build_audio_tf_records.py \
# --positive_data_dir '/scratch0/ilya/locDoc/data/TIMIT' \
# --train_path '/scratch1/ilya/locDoc/data/phones/v0_mfcc_hop10/train_48000' \
# --dataset_type 'TIMIT' \
# --example_length 48000 \
# --threads 8 \
# --max_per_record 500 \
# --hop_length 160

# python build_audio_tf_records.py \
# --positive_data_dir '/scratch0/ilya/locDoc/data/TIMIT' \
# --val_path '/scratch1/ilya/locDoc/data/phones/v0_mfcc_hop10/val_32000' \
# --dataset_type 'TIMIT' \
# --example_length 32000 \
# --threads 8 \
# --max_per_record 750 \
# --hop_length 160

# scp /scratch1/ilya/locDoc/data/phones/v0_spec_hop5.tar.gz ilyak@ramawks84:/scratch2/ilyak/locDoc/data/phones/

# python build_audio_tf_records.py \
# --positive_data_dir '/scratch0/ilya/locDoc/data/TIMIT' \
# --train_path '/scratch1/ilya/locDoc/data/phones/v0_spec_hop5/train_48000' \
# --dataset_type 'TIMIT' \
# --example_length 48000 \
# --threads 8 \
# --max_per_record 500 \
# --hop_length 80

# python build_audio_tf_records.py \
# --positive_data_dir '/scratch0/ilya/locDoc/data/TIMIT' \
# --val_path '/scratch1/ilya/locDoc/data/phones/v0_spec_hop5/val_32000' \
# --dataset_type 'TIMIT' \
# --example_length 32000 \
# --threads 8 \
# --max_per_record 750 \
# --hop_length 80


python build_audio_tf_records.py \
--positive_data_dir '/scratch0/ilya/locDoc/data/TIMIT' \
--val_path '/scratch1/ilya/locDoc/data/phones/v0_spec_hop10_gwn_15dBFS/val_32000' \
--dataset_type 'TIMIT' \
--example_length 32000 \
--threads 8 \
--max_per_record 650 \
--hop_length 160 \
--noise_type gwn

python build_audio_tf_records.py \
--positive_data_dir '/scratch0/ilya/locDoc/data/TIMIT' \
--val_path '/scratch1/ilya/locDoc/data/phones/v0_spec_hop10_gwn_15dBFS/val_48000' \
--dataset_type 'TIMIT' \
--example_length 48000 \
--threads 8 \
--max_per_record 650 \
--hop_length 160 \
--noise_type gwn

python build_audio_tf_records.py \
--positive_data_dir '/scratch0/ilya/locDoc/data/TIMIT' \
--train_path '/scratch1/ilya/locDoc/data/phones/v0_spec_hop10_gwn_15dBFS/train_64000' \
--dataset_type 'TIMIT' \
--example_length 48000 \
--threads 8 \
--max_per_record 225 \
--hop_length 160 \
--noise_type gwn