#!/bin/bash

python build_audio_tf_records.py \
--positive_data_dir '/scratch0/ilya/locDoc/data/TIMIT' \
--train_path '/scratch1/ilya/locDoc/data/phones/v0b/train_48000' \
--dataset_type 'TIMIT' \
--example_length 48000 \
--threads 8 \
--max_per_record 250 \
--hop_length 160

python build_audio_tf_records.py \
--positive_data_dir '/scratch0/ilya/locDoc/data/TIMIT' \
--val_path '/scratch1/ilya/locDoc/data/phones/v0b/val_19840' \
--dataset_type 'TIMIT' \
--example_length 19840 \
--threads 8 \
--max_per_record 625 \
--hop_length 160