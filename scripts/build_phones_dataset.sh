#!/bin/bash

python build_audio_tf_records.py \
--positive_data_dir '/scratch0/ilya/locDoc/data/TIMIT' \
--train_path '/scratch1/ilya/locDoc/data/phones/3s/train' \
--val_path '/scratch1/ilya/locDoc/data/phones/3s/val' \
--dataset_type 'TIMIT' \
--example_length 48000 \
--threads 8 \
--max_per_record 250 \
--hop_length 160
