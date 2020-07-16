#!/bin/bash
#This is an example script for creating wakeword training data.

python build_audio_tf_records.py \
--positive_data_dir '/scratch0/ilya/locDoc/data/alexa/v1/alexa' \
--negative_data_dir '/scratch1/ilya/locDoc/data/LibriSpeech/train-clean-360' \
--train_path '/scratch1/ilya/locDoc/data/alexa/v7.4/train_79840' \
--demand_data_dir '/scratch1/ilya/locDoc/data/demand_cut/80000' \
--example_length 79840 \
--positive_multiplier 40 \
--threads 8 \
--max_per_record 100 \
--hop_length 160

python build_audio_tf_records.py \
--positive_data_dir '/scratch0/ilya/locDoc/data/alexa/v1/alexa' \
--negative_data_dir '/scratch1/ilya/locDoc/data/LibriSpeech/train-clean-360' \
--val_path '/scratch1/ilya/locDoc/data/alexa/v7.4/val_19680' \
--demand_data_dir '/scratch1/ilya/locDoc/data/demand_cut/80000' \
--example_length 19680 \
--positive_multiplier 100 \
--negative_version_percentage 0.8 \
--threads 8 \
--max_per_record 400 \
--hop_length 160

# generate without real-world background noise
python build_audio_tf_records.py \
--positive_data_dir '/scratch0/ilya/locDoc/data/alexa/v1/alexa' \
--negative_data_dir '/scratch1/ilya/locDoc/data/LibriSpeech/train-clean-360' \
--val_path '/scratch1/ilya/locDoc/data/alexa/v7.4/val_19680_easy' \
--example_length 19680 \
--positive_multiplier 20 \
--negative_version_percentage 0.5 \
--threads 8 \
--max_per_record 400 \
--hop_length 160
