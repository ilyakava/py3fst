#!/bin/bash

python build_audio_tf_records.py \
--positive_data_dir '/scratch0/ilya/locDoc/data/alexa/v1/alexa' \
--negative_data_dir '/scratch1/ilya/locDoc/data/LibriSpeech/train-clean-100' \
--train_path '/scratch0/ilya/locDoc/data/alexa/v2/train' \
--val_path '/scratch0/ilya/locDoc/data/alexa/v2/val' \
--positive_multiplier 1 \
--threads 1