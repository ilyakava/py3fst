#!/bin/bash

python build_audio_tf_records.py \
--positive_data_dir '/scratch0/ilya/locDoc/data/alexa/v1/alexa' \
--wakeword_metafile '/cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/data/alexa.annotated.frank+ilya.json' \
--negative_data_dir '/scratch1/ilya/locDoc/data/LibriSpeech/train-clean-360' \
--train_path '/scratch1/ilya/locDoc/data/alexa/v4.logmel/train' \
--val_path '/scratch1/ilya/locDoc/data/alexa/v4.logmel/val' \
--positive_multiplier 1 \
--threads 1 \
--max_per_record 200 \
--hop_length 80