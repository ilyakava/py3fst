#!/bin/bash

python build_audio_tf_records.py \
--positive_data_dir '/scratch0/ilya/locDoc/data/alexa/v1/alexa' \
--wakeword_metafile '/cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/data/alexa.annotated.json' \
--negative_data_dir '/scratch1/ilya/locDoc/data/LibriSpeech/train-clean-360' \
--train_path '/scratch1/ilya/locDoc/data/alexa/v6b/train_80000' \
--example_length 80000 \
--positive_multiplier 60 \
--threads 8 \
--max_per_record 150 \
--hop_length 160

# python build_audio_tf_records.py \
# --positive_data_dir '/scratch0/ilya/locDoc/data/alexa/v1/alexa' \
# --wakeword_metafile '/cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/data/alexa.annotated.json' \
# --negative_data_dir '/scratch1/ilya/locDoc/data/LibriSpeech/train-clean-360' \
# --val_path '/scratch1/ilya/locDoc/data/alexa/v6/val_19840' \
# --example_length 19840 \
# --positive_multiplier 100 \
# --negative_version_percentage 0.8 \
# --threads 8 \
# --max_per_record 150 \
# --hop_length 160
