
# generate without real-world background noise
# python build_audio_tf_records.py \
# --positive_data_dir '/scratch0/ilya/locDoc/data/alexa/v1/alexa' \
# --negative_data_dir '/scratch1/ilya/locDoc/data/LibriSpeech/train-clean-360' \
# --val_path '/scratch1/ilya/locDoc/data/alexa/v7.4/val_19680_testing' \
# --example_length 19680 \
# --positive_multiplier 2 \
# --negative_version_percentage 0 \
# --threads 1 \
# --max_per_record 400 \
# --hop_length 160 \
# --feature_type dft

# python build_audio_tf_records.py \
# --positive_data_dir '/scratch0/ilya/locDoc/data/alexa/v1/alexa' \
# --negative_data_dir '/scratch1/ilya/locDoc/data/LibriSpeech/train-clean-360' \
# --val_path '/scratch1/ilya/locDoc/data/alexa/v7.9/val_19680' \
# --demand_data_dir '/scratch1/ilya/locDoc/data/demand_cut/80000' \
# --example_length 19680 \
# --positive_multiplier 100 \
# --negative_version_percentage 0.8 \
# --threads 12 \
# --max_per_record 400 \
# --hop_length 160 \
# --feature_type polar

# python build_audio_tf_records.py \
# --positive_data_dir '/scratch0/ilya/locDoc/data/alexa/v1/alexa' \
# --negative_data_dir '/scratch1/ilya/locDoc/data/LibriSpeech/train-clean-360' \
# --val_path '/scratch1/ilya/locDoc/data/alexa/v7.16/val_19680' \
# --demand_data_dir '/scratch1/ilya/locDoc/data/demand_cut/80000' \
# --example_length 19680 \
# --positive_multiplier 100 \
# --negative_version_percentage 0.8 \
# --threads 12 \
# --max_per_record 400 \
# --hop_length 160 \
# --feature_type spectrogram_tf

# python build_audio_tf_records.py \
# --positive_data_dir '/scratch0/ilya/locDoc/data/alexa/v1/alexa' \
# --negative_data_dir '/scratch1/ilya/locDoc/data/LibriSpeech/train-clean-360' \
# --train_path '/scratch1/ilya/locDoc/data/alexa/v7.16/train_79840' \
# --demand_data_dir '/scratch1/ilya/locDoc/data/demand_cut/80000' \
# --example_length 79840 \
# --positive_multiplier 40 \
# --threads 12 \
# --max_per_record 100 \
# --hop_length 160 \
# --feature_type spectrogram_tf

# python build_audio_tf_records.py \
# --positive_data_dir '/scratch0/ilya/locDoc/data/alexa/v1/alexa' \
# --negative_data_dir '/scratch1/ilya/locDoc/data/LibriSpeech/train-clean-360' \
# --val_path '/scratch1/ilya/locDoc/data/alexa/v7.16/val_21280_positive' \
# --demand_data_dir '/scratch1/ilya/locDoc/data/demand_cut/80000' \
# --example_length 21280 \
# --positive_multiplier 1 \
# --negative_version_percentage 0.0 \
# --threads 1 \
# --max_per_record 400 \
# --hop_length 160 \
# --feature_type spectrogram_tf


# CUDA_VISIBLE_DEVICES=0 python build_audio_tf_records.py \
# --positive_data_dir '/scratch0/ilya/locDoc/data/alexa/v1/alexa' \
# --negative_data_dir '/scratch1/ilya/locDoc/data/LibriSpeech/train-clean-360' \
# --train_path '/scratch1/ilya/locDoc/data/alexa/v7.18/train_19680' \
# --demand_data_dir '/scratch1/ilya/locDoc/data/demand_cut/80000' \
# --example_length 19680 \
# --positive_multiplier 40 \
# --negative_version_percentage 0.8 \
# --threads 12 \
# --max_per_record 400 \
# --hop_length 160 \
# --feature_type LFBE_tf

# CUDA_VISIBLE_DEVICES=0 python build_audio_tf_records.py \
# --positive_data_dir '/scratch0/ilya/locDoc/data/alexa/v1/alexa' \
# --negative_data_dir '/scratch1/ilya/locDoc/data/LibriSpeech/train-clean-360' \
# --train_path '/scratch1/ilya/locDoc/data/alexa/v7.16/train_smaller_21280' \
# --demand_data_dir '/scratch1/ilya/locDoc/data/demand_cut/80000' \
# --example_length 21280 \
# --positive_multiplier 4 \
# --negative_version_percentage 0.5 \
# --threads 12 \
# --max_per_record 400 \
# --hop_length 160 \
# --feature_type spectrogram_tf

# CUDA_VISIBLE_DEVICES=0 python build_audio_tf_records.py \
# --positive_data_dir '/scratch0/ilya/locDoc/data/alexa/v1/alexa' \
# --negative_data_dir '/scratch1/ilya/locDoc/data/LibriSpeech/train-clean-360' \
# --val_path '/scratch1/ilya/locDoc/data/alexa/v7.18/val_19680_positive_x8' \
# --demand_data_dir '/scratch1/ilya/locDoc/data/demand_cut/80000' \
# --example_length 19680 \
# --positive_multiplier 8 \
# --negative_version_percentage 0.0 \
# --threads 12 \
# --max_per_record 400 \
# --hop_length 160 \
# --feature_type LFBE_tf

CUDA_VISIBLE_DEVICES=0 python build_audio_tf_records.py \
--positive_data_dir '/scratch0/ilya/locDoc/data/alexa/v1/alexa' \
--negative_data_dir '/scratch1/ilya/locDoc/data/LibriSpeech/train-clean-360' \
--val_path '/scratch1/ilya/locDoc/data/alexa/v7.18/val_19680_20hr' \
--demand_data_dir '/scratch1/ilya/locDoc/data/demand_cut/80000' \
--example_length 19680 \
--positive_multiplier 800 \
--negative_version_percentage 0.8 \
--threads 12 \
--max_per_record 400 \
--hop_length 160 \
--feature_type LFBE_tf


# CUDA_VISIBLE_DEVICES=0 python build_audio_tf_records.py \
# --positive_data_dir '/scratch0/ilya/locDoc/data/alexa/v1/alexa' \
# --negative_data_dir '/scratch1/ilya/locDoc/data/LibriSpeech/train-clean-360' \
# --train_path '/scratch1/ilya/locDoc/data/alexa/v7.18/train_79840' \
# --demand_data_dir '/scratch1/ilya/locDoc/data/demand_cut/80000' \
# --example_length 79840 \
# --positive_multiplier 40 \
# --negative_version_percentage 0.8 \
# --threads 12 \
# --max_per_record 400 \
# --hop_length 160 \
# --feature_type LFBE_tf

# CUDA_VISIBLE_DEVICES=0 python build_audio_tf_records.py \
# --positive_data_dir '/scratch0/ilya/locDoc/data/alexa/v1/alexa' \
# --negative_data_dir '/scratch1/ilya/locDoc/data/LibriSpeech/train-clean-360' \
# --val_path '/scratch1/ilya/locDoc/data/alexa/v7.18/val_40800_positive_x1' \
# --demand_data_dir '/scratch1/ilya/locDoc/data/demand_cut/80000' \
# --example_length 40800 \
# --positive_multiplier 1 \
# --negative_version_percentage 0.0 \
# --threads 1 \
# --max_per_record 400 \
# --hop_length 160 \
# --feature_type LFBE_tf
