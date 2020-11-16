
CUDA_VISIBLE_DEVICES=1 python spectrogram_phoneme_classifier.py \
--model_root=/scratch2/ilyak/locDoc/adv_audio_tf_board/baseline_v0.2_gwn_15dBFS_48000 \
--train_data_root /scratch2/ilyak/locDoc/data/phones/v0.2_spec_hop10_gwn_15dBFS/train_48000 \
--val_data_root /scratch2/ilyak/locDoc/data/phones/v0.2_spec_hop10_gwn_15dBFS/val_48000 \
--feature_height 257 \
--tfrecord_feature_width 300 \
--network_feature_width 300 \
--hop_length 160 \
--batch_size 32 \
--dropout 0.2 \
--lr 0.0003 \
--network_name Guo_Li_net