
CUDA_VISIBLE_DEVICES=0 python spectrogram_phoneme_classifier.py \
--model_root=/vulcanscratch/ilyak/experiments/phones_tf_board/CBHG_v0.5_gwn_15dBFS_48000 \
--train_data_root /vulcanscratch/ilyak/data/phones/v0.5_spec_hop10_gwn_15dBFS/train_47840 \
--val_data_root /vulcanscratch/ilyak/data/phones/v0.5_spec_hop10_gwn_15dBFS/val_47840 \
--val_shift 0.04482328764215671 \
--val_center 0.12939575666867772 \
--train_shift 0.045172331416151845 \
--train_center 0.12910501353178505 \
--feature_height 257 \
--tfrecord_feature_width 300 \
--network_feature_width 300 \
--hop_length 160 \
--batch_size 32 \
--dropout 0.2 \
--lr 0.0003 \
--network_name CBHG_net