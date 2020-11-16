
CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
--model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/baseline_specnoclip_bs32_again \
--train_data_root /scratch1/ilya/locDoc/data/alexa/v7.10/train_79840 \
--val_data_root /scratch1/ilya/locDoc/data/alexa/v7.10/val_19680 \
--val_shift '[0.3307385399674817]' \
--val_center '[0.20462007975158264]' \
--train_shift '[0.31457945400645276]' \
--train_center '[0.19395138737221926]' \
--feature_height 257 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 32 \
--dropout 0.2 \
--network_name Guo_Li_net
# --pretrain_checkpoint=/vulcanscratch/ilyak/experiments/phones_tf_board/CBHG_skinny_v0.4_gwn_15dBFS_48000/export/best_loss_exporter/1594436033/variables/variables
