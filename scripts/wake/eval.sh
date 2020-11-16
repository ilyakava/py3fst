
# CUDA_VISIBLE_DEVICES=1 python spectrogram_NN.py \
# --model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/blank \
# --train_data_root /scratch1/ilya/locDoc/data/alexa/v7.4/train_79840 \
# --val_data_root /scratch1/ilya/locDoc/data/alexa/v7.4/val_19680 \
# --val_shift 0.3343832678897679 \
# --val_center 0.20798715062082082 \
# --train_shift 0.3246136855279846 \
# --train_center 0.19699797545310258 \
# --feature_height 257 \
# --lr 0.00005 \
# --tfrecord_train_feature_width 500 \
# --tfrecord_eval_feature_width 124 \
# --batch_size 64 \
# --dropout 0.2 \
# --network_name Guo_Li_net \
# --eval_only \
# --warm_start_from=/scratch0/ilya/locDoc/adv_audio_tf_board/baseline_bs64_v7.4/export/best_loss_exporter/*


CUDA_VISIBLE_DEVICES=1 python spectrogram_NN.py \
--model_root=/scratch0/ilya/locDoc/adv_audio_tf_board/blank \
--train_data_root /scratch1/ilya/locDoc/data/alexa/v7.4/train_79840 \
--val_data_root /scratch1/ilya/locDoc/data/alexa/v7.4/val_19680 \
--val_shift 0.3343832678897679 \
--val_center 0.20798715062082082 \
--train_shift 0.3246136855279846 \
--train_center 0.19699797545310258 \
--feature_height 257 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 64 \
--dropout 0.2 \
--network_name cortical_net_v0 \
--eval_only \
--warm_start_from=/scratch0/ilya/locDoc/adv_audio_tf_board/cortical_bs64_v7.4_trial2/export/best_loss_exporter/*
