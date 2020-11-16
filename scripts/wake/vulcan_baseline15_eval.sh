
CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
--model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/best_loss_eval \
--train_data_root /vulcanscratch/ilyak/data/alexa/v7.15/train_79840 \
--val_data_root /vulcanscratch/ilyak/data/alexa/v7.15/val_19680 \
--val_shift '[0.36857295399586554]' \
--val_center '[0.20140133345357206]' \
--train_shift '[0.35344529429466637]' \
--train_center '[0.19433748148262361]' \
--feature_height 257 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 64 \
--dropout 0.2 \
--network_name Guo_Li_net \
--eval_only \
--warm_start_from=/vulcanscratch/ilyak/experiments/wake_tf_board/baseline_tfspecv715_bs64/export/best_loss_exporter/*

CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
--model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/best_acc_eval \
--train_data_root /vulcanscratch/ilyak/data/alexa/v7.15/train_79840 \
--val_data_root /vulcanscratch/ilyak/data/alexa/v7.15/val_19680 \
--val_shift '[0.36857295399586554]' \
--val_center '[0.20140133345357206]' \
--train_shift '[0.35344529429466637]' \
--train_center '[0.19433748148262361]' \
--feature_height 257 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 64 \
--dropout 0.2 \
--network_name Guo_Li_net \
--eval_only \
--warm_start_from=/vulcanscratch/ilyak/experiments/wake_tf_board/baseline_tfspecv715_bs64/export/best_acc_exporter/*

CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
--model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/best_FAR_eval \
--train_data_root /vulcanscratch/ilyak/data/alexa/v7.15/train_79840 \
--val_data_root /vulcanscratch/ilyak/data/alexa/v7.15/val_19680 \
--val_shift '[0.36857295399586554]' \
--val_center '[0.20140133345357206]' \
--train_shift '[0.35344529429466637]' \
--train_center '[0.19433748148262361]' \
--feature_height 257 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 64 \
--dropout 0.2 \
--network_name Guo_Li_net \
--eval_only \
--warm_start_from=/vulcanscratch/ilyak/experiments/wake_tf_board/baseline_tfspecv715_bs64/export/FAR_exporter/*

CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
--model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/best_MR_eval \
--train_data_root /vulcanscratch/ilyak/data/alexa/v7.15/train_79840 \
--val_data_root /vulcanscratch/ilyak/data/alexa/v7.15/val_19680 \
--val_shift '[0.36857295399586554]' \
--val_center '[0.20140133345357206]' \
--train_shift '[0.35344529429466637]' \
--train_center '[0.19433748148262361]' \
--feature_height 257 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 64 \
--dropout 0.2 \
--network_name Guo_Li_net \
--eval_only \
--warm_start_from=/vulcanscratch/ilyak/experiments/wake_tf_board/baseline_tfspecv715_bs64/export/MR_exporter/*


scp /vulcanscratch/ilyak/experiments/wake_tf_board/best_loss_eval/eval_results.npz ramawks69:/scratch0/ilya/locDownloads/temp_base_v15_eval/best_loss_eval.npz
scp /vulcanscratch/ilyak/experiments/wake_tf_board/best_acc_eval/eval_results.npz ramawks69:/scratch0/ilya/locDownloads/temp_base_v15_eval/best_acc_eval.npz
scp /vulcanscratch/ilyak/experiments/wake_tf_board/best_FAR_eval/eval_results.npz ramawks69:/scratch0/ilya/locDownloads/temp_base_v15_eval/best_FAR_eval.npz
scp /vulcanscratch/ilyak/experiments/wake_tf_board/best_MR_eval/eval_results.npz ramawks69:/scratch0/ilya/locDownloads/temp_base_v15_eval/best_MR_eval.npz

# loss_eval = np.load('/scratch0/ilya/locDownloads/temp_base_v15_eval/best_loss_eval.npz')
# acc_eval = np.load('/scratch0/ilya/locDownloads/temp_base_v15_eval/best_acc_eval.npz')
# FAR_eval = np.load('/scratch0/ilya/locDownloads/temp_base_v15_eval/best_FAR_eval.npz')
# MR_eval = np.load('/scratch0/ilya/locDownloads/temp_base_v15_eval/best_MR_eval.npz')
