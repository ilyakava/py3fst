
CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
--model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/best_loss_eval \
--train_data_root /vulcanscratch/ilyak/data/alexa/v7.18/train_79840 \
--val_data_root /vulcanscratch/ilyak/data/alexa/v7.18/val_19680 \
--val_shift '[0.19373219918357965]' \
--val_center '[0.11091934887483722]' \
--train_shift '[0.18767338919868645]' \
--train_center '[0.10696341974853465]' \
--feature_height 256 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 64 \
--dropout 0.2 \
--network_name Guo_Li_net \
--eval_only \
--warm_start_from=/vulcanscratch/ilyak/experiments/wake_tf_board/baseline_v718_bs64_trial3/export/best_loss_exporter/*

CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
--model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/best_acc_eval \
--train_data_root /vulcanscratch/ilyak/data/alexa/v7.18/train_79840 \
--val_data_root /vulcanscratch/ilyak/data/alexa/v7.18/val_19680 \
--val_shift '[0.19373219918357965]' \
--val_center '[0.11091934887483722]' \
--train_shift '[0.18767338919868645]' \
--train_center '[0.10696341974853465]' \
--feature_height 256 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 64 \
--dropout 0.2 \
--network_name Guo_Li_net \
--eval_only \
--warm_start_from=/vulcanscratch/ilyak/experiments/wake_tf_board/baseline_v718_bs64_trial3/export/best_acc_exporter/*

CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
--model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/best_FAR_eval \
--train_data_root /vulcanscratch/ilyak/data/alexa/v7.18/train_79840 \
--val_data_root /vulcanscratch/ilyak/data/alexa/v7.18/val_19680 \
--val_shift '[0.19373219918357965]' \
--val_center '[0.11091934887483722]' \
--train_shift '[0.18767338919868645]' \
--train_center '[0.10696341974853465]' \
--feature_height 256 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 64 \
--dropout 0.2 \
--network_name Guo_Li_net \
--eval_only \
--warm_start_from=/vulcanscratch/ilyak/experiments/wake_tf_board/baseline_v718_bs64_trial3/export/FAR_exporter/*

CUDA_VISIBLE_DEVICES=0 python spectrogram_NN.py \
--model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/best_MR_eval \
--train_data_root /vulcanscratch/ilyak/data/alexa/v7.18/train_79840 \
--val_data_root /vulcanscratch/ilyak/data/alexa/v7.18/val_19680 \
--val_shift '[0.19373219918357965]' \
--val_center '[0.11091934887483722]' \
--train_shift '[0.18767338919868645]' \
--train_center '[0.10696341974853465]' \
--feature_height 256 \
--lr 0.00005 \
--tfrecord_train_feature_width 500 \
--tfrecord_eval_feature_width 124 \
--batch_size 64 \
--dropout 0.2 \
--network_name Guo_Li_net \
--eval_only \
--warm_start_from=/vulcanscratch/ilyak/experiments/wake_tf_board/baseline_v718_bs64_trial3/export/MR_exporter/*


scp /vulcanscratch/ilyak/experiments/wake_tf_board/best_loss_eval/eval_results.npz ramawks69:/scratch0/ilya/locDownloads/baseline_v718_bs64_trial3/best_loss_eval.npz
scp /vulcanscratch/ilyak/experiments/wake_tf_board/best_acc_eval/eval_results.npz ramawks69:/scratch0/ilya/locDownloads/baseline_v718_bs64_trial3/best_acc_eval.npz
scp /vulcanscratch/ilyak/experiments/wake_tf_board/best_FAR_eval/eval_results.npz ramawks69:/scratch0/ilya/locDownloads/baseline_v718_bs64_trial3/best_FAR_eval.npz
scp /vulcanscratch/ilyak/experiments/wake_tf_board/best_MR_eval/eval_results.npz ramawks69:/scratch0/ilya/locDownloads/baseline_v718_bs64_trial3/best_MR_eval.npz
