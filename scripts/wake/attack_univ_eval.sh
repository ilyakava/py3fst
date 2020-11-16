
# export MODEL_DIR=/vulcanscratch/ilyak/experiments/baseline_music_test/
# export MODEL_DIR=/vulcanscratch/ilyak/experiments/wake_tf_board/baseline_v718_bs64_trial2/len124_export/
export MODEL_DIR=/vulcanscratch/ilyak/experiments/wake_tf_board/corticalv0res1c_drop9_v718_bs64_trial2/len124_export/
# export MODEL_DIR=/vulcanscratch/ilyak/experiments/cortical_music_test/


# CUDA_VISIBLE_DEVICES=0 python attack_eval.py \
#     --model_root=$MODEL_DIR \
#     --train_data_root /vulcanscratch/ilyak/data/alexa/v7.18/train_19680 \
#     --val_data_root /vulcanscratch/ilyak/data/alexa/v7.18/val_19680 \
#     --val_shift '[0.19373219918357965]' \
#     --val_center '[0.11091934887483722]' \
#     --tfrecord_eval_feature_width 124 \
#     --batch_size 32 \
#     --load_noise data/first_test.wav \
#     --save_suffix test1


# CUDA_VISIBLE_DEVICES=0 python attack_eval.py \
#     --model_root=$MODEL_DIR \
#     --train_data_root /vulcanscratch/ilyak/data/alexa/v7.18/train_19680 \
#     --val_data_root /vulcanscratch/ilyak/data/alexa/v7.18/val_19680 \
#     --val_shift '[0.19373219918357965]' \
#     --val_center '[0.11091934887483722]' \
#     --tfrecord_eval_feature_width 124 \
#     --batch_size 32 \
#     --n_eval_batches 224 \
#     --load_noise data/clean_tnsprt.wav \
#     --init_noise_rescale 0.012932258228110977 \
#     --save_suffix clean

CUDA_VISIBLE_DEVICES=0 python attack_eval.py \
    --model_root=$MODEL_DIR \
    --train_data_root /vulcanscratch/ilyak/data/alexa/v7.18/train_19680 \
    --val_data_root /vulcanscratch/ilyak/data/alexa/v7.18/val_19680 \
    --val_shift '[0.19373219918357965]' \
    --val_center '[0.11091934887483722]' \
    --tfrecord_eval_feature_width 124 \
    --batch_size 32 \
    --n_eval_batches 224 \
    --load_noise data/sec_basetest.wav \
    --save_suffix dirty
    

rsync -rv /vulcanscratch/ilyak/experiments/wake_tf_board/corticalv0res1c_drop9_v718_bs64_trial2/len124_export/ ramawks69:/scratch0/ilya/locDownloads/cortical_music_adv_test

