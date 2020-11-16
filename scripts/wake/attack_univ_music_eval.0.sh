
# export MODEL_DIR=/vulcanscratch/ilyak/experiments/adv_music/cort_net_1/len124_export/
# export MODEL_DIR=/vulcanscratch/ilyak/experiments/adv_music/base_net_1/len124_export/

# clean
CUDA_VISIBLE_DEVICES=0 python attack_eval.py \
    --model_root=$MODEL_DIR \
    --train_data_root /vulcanscratch/ilyak/data/alexa/v7.18/train_19680 \
    --val_data_root /vulcanscratch/ilyak/data/alexa/v7.18/val_19680 \
    --val_shift '[0.19373219918357965]' \
    --val_center '[0.11091934887483722]' \
    --tfrecord_eval_feature_width 124 \
    --batch_size 32 \
    --n_eval_batches 224 \
    --load_noise data/clean_tnsprt.wav \
    --init_noise_rescale 0.012079834164302784 \
    --save_suffix clean

# three cortical noises
for nidx in 1 2 3
do
CUDA_VISIBLE_DEVICES=0 python attack_eval.py \
    --model_root=$MODEL_DIR \
    --train_data_root /vulcanscratch/ilyak/data/alexa/v7.18/train_19680 \
    --val_data_root /vulcanscratch/ilyak/data/alexa/v7.18/val_19680 \
    --val_shift '[0.19373219918357965]' \
    --val_center '[0.11091934887483722]' \
    --tfrecord_eval_feature_width 124 \
    --batch_size 32 \
    --n_eval_batches 224 \
    --load_noise data/dirty_cort_${nidx}_t1.wav \
    --save_suffix dirty
done

# three base noises
for nidx in 1 2 3
do
CUDA_VISIBLE_DEVICES=0 python attack_eval.py \
    --model_root=$MODEL_DIR \
    --train_data_root /vulcanscratch/ilyak/data/alexa/v7.18/train_19680 \
    --val_data_root /vulcanscratch/ilyak/data/alexa/v7.18/val_19680 \
    --val_shift '[0.19373219918357965]' \
    --val_center '[0.11091934887483722]' \
    --tfrecord_eval_feature_width 124 \
    --batch_size 32 \
    --n_eval_batches 224 \
    --load_noise data/dirty_base_${nidx}_t1.wav \
    --save_suffix dirty
done
    
