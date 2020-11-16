
# export MODEL_DIR=/vulcanscratch/ilyak/experiments/baseline_music_test/
# export MODEL_DIR=/vulcanscratch/ilyak/experiments/adv_music/cort_net_1/

for part in 0 1 2
do
CUDA_VISIBLE_DEVICES=1 python attack_univ_music.py \
    --model_root=$MODEL_DIR \
    --max_steps 400 \
    --eval_period 40 \
    --train_data_root /vulcanscratch/ilyak/data/alexa/v7.18/train_19680 \
    --val_data_root /vulcanscratch/ilyak/data/alexa/v7.18/val_19680_positive_x8 \
    --train_shift '[0.19943352013519514]' \
    --train_center '[0.11222272143962694]' \
    --eval_steps 200 \
    --attacker deepfool \
    --attack_norm inf \
    --attack_eps 0.001 \
    --init_noise data/nirvana_chord${part}.wav \
    --init_noise_snr 20.0 \
    --save_suffix nirvanachord${part}_t1
done

# rsync -rv /vulcanscratch/ilyak/experiments/baseline_music_test/ ramawks69:/scratch0/ilya/locDownloads/baseline_music_test
# rsync -rv /vulcanscratch/ilyak/experiments/cortical_music_test/ ramawks69:/scratch0/ilya/locDownloads/cortical_music_test
