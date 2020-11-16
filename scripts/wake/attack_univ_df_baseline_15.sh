
# 0.0025 0.005 0.0075 0.01 0.02

for eps in 0.0025 0.005 0.0075 0.01 0.02
do
    CUDA_VISIBLE_DEVICES=0 python attack_univ.py \
    --model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/baseline_tfspecv715_bs64/export/best_loss_exporter/ \
    --attack_norm inf \
    --attack_eps $eps \
    --attack_eps_step 0.0001 \
    --max_steps 4000 \
    --eval_period 400 \
    --train_data_root /vulcanscratch/ilyak/data/alexa/v7.16/train_19680 \
    --val_data_root /vulcanscratch/ilyak/data/alexa/v7.16/val_19680_positive_x8 \
    --train_shift '[0.3729283923166141]' \
    --train_center '[0.20274774105000434]' \
    --eval_steps 200
done

rsync -v /vulcanscratch/ilyak/experiments/wake_tf_board/baseline_tfspecv715_bs64/export/best_loss_exporter/*npz ramawks69:/scratch0/ilya/locDownloads/baseline_tfspecv715_bs64/