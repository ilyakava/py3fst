


# eps, snr, base acc
# 0.0025, 26, 84
# 0.005, 20, 67
# 0.0075, 18, 28, 49
# 0.01, 15, __, 8
# 0.02, 12, 1

for eps in 0.005 0.01 0.02
do
    CUDA_VISIBLE_DEVICES=0 python attack_univ.py \
    --model_root=/vulcanscratch/ilyak/experiments/wake_tf_board/corticalv0res1c_drop9_v718_bs64/export/best_loss_exporter/ \
    --max_steps 4000 \
    --eval_period 400 \
    --train_data_root /vulcanscratch/ilyak/data/alexa/v7.18/train_19680 \
    --val_data_root /vulcanscratch/ilyak/data/alexa/v7.18/val_19680_positive_x8 \
    --train_shift '[0.19943352013519514]' \
    --train_center '[0.11222272143962694]' \
    --eval_steps 200 \
    --attacker carlini_inf \
    --attack_norm inf \
    --attack_eps $eps \
    --save_suffix n1t1
done

rsync /vulcanscratch/ilyak/experiments/wake_tf_board/corticalv0res1c_drop9_v718_bs64_trial2/export/best_loss_exporter/*npz ramawks69:/scratch0/ilya/locDownloads/corticalv0res1c_drop9_v718_bs64_trial2/