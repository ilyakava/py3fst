# sh scripts/wake/attack_univ_all_18_pgd_more.sh
# export MODEL_DIR=/vulcanscratch/ilyak/experiments/wake_tf_board/corticalv0res1c_drop9_v718_bs64_trial3/export/best_loss_exporter/
# export MODEL_DIR=/vulcanscratch/ilyak/experiments/wake_tf_board/baseline_v718_bs64_trial3/export/best_loss_exporter/
# export SAVE_SUFFIX=n2t4

for eps in 0.0025 0.00375 0.005 0.0075 0.01 0.015 0.02
do
    CUDA_VISIBLE_DEVICES=0 python attack_univ.py \
    --model_root=$MODEL_DIR \
    --max_steps 4000 \
    --eval_period 100 \
    --train_data_root /vulcanscratch/ilyak/data/alexa/v7.18/train_19680 \
    --val_data_root /vulcanscratch/ilyak/data/alexa/v7.18/val_19680_positive_x8 \
    --train_shift '[0.19943352013519514]' \
    --train_center '[0.11222272143962694]' \
    --eval_steps 200 \
    --attacker pgd \
    --attack_norm inf \
    --attack_eps $eps \
    --save_suffix ttt2
done

for eps in 0.0025 0.00375 0.005 0.0075 0.01 0.015 0.02
do
    CUDA_VISIBLE_DEVICES=0 python attack_univ.py \
    --model_root=$MODEL_DIR \
    --max_steps 4000 \
    --eval_period 100 \
    --train_data_root /vulcanscratch/ilyak/data/alexa/v7.18/train_19680 \
    --val_data_root /vulcanscratch/ilyak/data/alexa/v7.18/val_19680_positive_x8 \
    --train_shift '[0.19943352013519514]' \
    --train_center '[0.11222272143962694]' \
    --eval_steps 200 \
    --attacker pgd \
    --attack_norm inf \
    --attack_eps $eps \
    --save_suffix ttt3
done

for eps in 0.0025 0.00375 0.005 0.0075 0.01 0.015 0.02
do
    CUDA_VISIBLE_DEVICES=0 python attack_univ.py \
    --model_root=$MODEL_DIR \
    --max_steps 4000 \
    --eval_period 100 \
    --train_data_root /vulcanscratch/ilyak/data/alexa/v7.18/train_19680 \
    --val_data_root /vulcanscratch/ilyak/data/alexa/v7.18/val_19680_positive_x8 \
    --train_shift '[0.19943352013519514]' \
    --train_center '[0.11222272143962694]' \
    --eval_steps 200 \
    --attacker pgd \
    --attack_norm inf \
    --attack_eps $eps \
    --save_suffix ttt4
done
