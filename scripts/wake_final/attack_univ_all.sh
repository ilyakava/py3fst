
export MODEL_DIR=/vulcanscratch/ilyak/experiments/wake_tf_board/corticalv0res1c_drop9_v718_bs64_trial3/export/best_loss_exporter/
export SAVE_SUFFIX=n3t4

for eps in 0.0025 0.005 0.0075 0.01 0.02 0.00375 0.015
do
    CUDA_VISIBLE_DEVICES=1 python attack_univ.py \
    --model_root=$MODEL_DIR \
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
    --save_suffix $SAVE_SUFFIX
done

for eps in 0.0025 0.005 0.0075 0.01 0.02 0.00375 0.015
do
    CUDA_VISIBLE_DEVICES=1 python attack_univ.py \
    --model_root=$MODEL_DIR \
    --max_steps 4000 \
    --eval_period 400 \
    --train_data_root /vulcanscratch/ilyak/data/alexa/v7.18/train_19680 \
    --val_data_root /vulcanscratch/ilyak/data/alexa/v7.18/val_19680_positive_x8 \
    --train_shift '[0.19943352013519514]' \
    --train_center '[0.11222272143962694]' \
    --eval_steps 200 \
    --attacker fgsm \
    --attack_norm inf \
    --attack_eps $eps \
    --save_suffix $SAVE_SUFFIX
done

for eps in 0.0025 0.005 0.0075 0.01 0.02 0.00375 0.015
do
    CUDA_VISIBLE_DEVICES=1 python attack_univ.py \
    --model_root=$MODEL_DIR \
    --max_steps 4000 \
    --eval_period 400 \
    --train_data_root /vulcanscratch/ilyak/data/alexa/v7.18/train_19680 \
    --val_data_root /vulcanscratch/ilyak/data/alexa/v7.18/val_19680_positive_x8 \
    --train_shift '[0.19943352013519514]' \
    --train_center '[0.11222272143962694]' \
    --eval_steps 200 \
    --attacker deepfool \
    --attack_norm inf \
    --attack_eps $eps \
    --save_suffix $SAVE_SUFFIX
done

for eps in 0.0025 0.00375 0.005 0.0075 0.01 0.015 0.02
do
    CUDA_VISIBLE_DEVICES=1 python attack_univ.py \
    --model_root=$MODEL_DIR \
    --max_steps 250 \
    --eval_period 25 \
    --train_data_root /vulcanscratch/ilyak/data/alexa/v7.18/train_19680 \
    --val_data_root /vulcanscratch/ilyak/data/alexa/v7.18/val_19680_positive_x8 \
    --train_shift '[0.19943352013519514]' \
    --train_center '[0.11222272143962694]' \
    --eval_steps 200 \
    --attacker pgd \
    --attack_norm inf \
    --attack_eps $eps \
    --save_suffix $SAVE_SUFFIX
done

rsync -v /vulcanscratch/ilyak/experiments/wake_tf_board/corticalv0res1c_drop9_v718_bs64_trial3/export/best_loss_exporter/*npz ramawks69:/scratch0/ilya/locDownloads/corticalv0res1c_drop9_v718_bs64_trial3/