
CUDA_VISIBLE_DEVICES=0 python attack_univ.py \
--model_root=/scratch0/ilya/locDownloads/temp_7.15_model/baseline_tfspecv715_bs64_trial2/export/best_loss_exporter/ \
--max_steps 1000 \
--eval_period 100 \
--train_data_root /scratch1/ilya/locDoc/data/alexa/v7.16/train_19680 \
--val_data_root /scratch1/ilya/locDoc/data/alexa/v7.16/val_19680_positive_x8 \
--train_shift '[0.3729283923166141]' \
--train_center '[0.20274774105000434]' \
--eval_steps 200 \
--attack_fool_rate 0.80 \
--attacker pgd \
--attack_norm inf \
--attack_eps 0.004 \
--save_suffix pgdtest