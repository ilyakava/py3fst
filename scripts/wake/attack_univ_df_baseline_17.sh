
CUDA_VISIBLE_DEVICES=0 python attack_univ.py \
--model_root=/scratch0/ilya/locDownloads/temp_7.17_model/best_acc_exporter/ \
--attack_norm inf \
--attack_eps 0.005 \
--attack_eps_step 0.0005 \
--max_steps 10000 \
--eval_period 1000 \
--train_data_root /scratch1/ilya/locDoc/data/alexa/v7.17/train_19680 \
--val_data_root /scratch1/ilya/locDoc/data/alexa/v7.17/val_19680_positive_x8 \
--train_shift '[0.1621694629344238]' \
--train_center '[0.08856695466572413]' \
--eval_steps 200 \
--attack_fool_rate 0.99