
CUDA_VISIBLE_DEVICES=0 python attack_trial.py \
--log_dir=/scratch0/ilya/locDownloads/temp_7.15_model/baseline_tfspecv715_bs64_trial2/export/best_loss_exporter \
--exported_model_glob='/scratch0/ilya/locDownloads/temp_7.15_model/baseline_tfspecv715_bs64_trial2/export/best_loss_exporter/*' \
--attack_norm inf \
--attack_eps 0.0001 \
--attack_eps_step 0.00001 \
--tfrecord_glob='/scratch1/ilya/locDoc/data/alexa/v7.16/val_21280_positive/*.tfrecord' \
--data_shift '[0.3715735405001496]' \
--data_center '[0.2031791745456392]'

# INFO:tensorflow:[001] Mask acc 0.9375/0.4688 (0.9375/0.4688). Det acc 0.8667/0.4667 (0.8667/0.4667). SNR 55.19 (55.19)
# INFO:tensorflow:[002] Mask acc 0.8125/0.6875 (0.8750/0.5781). Det acc 0.6000/0.8000 (0.7333/0.6333). SNR 54.58 (54.89)
# INFO:tensorflow:[003] Mask acc 0.9062/0.5625 (0.8854/0.5729). Det acc 0.9286/0.5000 (0.7984/0.5889). SNR 53.16 (54.31)
# INFO:tensorflow:[004] Mask acc 0.9375/0.3125 (0.8984/0.5078). Det acc 0.9375/0.3750 (0.8332/0.5354). SNR 55.99 (54.73)
# INFO:tensorflow:[005] Mask acc 0.9688/0.5000 (0.9125/0.5062). Det acc 1.0000/0.6667 (0.8665/0.5617). SNR 54.54 (54.69)
# INFO:tensorflow:[006] Mask acc 0.9062/0.6250 (0.9115/0.5260). Det acc 0.9286/0.5714 (0.8769/0.5633). SNR 56.03 (54.92)
# INFO:tensorflow:[007] Mask acc 0.8750/0.4375 (0.9062/0.5134). Det acc 0.8667/0.5333 (0.8754/0.5590). SNR 53.38 (54.70)
# INFO:tensorflow:[008] Mask acc 0.8438/0.5000 (0.8984/0.5117). Det acc 0.9231/0.6154 (0.8814/0.5661). SNR 54.52 (54.67)
# INFO:tensorflow:[009] Mask acc 0.9375/0.6875 (0.9028/0.5312). Det acc 1.0000/0.7500 (0.8946/0.5865). SNR 53.00 (54.49)
# INFO:tensorflow:[010] Mask acc 0.9688/0.4688 (0.9094/0.5250). Det acc 1.0000/0.4000 (0.9051/0.5678). SNR 53.72 (54.41)