
CUDA_VISIBLE_DEVICES=0 python attack_trial.py \
--log_dir=/scratch0/ilya/locDownloads/temp_v715_cortical_model_res1b/export/best_acc_exporter \
--exported_model_glob='/scratch0/ilya/locDownloads/temp_v715_cortical_model_res1b/export/best_acc_exporter/*' \
--attack_norm inf \
--attack_eps 0.0001 \
--attack_eps_step 0.00001 \
--tfrecord_glob='/scratch1/ilya/locDoc/data/alexa/v7.16/val_21280_positive/*.tfrecord' \
--data_shift '[0.3715735405001496]' \
--data_center '[0.2031791745456392]'

# INFO:tensorflow:[001] Mask acc 0.9375/0.6250 (0.9375/0.6250). Det acc 0.8667/0.7333 (0.8667/0.7333). SNR 55.38 (55.38)
# INFO:tensorflow:[002] Mask acc 0.8125/0.6562 (0.8750/0.6406). Det acc 0.6000/0.5333 (0.7333/0.6333). SNR 54.62 (55.00)
# INFO:tensorflow:[003] Mask acc 0.8438/0.6562 (0.8646/0.6458). Det acc 0.7857/0.7857 (0.7508/0.6841). SNR 53.18 (54.39)
# INFO:tensorflow:[004] Mask acc 0.9062/0.5312 (0.8750/0.6172). Det acc 0.8750/0.6875 (0.7818/0.6850). SNR 56.20 (54.85)
# INFO:tensorflow:[005] Mask acc 0.9375/0.5938 (0.8875/0.6125). Det acc 1.0000/0.7500 (0.8255/0.6980). SNR 54.51 (54.78)
# INFO:tensorflow:[006] Mask acc 0.8438/0.6250 (0.8802/0.6146). Det acc 0.8571/0.6429 (0.8308/0.6888). SNR 55.88 (54.96)
# INFO:tensorflow:[007] Mask acc 0.9375/0.5000 (0.8884/0.5982). Det acc 0.9333/0.6000 (0.8454/0.6761). SNR 53.55 (54.76)
# INFO:tensorflow:[008] Mask acc 0.8125/0.6562 (0.8789/0.6055). Det acc 0.9231/0.6923 (0.8551/0.6781). SNR 54.70 (54.75)
# INFO:tensorflow:[009] Mask acc 0.9062/0.8438 (0.8819/0.6319). Det acc 0.9167/0.7500 (0.8620/0.6861). SNR 53.19 (54.58)
# INFO:tensorflow:[010] Mask acc 0.9375/0.5938 (0.8875/0.6281). Det acc 0.9333/0.4000 (0.8691/0.6575). SNR 53.76 (54.50)