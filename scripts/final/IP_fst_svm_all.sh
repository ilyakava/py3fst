#!/bin/bash

CUDA_VISIBLE_DEVICES= \
python hyper_pixelNN.py \
--dataset IP \
--svm_multi_mask_file_list ./mask_lists/IP_distributed.txt \
--model_root /scratch0/ilya/locDoc/pyfst/models/ip_fst_svm_third \
--preprocessed_data_path /scratch0/ilya/locDoc/pyfst/models/ip_fst_svm_third/IP.npz \
--st_type IP_SSS \
--fst_preprocessing \
--network_spatial_size 1 \
--batch_size 1000 \
--svm_regularization_param 1000


# --mask_root /scratch0/ilya/locDoc/data/hyperspec/Botswana_gt_traintest_s03_1_ec2488.mat \