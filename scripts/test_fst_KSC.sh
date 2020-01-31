#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python hyper_pixelNN.py \
--network fst_net \
--dataset KSC \
--model_root /scratch0/ilya/locDoc/pyfst/models/ksc_svm \
--preprocessed_data_path /scratch0/ilya/locDoc/pyfst/models/ksc_svm/preprocessed.npz \
--st_type KSC \
--fst_preprocessing \
--network_spatial_size 1 \
--predict


# --mask_root /scratch0/ilya/locDoc/data/hyperspec/Botswana_gt_traintest_s03_1_ec2488.mat \