#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python hyper_pixelNN.py \
--dataset KSC \
--mask_root  /scratch0/ilya/locDoc/data/hyperspec/KSC_strictsinglesite_trainval_s03_1_046456.mat \
--model_root /scratch0/ilya/locDoc/pyfst/models/ksc_svm \
--preprocessed_data_path /scratch0/ilya/locDoc/pyfst/models/ksc_svm/preprocessed.npz \
--st_type KSC \
--fst_preprocessing \
--network_spatial_size 1 \
--svm_predict \
--batch_size 1000


# --mask_root /scratch0/ilya/locDoc/data/hyperspec/Botswana_gt_traintest_s03_1_ec2488.mat \