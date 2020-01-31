#!/bin/bash

rm -rf /scratch0/ilya/locDoc/pyfst/models/throw;
mkdir /scratch0/ilya/locDoc/pyfst/models/throw;
CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--network fst_net \
--dataset KSC \
--eval_period 200 \
--num_epochs 100000 \
--mask_root  /scratch0/ilya/locDoc/data/hyperspec/KSC_singlesite_trainval_s03_0_898132.mat \
--model_root /scratch0/ilya/locDoc/pyfst/models/throw \
--preprocessed_data_path /scratch0/ilya/locDoc/pyfst/models/throw/preprocessed.npz \
--st_type KSC \
--fst_preprocessing \
--network_spatial_size 1


# --mask_root /scratch0/ilya/locDoc/data/hyperspec/Botswana_gt_traintest_s03_1_ec2488.mat \