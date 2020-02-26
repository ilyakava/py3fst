#!/bin/bash

# This script replicates the results of DFFN for indian pines

rm -rf /scratch0/ilya/locDoc/caffe-tensorflow/models/throw;
CUDA_VISIBLE_DEVICES=0 \
python hyper_pixelNN.py \
--dataset KSC \
--npca_components 5 \
--batch_size 100 \
--lr 0.00005 \
--network DFFN_3tower_2depth \
--network_spatial_size 25 \
--eval_period 20 \
--num_epochs 100000 \
--mask_root  /scratch0/ilya/locDoc/data/hyperspec/KSC_strictsinglesite_trainval_s03_0_880024.mat \
--model_root /scratch0/ilya/locDoc/caffe-tensorflow/models/throw \
--terminate_if_n_nondecreasing_evals 10
# --predict

# Indian_pines_gt_traintest_ma2015_1_9146f0, Indian_pines_gt_traintest_p05_1_f0b0f8