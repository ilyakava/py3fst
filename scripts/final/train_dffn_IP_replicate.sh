#!/bin/bash

# This script replicates the results of DFFN for indian pines

rm -rf /scratch0/ilya/locDoc/pyfst/models/throw;
CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--dataset IP \
--npca_components 3 \
--batch_size 100 \
--lr 0.00005 \
--network DFFN_3tower_4depth \
--network_spatial_size 25 \
--eval_period 5 \
--num_epochs 100000 \
--mask_root  /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/IP_distributed_trainval_p1000_0_970730.mat \
--model_root /scratch0/ilya/locDoc/pyfst/models/throw

# Indian_pines_gt_traintest_ma2015_1_9146f0, Indian_pines_gt_traintest_p05_1_f0b0f8