#!/bin/bash

# rm -rf /scratch0/ilya/locDoc/pyfst/models/throw;
CUDA_VISIBLE_DEVICES=0 \
python hyper_pixelNN.py \
--dataset Botswana \
--npca_components 5 \
--batch_size 100 \
--lr 0.0001 \
--network DFFN_3tower_4depth \
--network_spatial_size 25 \
--eval_period 2 \
--num_epochs 100000 \
--mask_root  /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/Botswana_strictsinglesite_trainval_s20_8_379701.mat \
--model_root /scratch0/ilya/locDoc/pyfst/models/throw \
--terminate_if_n_nondecreasing_evals 10 \
--predict

# Indian_pines_gt_traintest_ma2015_1_9146f0, Indian_pines_gt_traintest_p05_1_f0b0f8