#!/bin/bash

rm -rf /scratch0/ilya/locDoc/pyfst/models/throw;
CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--network yu \
--dataset KSC \
--eval_period 200 \
--num_epochs 100000 \
--mask_root  /scratch0/ilya/locDoc/data/hyperspec/KSC_strictsinglesite_trainval_s03_0_880024.mat \
--model_root /scratch0/ilya/locDoc/pyfst/models/throw \
--network_spatial_size 5


# --mask_root /scratch0/ilya/locDoc/data/hyperspec/Botswana_gt_traintest_s03_1_ec2488.mat \