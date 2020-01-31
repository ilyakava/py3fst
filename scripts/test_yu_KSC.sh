#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python hyper_pixelNN.py \
--network yu \
--dataset KSC \
--model_root /scratch0/ilya/locDoc/pyfst/models/throw \
--network_spatial_size 5 \
--predict


# --mask_root /scratch0/ilya/locDoc/data/hyperspec/Botswana_gt_traintest_s03_1_ec2488.mat \