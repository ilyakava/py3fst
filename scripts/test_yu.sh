#!/bin/bash

CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--network yu \
--dataset PaviaU \
--model_root /scratch0/ilya/locDoc/data/hypernet/models/throw \
--predict



# other mask choices
#trainfilename = 'PaviaU_gt_traintest_coarse_128px128p.mat'
#trainfilename = 'Pavia_center_right_gt_traintest_coarse_128px128p.mat'
# trainfilename = 'PaviaU_gt_traintest_s03_1_3f6384.mat' #'PaviaU_gt_traintest_s200_1_591636.mat' #'PaviaU_gt_traintest_s03_1_3f6384.mat' #'PaviaU_gt_traintest_s60_1_dd069a.mat'
# PaviaU_gt_traintest_s50_1_d5e6bd.mat
