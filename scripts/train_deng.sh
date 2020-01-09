#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 \
python hyper_pixelNN.py \
--network deng \
--dataset PaviaU \
--mask_root /scratch0/ilya/locDoc/data/hyperspec/PaviaU_gt_traintest_s03_1_3f6384.mat



# other mask choices
#trainfilename = 'PaviaU_gt_traintest_coarse_128px128p.mat'
#trainfilename = 'Pavia_center_right_gt_traintest_coarse_128px128p.mat'
# trainfilename = 'PaviaU_gt_traintest_s03_1_3f6384.mat' #'PaviaU_gt_traintest_s200_1_591636.mat' #'PaviaU_gt_traintest_s03_1_3f6384.mat' #'PaviaU_gt_traintest_s60_1_dd069a.mat'

