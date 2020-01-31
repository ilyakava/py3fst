#!/bin/bash

CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--network fst_net \
--dataset PaviaU \
--eval_period 200 \
--num_epochs 100000 \
--fst_preprocessing \
--model_root /scratch0/ilya/locDoc/data/hypernet/models/throw \
--predict \
--preprocessed_data_path /scratch0/ilya/locDoc/data/hypernet/models/throw/preprocess_data.npz


# other mask choices
#trainfilename = 'PaviaU_gt_traintest_coarse_128px128p.mat'
#trainfilename = 'Pavia_center_right_gt_traintest_coarse_128px128p.mat'
# trainfilename = 'PaviaU_gt_traintest_s03_1_3f6384.mat' #'PaviaU_gt_traintest_s200_1_591636.mat' #'PaviaU_gt_traintest_s03_1_3f6384.mat' #'PaviaU_gt_traintest_s60_1_dd069a.mat'

