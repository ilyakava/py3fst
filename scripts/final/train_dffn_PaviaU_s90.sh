#!/bin/bash

# rm -rf /scratch0/ilya/locDoc/pyfst/models/throw;
CUDA_VISIBLE_DEVICES=0 \
python hyper_pixelNN.py \
--dataset PaviaU \
--npca_components 5 \
--batch_size 100 \
--lr 0.00005 \
--network DFFN_3tower_5depth \
--network_spatial_size 23 \
--eval_period 2 \
--num_epochs 100000 \
--mask_root  /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/PaviaU_distributed_trainval_p0200_2_123909.mat \
--model_root /scratch0/ilya/locDoc/pyfst/models/dffn/PaviaU_distributed_trainval_p0200_2_123909/0/ \
--predict



# Indian_pines_gt_traintest_ma2015_1_9146f0, Indian_pines_gt_traintest_p05_1_f0b0f8