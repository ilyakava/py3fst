#!/bin/bash

# rm -rf /scratch0/ilya/locDoc/pyfst/models/throw;
CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--dataset PaviaU \
--npca_components 4 \
--attribute_profile \
--batch_size 50 \
--lr 0.0001 \
--network aptoula \
--network_spatial_size 9 \
--eval_period 2 \
--num_epochs 100000 \
--mask_root  /scratch0/ilya/locDoc/data/hyperspec/PaviaU_distributed_trainval_p0200_7_971587.mat \
--model_root /scratch0/ilya/locDoc/pyfst/models/eap/PaviaU_distributed_trainval_p0200_7_971587/0/ \
--predict

# Indian_pines_gt_traintest_ma2015_1_9146f0, Indian_pines_gt_traintest_p05_1_f0b0f8
# --mask_root  /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/PaviaU_distributed_trainval_p0900_0_656023.mat \