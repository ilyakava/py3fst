#!/bin/bash

# This script replicates the results of EAP-Area for PaviaU
# Overall Accuracy: Best by Eval Acc: 0.9959. Best by Eval Loss: 0.9956

rm -rf /scratch0/ilya/locDoc/pyfst/models/throw;
CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--dataset PaviaU \
--npca_components 4 \
--attribute_profile \
--batch_size 50 \
--lr 0.00005 \
--network aptoula \
--network_spatial_size 9 \
--eval_period 25 \
--num_epochs 100000 \
--mask_root  /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/PaviaU_distributed_trainval_p0900_0_656023.mat \
--model_root /scratch0/ilya/locDoc/pyfst/models/throw
# --predict
