#!/bin/bash

rm -rf /scratch0/ilya/locDoc/pyfst/models/throw;
mkdir /scratch0/ilya/locDoc/pyfst/models/throw;
CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--network yu \
--fst_preprocessing \
--st_type Botswana \
--dataset Botswana \
--eval_period 200 \
--num_epochs 100000 \
--mask_root /scratch0/ilya/locDoc/data/hyperspec/Botswana_singlesite_trainval_s03_0_431593.mat \
--model_root /scratch0/ilya/locDoc/pyfst/models/throw

# --fst_preprocessing \
# --mask_root /scratch0/ilya/locDoc/data/hyperspec/Botswana_gt_traintest_s03_1_ec2488.mat \