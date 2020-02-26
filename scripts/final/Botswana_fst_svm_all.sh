#!/bin/bash

CUDA_VISIBLE_DEVICES= \
python hyper_pixelNN.py \
--dataset Botswana \
--svm_multi_mask_file_list ./mask_lists/Botswana_sss.txt \
--model_root /scratch0/ilya/locDoc/pyfst/models/bots_fst \
--preprocessed_data_path /scratch0/ilya/locDoc/pyfst/models/bots_fst/preprocessed.npz \
--st_type Botswana \
--fst_preprocessing \
--network_spatial_size 1 \
--batch_size 1000


# --mask_root /scratch0/ilya/locDoc/data/hyperspec/Botswana_gt_traintest_s03_1_ec2488.mat \
# --preprocessed_data_path /scratch0/ilya/locDoc/data/hyperspec/features/npz_feat/KSC__5-9-9_5-7-7_5-7-7.npz \