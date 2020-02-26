#!/bin/bash

CUDA_VISIBLE_DEVICES= \
python hyper_pixelNN.py \
--dataset PaviaU \
--svm_multi_mask_file_list ./mask_lists/PaviaU_sss.txt \
--model_root /scratch0/ilya/locDoc/pyfst/models/paviau_fst_third \
--preprocessed_data_path /scratch0/ilya/locDoc/pyfst/models/paviau_fst_third/PU.npz \
--st_type PU_SSS \
--fst_preprocessing \
--network_spatial_size 1 \
--batch_size 1000 \
--svm_regularization_param 1000


# --svm_multi_mask_file_list ./mask_lists/PaviaU_distributed.txt \