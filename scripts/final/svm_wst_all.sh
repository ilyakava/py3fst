#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python hyper_pixelNN.py \
--fst_preprocessing \
--st_type tang \
--dataset IP \
--svm_multi_mask_file_list ./mask_lists/IP_distributed.txt \
--model_root /scratch0/ilya/locDoc/pyfst/models/ip_tang_svm \
--network_spatial_size 1 \
--batch_size 1000 \
--svm_regularization_param 1000


CUDA_VISIBLE_DEVICES=0 \
python hyper_pixelNN.py \
--fst_preprocessing \
--st_type tang \
--dataset PaviaU \
--svm_multi_mask_file_list ./mask_lists/PaviaU_distributed.txt \
--model_root /scratch0/ilya/locDoc/pyfst/models/pu_tang_svm \
--network_spatial_size 1 \
--batch_size 1000

CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--fst_preprocessing \
--st_type tang \
--dataset PaviaU \
--svm_multi_mask_file_list ./mask_lists/PaviaU_sss.txt \
--model_root /scratch0/ilya/locDoc/pyfst/models/pu_tang_svm \
--network_spatial_size 1 \
--batch_size 1000

CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--fst_preprocessing \
--st_type tang \
--dataset KSC \
--svm_multi_mask_file_list ./mask_lists/KSC_sss.txt \
--model_root /scratch0/ilya/locDoc/pyfst/models/KSC_tang_svm \
--network_spatial_size 1 \
--batch_size 1000

CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--fst_preprocessing \
--st_type tang \
--dataset Botswana \
--svm_multi_mask_file_list ./mask_lists/Botswana_sss.txt \
--model_root /scratch0/ilya/locDoc/pyfst/models/Botswana_tang_svm \
--network_spatial_size 1 \
--batch_size 1000