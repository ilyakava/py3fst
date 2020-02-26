#!/bin/bash

CUDA_VISIBLE_DEVICES= \
python hyper_pixelNN.py \
--dataset IP \
--svm_multi_mask_file_list ./mask_lists/IP_distributed.txt \
--model_root /scratch0/ilya/locDoc/pyfst/models/ip_raw_svm \
--network_spatial_size 1 \
--batch_size 1000 \
--svm_regularization_param 1000


CUDA_VISIBLE_DEVICES= \
python hyper_pixelNN.py \
--dataset PaviaU \
--svm_multi_mask_file_list ./mask_lists/PaviaU_distributed.txt \
--model_root /scratch0/ilya/locDoc/pyfst/models/pu_raw_svm \
--network_spatial_size 1 \
--batch_size 1000
																											
CUDA_VISIBLE_DEVICES= \
python hyper_pixelNN.py \
--dataset PaviaU \
--svm_multi_mask_file_list ./mask_lists/PaviaU_sss.txt \
--model_root /scratch0/ilya/locDoc/pyfst/models/pu_raw_svm \
--network_spatial_size 1 \
--batch_size 1000

CUDA_VISIBLE_DEVICES= \
python hyper_pixelNN.py \
--dataset KSC \
--svm_multi_mask_file_list ./mask_lists/KSC_sss.txt \
--model_root /scratch0/ilya/locDoc/pyfst/models/KSC_raw_svm \
--network_spatial_size 1 \
--batch_size 1000

CUDA_VISIBLE_DEVICES= \
python hyper_pixelNN.py \
--dataset Botswana \
--svm_multi_mask_file_list ./mask_lists/Botswana_sss.txt \
--model_root /scratch0/ilya/locDoc/pyfst/models/Botswana_raw_svm \
--network_spatial_size 1 \
--batch_size 1000