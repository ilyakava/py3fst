#!/bin/bash

CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--network yu \
--dataset Botswana \
--predict \
--model_root /scratch0/ilya/locDoc/pyfst/models/Botswana_s03_yu_1_ec2488

# --fst_preprocessing \
