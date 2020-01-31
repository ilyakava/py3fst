"""
This script creates a .sh script that launches many models that need to be trained.
"""


import ntpath
import os

import pdb

mask_list_f = open('scripts/KSC_masks_2more.txt', "r")

masks = [line.strip() for line in mask_list_f.readlines() if line != "\n"]
mask_list_f.close()
valid_masks = [m for m in masks if m and os.path.exists(m)]

with open('scripts/KSC_dffn_all.sh', 'w') as fw:
    fw.write("#!/bin/bash")
    
    for mask_path in valid_masks:
        mask_name = ntpath.basename(mask_path)
        mask_name = mask_name.split('.')[0]
        
        for trial in range(3):
            model_root = '/scratch0/ilya/locDoc/pyfst/models/dffn'
            model_root = os.path.join(model_root, mask_name, str(trial))
            command = """CUDA_VISIBLE_DEVICES=0 \
            python hyper_pixelNN.py \
            --dataset KSC \
            --npca_components 5 \
            --batch_size 100 \
            --lr 0.00005 \
            --network DFFN_3tower_2depth \
            --network_spatial_size 25 \
            --eval_period 20 \
            --num_epochs 100000 \
            --mask_root  {} \
            --model_root {} \
            --terminate_if_n_nondecreasing_evals 10;""".format(mask_path, model_root)
            
            fw.write("\n")
            fw.write(command)
            fw.write("\n")


with open('scripts/KSC_yu_all.sh', 'w') as fw:
    fw.write("#!/bin/bash")
    
    for mask_path in valid_masks:
        mask_name = ntpath.basename(mask_path)
        mask_name = mask_name.split('.')[0]
        
        for trial in range(3):
            model_root = '/scratch0/ilya/locDoc/pyfst/models/yu'
            model_root = os.path.join(model_root, mask_name, str(trial))
            command = """CUDA_VISIBLE_DEVICES=1 \
            python hyper_pixelNN.py \
            --network yu \
            --dataset KSC \
            --eval_period 200 \
            --num_epochs 100000 \
            --mask_root {} \
            --model_root {} \
            --network_spatial_size 5 \
            --terminate_if_n_nondecreasing_evals 10;""".format(mask_path, model_root)
            
            fw.write("\n")
            fw.write(command)
            fw.write("\n")
        
        