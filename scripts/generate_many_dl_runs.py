"""
This script creates a .sh script that launches many models that need to be trained.
"""


import ntpath
import os

import pdb



# mask_list_f = open('scripts/KSC_distributed_masks.txt', "r")

# masks = [line.strip() for line in mask_list_f.readlines() if line != "\n"]
# mask_list_f.close()
# valid_masks = [m for m in masks if m and os.path.exists(m)]

# with open('scripts/KSC_dffn_all.sh', 'w') as fw:
#     fw.write("#!/bin/bash")
    
#     for mask_path in valid_masks:
#         mask_name = ntpath.basename(mask_path)
#         mask_name = mask_name.split('.')[0]
        
#         for trial in range(1):
#             model_root = '/scratch2/ilyak/locDoc/pyfst/models/dffn'
#             model_root = os.path.join(model_root, mask_name, str(trial))
#             command = """CUDA_VISIBLE_DEVICES=0 \
#             python hyper_pixelNN.py \
#             --dataset KSC \
#             --npca_components 5 \
#             --batch_size 100 \
#             --lr 0.00005 \
#             --network DFFN_3tower_4depth \
#             --network_spatial_size 25 \
#             --eval_period 2 \
#             --num_epochs 100000 \
#             --mask_root  {} \
#             --model_root {} \
#             --terminate_if_n_nondecreasing_evals 10;""".format(mask_path, model_root)
            
#             fw.write("\n")
#             fw.write(command)
#             fw.write("\n")


# mask_list_f = open('scripts/PaviaU_distributed_masks.txt', "r")

# masks = [line.strip() for line in mask_list_f.readlines() if line != "\n"]
# mask_list_f.close()
# valid_masks = [m for m in masks if m and os.path.exists(m)]

# with open('scripts/PaviaU_dffn_all2.sh', 'w') as fw:
#     fw.write("#!/bin/bash")
    
#     for mask_path in valid_masks:
#         mask_name = ntpath.basename(mask_path)
#         mask_name = mask_name.split('.')[0]
        
#         for trial in range(1):
#             model_root = '/scratch0/ilya/locDoc/pyfst/models/dffn'
#             model_root = os.path.join(model_root, mask_name, str(trial))
#             command = """CUDA_VISIBLE_DEVICES=0 \
#             python hyper_pixelNN.py \
#             --dataset PaviaU \
#             --npca_components 5 \
#             --batch_size 100 \
#             --lr 0.00005 \
#             --network DFFN_3tower_5depth \
#             --network_spatial_size 23 \
#             --eval_period 2 \
#             --num_epochs 100000 \
#             --mask_root  {} \
#             --model_root {} \
#             --terminate_if_n_nondecreasing_evals 10;""".format(mask_path, model_root)
            
#             fw.write("\n")
#             fw.write(command)
#             fw.write("\n")
            
            
# mask_list_f = open('scripts/IP_masks.txt', "r")

# masks = [line.strip() for line in mask_list_f.readlines() if line != "\n"]
# mask_list_f.close()
# valid_masks = [m for m in masks if m and os.path.exists(m)]

# with open('scripts/IP_dffn_all.sh', 'w') as fw:
#     fw.write("#!/bin/bash")
    
#     for mask_path in valid_masks:
#         mask_name = ntpath.basename(mask_path)
#         mask_name = mask_name.split('.')[0]
        
#         for trial in range(1):
#             model_root = '/scratch0/ilya/locDoc/pyfst/models/dffn'
#             model_root = os.path.join(model_root, mask_name, str(trial))
#             command = """CUDA_VISIBLE_DEVICES=0 \
#             python hyper_pixelNN.py \
#             --dataset IP \
#             --npca_components 3 \
#             --batch_size 100 \
#             --lr 0.00005 \
#             --network DFFN_3tower_4depth \
#             --network_spatial_size 25 \
#             --eval_period 2 \
#             --num_epochs 100000 \
#             --mask_root  {} \
#             --model_root {} \
#             --terminate_if_n_nondecreasing_evals 10;""".format(mask_path, model_root)
            
#             fw.write("\n")
#             fw.write(command)
#             fw.write("\n")

# mask_list_f = open('scripts/PaviaU_hi_data_masks.txt', "r")

# masks = [line.strip() for line in mask_list_f.readlines() if line != "\n"]
# mask_list_f.close()
# valid_masks = [m for m in masks if m and os.path.exists(m)]

# with open('scripts/PaviaU_dffn_all_hi_data.sh', 'w') as fw:
#     fw.write("#!/bin/bash")
    
#     for mask_path in valid_masks:
#         mask_name = ntpath.basename(mask_path)
#         mask_name = mask_name.split('.')[0]
        
#         for trial in range(1):
#             model_root = '/scratch2/ilyak/locDoc/pyfst/models/dffn'
#             model_root = os.path.join(model_root, mask_name, str(trial))
#             command = """CUDA_VISIBLE_DEVICES=0 \
#             python hyper_pixelNN.py \
#             --data_root /scratch2/ilyak/locDoc/data/hyperspec/datasets \
#             --dataset PaviaU \
#             --npca_components 5 \
#             --batch_size 100 \
#             --lr 0.00005 \
#             --network DFFN_3tower_5depth \
#             --network_spatial_size 23 \
#             --eval_period 2 \
#             --num_epochs 100000 \
#             --mask_root  {} \
#             --model_root {} \
#             --terminate_if_n_nondecreasing_evals 10;""".format(mask_path, model_root)
            
#             fw.write("\n")
#             fw.write(command)
#             fw.write("\n")


# mask_list_f = open('scripts/IP_hi_data_masks.txt', "r")

# masks = [line.strip() for line in mask_list_f.readlines() if line != "\n"]
# mask_list_f.close()
# valid_masks = [m for m in masks if m and os.path.exists(m)]

# with open('scripts/IP_dffn_all_hi_data.sh', 'w') as fw:
#     fw.write("#!/bin/bash")
    
#     for mask_path in valid_masks:
#         mask_name = ntpath.basename(mask_path)
#         mask_name = mask_name.split('.')[0]
        
#         for trial in range(1):
#             model_root = '/scratch2/ilyak/locDoc/pyfst/models/dffn'
#             model_root = os.path.join(model_root, mask_name, str(trial))
#             command = """CUDA_VISIBLE_DEVICES=1 \
#             python hyper_pixelNN.py \
#             --data_root /scratch2/ilyak/locDoc/data/hyperspec/datasets \
#             --dataset IP \
#             --npca_components 3 \
#             --batch_size 100 \
#             --lr 0.00005 \
#             --network DFFN_3tower_4depth \
#             --network_spatial_size 25 \
#             --eval_period 2 \
#             --num_epochs 100000 \
#             --mask_root  {} \
#             --model_root {} \
#             --terminate_if_n_nondecreasing_evals 10;""".format(mask_path, model_root)
            
#             fw.write("\n")
#             fw.write(command)
#             fw.write("\n")

# mask_list_f = open('scripts/IP_distributed_masks.txt', "r")

# masks = [line.strip() for line in mask_list_f.readlines() if line != "\n"]
# mask_list_f.close()
# valid_masks = [m for m in masks if m and os.path.exists(m)]

# with open('scripts/IP_dffn_all_distributed.sh', 'w') as fw:
#     fw.write("#!/bin/bash")
    
#     for mask_path in valid_masks:
#         mask_name = ntpath.basename(mask_path)
#         mask_name = mask_name.split('.')[0]
        
#         for trial in range(1):
#             model_root = '/scratch2/ilyak/locDoc/pyfst/models/dffn'
#             model_root = os.path.join(model_root, mask_name, str(trial))
#             command = """CUDA_VISIBLE_DEVICES=1 \
#             python hyper_pixelNN.py \
#             --data_root /scratch2/ilyak/locDoc/data/hyperspec/datasets \
#             --dataset IP \
#             --npca_components 3 \
#             --batch_size 100 \
#             --lr 0.00002 \
#             --network DFFN_3tower_4depth \
#             --network_spatial_size 25 \
#             --eval_period 2 \
#             --num_epochs 100000 \
#             --mask_root  {} \
#             --model_root {} \
#             --terminate_if_n_nondecreasing_evals 10;""".format(mask_path, model_root)
            
#             fw.write("\n")
#             fw.write(command)
#             fw.write("\n")


#### Bots DFFN
# mask_list_f = open('scripts/Bots_masks.txt', "r")

# masks = [line.strip() for line in mask_list_f.readlines() if line != "\n"]
# mask_list_f.close()
# valid_masks = [m for m in masks if m and os.path.exists(m)]

# with open('scripts/Bots_dffn_all.sh', 'w') as fw:
#     fw.write("#!/bin/bash")
    
#     for mask_path in valid_masks:
#         mask_name = ntpath.basename(mask_path)
#         mask_name = mask_name.split('.')[0]
        
#         for trial in range(1):
#             model_root = '/scratch0/ilya/locDoc/pyfst/models/dffn'
#             model_root = os.path.join(model_root, mask_name, str(trial))
#             command = """CUDA_VISIBLE_DEVICES=1 \
#             python hyper_pixelNN.py \
#             --dataset Botswana \
#             --npca_components 5 \
#             --batch_size 100 \
#             --lr 0.00005 \
#             --network DFFN_3tower_4depth \
#             --network_spatial_size 25 \
#             --eval_period 2 \
#             --num_epochs 100000 \
#             --num_epochs 100000 \
#             --mask_root  {} \
#             --model_root {} \
#             --terminate_if_n_nondecreasing_evals 10;""".format(mask_path, model_root)
            
#             fw.write("\n")
#             fw.write(command)
#             fw.write("\n")


### EAP IP distributed
# mask_list_f = open('mask_lists/IP_distributed.txt', "r")

# masks = [line.strip() for line in mask_list_f.readlines() if line != "\n"]
# mask_list_f.close()
# valid_masks = [m for m in masks if m and os.path.exists(m)]

# with open('scripts/IP_eap_distibuted_all.sh', 'w') as fw:
#     fw.write("#!/bin/bash")
    
#     for mask_path in valid_masks:
#         mask_name = ntpath.basename(mask_path)
#         mask_name = mask_name.split('.')[0]
        
#         for trial in range(1):
#             model_root = '/scratch2/ilyak/locDoc/pyfst/models/eap'
#             model_root = os.path.join(model_root, mask_name, str(trial))
#             command = """CUDA_VISIBLE_DEVICES=1 \
#             python hyper_pixelNN.py \
#             --data_root /scratch2/ilyak/locDoc/data/hyperspec/datasets \
#             --dataset IP \
#             --npca_components 4 \
#             --attribute_profile \
#             --batch_size 50 \
#             --lr 0.00005 \
#             --network aptoula \
#             --network_spatial_size 9 \
#             --eval_period 25 \
#             --num_epochs 100000 \
#             --mask_root  {} \
#             --model_root {} \
#             --terminate_if_n_nondecreasing_evals 10;""".format(mask_path, model_root)
            
#             fw.write("\n")
#             fw.write(command)
#             fw.write("\n")


### EAP PaviaU distributed
# mask_list_f = open('mask_lists/PaviaU_distributed.txt', "r")

# masks = [line.strip() for line in mask_list_f.readlines() if line != "\n"]
# mask_list_f.close()
# valid_masks = [m for m in masks if m and os.path.exists(m)]

# with open('scripts/PaviaU_eap_distibuted_all.sh', 'w') as fw:
#     fw.write("#!/bin/bash")
    
#     for mask_path in valid_masks:
#         mask_name = ntpath.basename(mask_path)
#         mask_name = mask_name.split('.')[0]
        
#         for trial in range(1):
#             model_root = '/scratch2/ilyak/locDoc/pyfst/models/eap'
#             model_root = os.path.join(model_root, mask_name, str(trial))
#             command = """CUDA_VISIBLE_DEVICES=0 \
#             python hyper_pixelNN.py \
#             --data_root /scratch2/ilyak/locDoc/data/hyperspec/datasets \
#             --dataset PaviaU \
#             --npca_components 4 \
#             --attribute_profile \
#             --batch_size 50 \
#             --lr 0.00005 \
#             --network aptoula \
#             --network_spatial_size 9 \
#             --eval_period 25 \
#             --num_epochs 100000 \
#             --mask_root  {} \
#             --model_root {} \
#             --terminate_if_n_nondecreasing_evals 10;""".format(mask_path, model_root)
            
#             fw.write("\n")
#             fw.write(command)
#             fw.write("\n")

### EAP PaviaU SSS
# mask_list_f = open('mask_lists/PaviaU_sss.txt', "r")

# masks = [line.strip() for line in mask_list_f.readlines() if line != "\n"]
# mask_list_f.close()
# valid_masks = [m for m in masks if m and os.path.exists(m)]

# with open('scripts/PaviaU_eap_sss_all.sh', 'w') as fw:
#     fw.write("#!/bin/bash")
    
#     for mask_path in valid_masks:
#         mask_name = ntpath.basename(mask_path)
#         mask_name = mask_name.split('.')[0]
        
#         for trial in range(1):
#             model_root = '/scratch2/ilyak/locDoc/pyfst/models/eap'
#             model_root = os.path.join(model_root, mask_name, str(trial))
#             command = """CUDA_VISIBLE_DEVICES=1 \
#             python hyper_pixelNN.py \
#             --data_root /scratch2/ilyak/locDoc/data/hyperspec/datasets \
#             --dataset PaviaU \
#             --npca_components 4 \
#             --attribute_profile \
#             --batch_size 50 \
#             --lr 0.0001 \
#             --network aptoula \
#             --network_spatial_size 9 \
#             --eval_period 2 \
#             --num_epochs 100000 \
#             --mask_root  {} \
#             --model_root {} \
#             --terminate_if_n_nondecreasing_evals 10;""".format(mask_path, model_root)
            
#             fw.write("\n")
#             fw.write(command)
#             fw.write("\n")


### EAP KSC SSS
# mask_list_f = open('mask_lists/KSC_sss.txt', "r")

# masks = [line.strip() for line in mask_list_f.readlines() if line != "\n"]
# mask_list_f.close()
# valid_masks = [m for m in masks if m and os.path.exists(m)]

# with open('scripts/KSC_eap_sss_all.sh', 'w') as fw:
#     fw.write("#!/bin/bash")
    
#     for mask_path in valid_masks:
#         mask_name = ntpath.basename(mask_path)
#         mask_name = mask_name.split('.')[0]
        
#         for trial in range(1):
#             model_root = '/scratch2/ilyak/locDoc/pyfst/models/eap'
#             model_root = os.path.join(model_root, mask_name, str(trial))
#             command = """CUDA_VISIBLE_DEVICES=0 \
#             python hyper_pixelNN.py \
#             --data_root /scratch2/ilyak/locDoc/data/hyperspec/datasets \
#             --dataset KSC \
#             --npca_components 4 \
#             --attribute_profile \
#             --batch_size 50 \
#             --lr 0.0001 \
#             --network aptoula \
#             --network_spatial_size 9 \
#             --eval_period 2 \
#             --num_epochs 100000 \
#             --mask_root  {} \
#             --model_root {} \
#             --terminate_if_n_nondecreasing_evals 10;""".format(mask_path, model_root)
            
#             fw.write("\n")
#             fw.write(command)
#             fw.write("\n")

### EAP Botswana SSS
# mask_list_f = open('mask_lists/Botswana_sss.txt', "r")

# masks = [line.strip() for line in mask_list_f.readlines() if line != "\n"]
# mask_list_f.close()
# valid_masks = [m for m in masks if m and os.path.exists(m)]

# with open('scripts/Botswana_eap_sss_all.sh', 'w') as fw:
#     fw.write("#!/bin/bash")
    
#     for mask_path in valid_masks:
#         mask_name = ntpath.basename(mask_path)
#         mask_name = mask_name.split('.')[0]
        
#         for trial in range(1):
#             model_root = '/scratch0/ilya/locDoc/pyfst/models/eap'
#             model_root = os.path.join(model_root, mask_name, str(trial))
#             command = """CUDA_VISIBLE_DEVICES=1 \
#             python hyper_pixelNN.py \
#             --dataset Botswana \
#             --npca_components 4 \
#             --attribute_profile \
#             --batch_size 50 \
#             --lr 0.0001 \
#             --network aptoula \
#             --network_spatial_size 9 \
#             --eval_period 2 \
#             --num_epochs 100000 \
#             --mask_root  {} \
#             --model_root {} \
#             --terminate_if_n_nondecreasing_evals 20;""".format(mask_path, model_root)
            
#             fw.write("\n")
#             fw.write(command)
#             fw.write("\n")


#### IP DFFN
# mask_list_f = open('mask_lists/IP_distributed.txt', "r")

# masks = [line.strip() for line in mask_list_f.readlines() if line != "\n"]
# mask_list_f.close()
# valid_masks = [m for m in masks if m and os.path.exists(m)]

# with open('scripts/IP_dffn_distibuted_all.sh', 'w') as fw:
#     fw.write("#!/bin/bash")
    
#     for mask_path in valid_masks:
#         mask_name = ntpath.basename(mask_path)
#         mask_name = mask_name.split('.')[0]
        
#         for trial in range(1):
#             model_root = '/scratch0/ilya/locDoc/pyfst/models/dffn'
#             model_root = os.path.join(model_root, mask_name, str(trial))
#             command = """CUDA_VISIBLE_DEVICES=0 \
#             python hyper_pixelNN.py \
#             --dataset IP \
#             --npca_components 3 \
#             --batch_size 100 \
#             --lr 0.0001 \
#             --network DFFN_3tower_4depth \
#             --network_spatial_size 25 \
#             --eval_period 2 \
#             --num_epochs 100000 \
#             --mask_root  {} \
#             --model_root {} \
#             --terminate_if_n_nondecreasing_evals 10;""".format(mask_path, model_root)
            
#             fw.write("\n")
#             fw.write(command)
#             fw.write("\n")


### DFFN Botswana SSS
mask_list_f = open('mask_lists/Botswana_sss.1.txt', "r")

masks = [line.strip() for line in mask_list_f.readlines() if line != "\n"]
mask_list_f.close()
valid_masks = [m for m in masks if m and os.path.exists(m)]

with open('scripts/Botswana_dffn_sss1.sh', 'w') as fw:
    fw.write("#!/bin/bash")
    
    for mask_path in valid_masks:
        mask_name = ntpath.basename(mask_path)
        mask_name = mask_name.split('.')[0]
        
        for trial in range(1):
            model_root = '/scratch2/ilyak/locDoc/pyfst/models/dffn'
            
            model_root = os.path.join(model_root, mask_name, str(trial))
            command = """CUDA_VISIBLE_DEVICES=1 \
            python hyper_pixelNN.py \
            --data_root /scratch2/ilyak/locDoc/data/hyperspec/datasets \
            --dataset Botswana \
            --npca_components 5 \
            --batch_size 100 \
            --lr 0.0001 \
            --network DFFN_3tower_4depth \
            --network_spatial_size 25 \
            --eval_period 2 \
            --num_epochs 100000 \
            --mask_root  {} \
            --model_root {} \
            --terminate_if_n_nondecreasing_evals 10;""".format(mask_path, model_root)
            
            fw.write("\n")
            fw.write(command)
            fw.write("\n")