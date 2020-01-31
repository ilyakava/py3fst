"""
Create a train/val mask for HSI where labeled pixels of the same class
are chosen from the same site (i.e. form one connected component).
"""

from collections import defaultdict, deque
import copy
import os
import random

import h5py
import hdf5storage
import numpy as np

import hsi_data


import pdb

valid_moves = [(0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1), (1,0), (1,1)]

def get_labeled_sites(labels):
    """
    """
    class_to_labeled_px = defaultdict(list)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            c = labels[i,j]
            if c:
                class_to_labeled_px[c].append((i,j))
    
    
    class_to_components = defaultdict(list)
    for c in class_to_labeled_px.keys():
        nodes = class_to_labeled_px[c]
        
        def extract_connected_component(nodes):
            """Takes out one connected component out of nodes.
            """
            queue = [nodes.pop(0)]
            component = []
            while len(queue):
                p = queue.pop(0)
                component.append(p)
                for m in valid_moves:
                    adj_p = hsi_data.tupsum(m, p)
                    if adj_p in nodes:
                        i = nodes.index(adj_p)
                        queue.append(nodes.pop(i))
            
            return component, nodes
        
        while len(nodes):
            component, nodes = extract_connected_component(nodes)
            # assert check_component_is_connected(component), 'Component was not connected!'
            class_to_components[c].append(component)
    return class_to_components

def check_no_isolated_nodes(nodes):
    for node in nodes:
        nlinks = 0
        for m in valid_moves:
            adj_p = hsi_data.tupsum(m, node)
            if adj_p in nodes:
                nlinks += 1
        if nlinks < 1:
            return False
    return True

def random_connected_subset(nodes_, sz):
    """Returns a random connected subset of a component.
    
    Because the subset needs to be connected, we grow it outwards from a random
    element of the component.
    """
    nodes = copy.copy(nodes_)
    init_len = len(nodes)
    
    
    assert len(nodes) >= sz, 'Cannot find a subset of size %i in a component size %i!' % (sz, len(nodes))
    node_i = random.randint(0,len(nodes)-1)
    queue = [nodes.pop(node_i)]
    subset = []
    
    while len(subset) < sz:
        node = queue.pop(0)
        subset.append(node)
        for m in valid_moves:
            adj_n = hsi_data.tupsum(node, m)
            if adj_n in nodes:
                i = nodes.index(adj_n)
                nodes.pop(i)
                queue.append(adj_n)
    
    return subset[:sz]


def make_train_mask_from_components(mask_dims, class_to_components, n_samp):
    """Randomly sample pixels in 1 connected component to make a training mask.
    """
    mask = np.zeros(mask_dims).astype(int)
    
    for c, components in class_to_components.iteritems():
        valid_components = [comp for comp in components if len(comp) >= n_samp]
        assert len(valid_components), 'No component exists in class %i of size %i or greater' % (c, n_samp)
        comp_i = random.randint(0,len(valid_components)-1)
        comp = valid_components[comp_i]
        
        comp_subset = random_connected_subset(comp, n_samp)
        for node in comp_subset:
            i,j = node
            mask[i,j] = 1
    return mask
    
def save_masks_matlab_style(data_path, name_prefix, train_mask, val_mask):
    """
    """
    
    matfiledata = {}
    # matlab style is to collapse a matrix with columns first
    matfiledata[u'train_mask'] = train_mask.T.flatten()
    matfiledata[u'test_mask'] = val_mask.T.flatten()
    
    id = str(hash( tuple(np.concatenate([matfiledata[u'train_mask'], matfiledata[u'test_mask']])) ))
    
    outfilename = '%s_%s.mat' % (name_prefix, id[-6:])
    outfile = os.path.join(data_path, outfilename)
    
    hdf5storage.write(matfiledata, filename=outfile, matlab_compatible=True)
    print('Saved %s' % outfile)
    


def create_train_val_splits(dataset, n_samp, out_path='/scratch0/ilya/locDoc/data/hyperspec', n_trials=10):
    trainimgname, trainlabelname = hsi_data.dset_filenames_dict[dataset]
    trainimgfield, trainlabelfield = hsi_data.dset_fieldnames_dict[dataset]
    labels = hsi_data.load_labels(trainlabelname, trainlabelfield)
    
    h,w,b = hsi_data.dset_dims[trainimgname]
    nclass = hsi_data.nclass_dict[dataset]
    
    class_to_components = get_labeled_sites(labels)
    
    for trial_i in range(n_trials):
        train_mask = make_train_mask_from_components((h,w), class_to_components, n_samp)
        
        assert train_mask.sum() == (nclass*n_samp), 'Train mask has %i selections expected %i' % (train_mask.sum(), nclass*n_samp)
        label_mask = (labels != 0).astype(int)
        val_mask = label_mask - train_mask
        
        prefix = '%s_strictsinglesite_trainval_s%s_%i' % (dataset, str(n_samp).zfill(2), trial_i)
        
        save_masks_matlab_style(out_path, prefix, train_mask, val_mask)

def main():
    # create_train_val_splits('Botswana', 3)
    create_train_val_splits('PaviaU', 5, n_trials=10)
    create_train_val_splits('PaviaU', 10, n_trials=10)
    create_train_val_splits('PaviaU', 20, n_trials=10)
    create_train_val_splits('PaviaU', 50, n_trials=10)
    
if __name__ == '__main__':
    main()
    