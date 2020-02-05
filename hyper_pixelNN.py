"""DL networks for HSI
"""

from __future__ import division, print_function, absolute_import

from collections import namedtuple
import h5py
import hdf5storage
import itertools
import glob
import logging
import operator
import os
import random
import shutil
import time

import argparse
import numpy as np
from PIL import Image
import scipy.io as sio
from sklearn.svm import SVC, LinearSVC
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
from tqdm import tqdm


import windows as win
from rle import myrlestring
# import salt_baseline as sb
# import salt_data as sd
import fst3d_feat as fst
from hsi_data import load_data, multiversion_matfile_get_field, get_train_val_splits, nclass_dict, dset_dims, dset_filenames_dict, dset_fieldnames_dict, pca_embedding, tupsum
import DFFN
from AP import build_profile, aptoula_net

import pdb

PB_EXPORT_DIR2 = 'best_loss' # protobuffer for tf serving
PB_EXPORT_DIR = 'best_acc' # protobuffer for tf serving
DATA_PATH = '/scratch0/ilya/locDoc/data/hyperspec'
DATASET_PATH = '/scratch0/ilya/locDoc/data/hyperspec/datasets'
PAD_TYPE = 'symmetric'

layerO = namedtuple('layerO', ['strides', 'padding'])

# in nm/pixel
bandwidth_dict = {
    'PaviaU': 9,
    'PaviaCR': 9,
    'Botswana': 2100 / 242.0,
    'KSC': 2100 / 224.0
}
# in meters/pix
spatial_res_dict = {
    'PaviaU': 9,
    'PaviaCR': 9,
    'Botswana': 30.0,
    'KSC': 18.0
}



############ END OF CONSTANTS

def scat3d_to_3d_nxn_2layer(x, reuse=tf.AUTO_REUSE, psis=None, phi=None, layer_params=None, final_size=5):
    """Computes features for a specific pixel.

    Args:
        x: image in (height, width, bands) format
        psis: array of winO struct, filters are in (bands, height, width) format!
        phi: winO struct, filters are in (bands, height, width) format!
        final_size: int, the outputs spatial size (will be a spatial square)
    Output:
        center pixel feature vector in the s
    """
    assert len(layer_params) == 3, 'this network is 2 layers only'
    assert len(psis) == 2, 'this network is 2 layers only'

    
    with tf.variable_scope('Hyper3DNet', reuse=reuse):
        x = tf.transpose(x, [2, 0, 1])

        x = tf.expand_dims(x, 0)
        x = tf.expand_dims(x, -1)
        # x is (1, bands, h, w, 1)
        U1 = fst.scat3d(x, psis[0], layer_params[0])
        # U1 is (1, bands, h, w, lambda1)

        U1 = tf.transpose(U1, [0, 4, 1, 2, 3])
        # U1 is (1, lambda1, bands, h, w)

        # downsampling amounts
        ds_amounts = {
            9: 7,
            7: 5,
            5: 3,
            3: 1
        }
        lambda1_d = ds_amounts[psis[0].kernel_size[1]]; band1_d = 3
        lambda2_d = ds_amounts[psis[1].kernel_size[1]]; band2_d = 3
        lambdax_d = ds_amounts[phi.kernel_size[1]];
        
        U1 = tf.layers.max_pooling3d(U1, (lambda1_d,band1_d,1), (lambda1_d,band1_d,1), padding='same')

        U1 = tf.transpose(U1, [1,2,3,4,0])
        # U1 is (lambda1, bands, h, w, 1)
        
        U2s = []
        # only procede with increasing frequency paths
        for res_i, used_params in enumerate(psis[0].filter_params[::lambda1_d]):
            increasing_psi = win.fst3d_psi_factory(psis[1].kernel_size, used_params)
            if increasing_psi.nfilt > 0:
                U2s.append(fst.scat3d(U1[res_i:(res_i+1),:,:,:,:], increasing_psi, layer_params[1]))
        
        U2 = tf.concat(U2s, 4)
        # U2 is (1,bands,h,w,lambda2)
        
        U2 = tf.transpose(U2, [0, 4, 1, 2, 3])
        # U2 is (1, lambda2, bands, h, w)
      
        U2 = tf.layers.max_pooling3d(U2, (lambda2_d,band2_d,1), (lambda2_d,band2_d,1), padding='same')
        
        U2 = tf.transpose(U2, [1, 2, 3, 4, 0])
        # U2 is (lambda2, bands, h, w, 1)

        # convolve with phi
        S2 = fst.scat3d(U2, phi, layer_params[2])

        def slice_idxs(sig_size, kernel_size):
            """
            return slice indexes to slice signal so that after convolving with
            the kernel it is the desired final size
            """
            def slice_idx(s, k, f):
                """
                s: signal size
                k: kernel size
                f: final size
                """
                if k % 2 == 0:
                    raise('not implemented even padding')
                else:
                    return int((s - k - f)//2)
            final_size_ = [1,final_size,final_size]
            return [slice_idx(s,k,f-1) for s,k,f in zip(sig_size, kernel_size,final_size_)]

        # we will stack S0,S1,S2 so we need to trim them to be the same size
        # after each of them are convolved with phi
        [p1b, p1h, p1w] = slice_idxs(U1.shape[1:4], phi.kernel_size)
        [p2b, p2h, p2w] = slice_idxs(x.shape[1:4], phi.kernel_size)

        S1 = fst.scat3d(U1[:, :,(p1h):-(p1h), (p1w):-(p1w), :], phi, layer_params[2])
        S0 = fst.scat3d(x[:, :,(p2h):-(p2h), (p2w):-(p2w), :], phi, layer_params[2])

        # just to get the size down to 1 (flattening step)
        S0 = tf.reshape(S0, [-1, final_size, final_size, 1])
        S1 = tf.reshape(S1, [-1, final_size, final_size, 1])
        S2 = tf.reshape(S2, [-1, final_size, final_size, 1])
        
        SX = tf.concat([S0,S1,S2], 0)
        # SX is (lambdax, h, w, 1)
        
        SX = tf.expand_dims(SX, 0)
        SX = tf.layers.max_pooling3d(SX, (lambdax_d,1,1), (lambdax_d,1,1), padding='same')
        
        SX = tf.squeeze(SX)
        # SX is (channels, h, w)

    return tf.transpose(SX, [1, 2, 0])
    
def gabor_mag_filter(x, reuse=tf.AUTO_REUSE, psis=None, layer_params=None, final_size=5):
    """Computes gabor magnitude features for a specific pixel.

    Args:
        x: image in (height, width, bands) format
        psis: array of length 1 winO struct, filters are in (bands, height, width) format!
        final_size: int, the outputs spatial size (will be a spatial square)
    Output:
        center pixel feature vector in the s
    """
    assert len(layer_params) == 1, 'this network is 1 layer only'
    assert len(psis) == 1, 'this network is 1 layer only'
    
    with tf.variable_scope('Hyper3DNet', reuse=reuse):
        x = tf.transpose(x, [2, 0, 1])

        x = tf.expand_dims(x, 0)
        x = tf.expand_dims(x, -1)
        # x is (1, bands, h, w, 1)
        U1 = fst.scat3d(x, psis[0], layer_params[0])
        # U1 is (1, bands, h, w, lambda1)

        U1 = tf.transpose(U1, [0, 4, 1, 2, 3])
        # U1 is (1, lambda1, bands, h, w)
        
        ds_amounts = {
            9: 7,
            7: 5,
            5: 3,
            3: 1
        }
        lambda1_d = ds_amounts[psis[0].kernel_size[1]]; band1_d = 3
        
        U1 = tf.layers.max_pooling3d(U1, (lambda1_d,band1_d,1), (lambda1_d,band1_d,1), padding='same')
        U1 = tf.reshape(U1, [-1, final_size, final_size])

    return tf.transpose(U1, [1, 2, 0])

def hyper_3x3_net(x_dict, dropout, reuse, is_training, n_classes):
    """
    x should be (batch, channel, h, w)
    """
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, like in MNIST example
        x = x_dict['subimages']

        # 1x1 conv replaces PCA step
        conv1 = tf.layers.conv2d(x, 1024, 1, data_format='channels_first')
        # Convolution Layer with filters of size 3
        conv2 = tf.layers.conv2d(conv1, 512, 3, activation=tf.nn.relu, padding='same', data_format='channels_first')
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2, data_format='channels_first')
        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1, 500)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        fc2 = tf.layers.dense(fc1, 100)
        out = tf.layers.dense(fc2, n_classes)

    return out

def deng_cnn(x_dict, dropout, reuse, is_training, n_classes):
    """
    x should be (batch, channel, h, w)
    
    Notes:
    Not sure about max pooling size
    bs=64, dropout=0.6, 
    """
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = x_dict['subimages']

        conv1 = tf.layers.conv2d(x, 64, 4, activation=None)
        conv1 = tf.layers.batch_normalization(conv1)
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.layers.max_pooling2d(conv1, 4, 1, padding='same')

        fc1 = tf.contrib.layers.flatten(conv1)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        out = tf.layers.dense(fc1, n_classes)

    return out

def yu_net(x_dict, dropout, reuse, is_training, n_classes):
    """
    Based on:
    
    Convolutional neural networks for hyperspectral image classification
    Yu, Jia, Xu
    
    bs=16, dropout=0.6, input should be (batch,5,5,nbands)
    """
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = x_dict['subimages']
        # x should be (batch, h, w, channel)

        conv1 = tf.layers.conv2d(x, 128, 1, activation=None)
        conv1 = tf.layers.batch_normalization(conv1)
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.layers.dropout(conv1, rate=dropout, training=is_training)

        conv2 = tf.layers.conv2d(conv1, 64, 1, activation=None)
        conv2 = tf.layers.batch_normalization(conv2)
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.layers.dropout(conv2, rate=dropout, training=is_training)

        conv3 = tf.layers.conv2d(conv2, n_classes, 1, activation=None)
        conv3 = tf.nn.relu(conv3)
        out = tf.layers.average_pooling2d(conv3, 5, 1, name="out")

    return tf.squeeze(out, axis=(1,2))

def yu2_net(x_dict, dropout, reuse, is_training, n_classes):
    """
    x should be (batch, channel, h, w)
    """
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, like in MNIST example
        x = x_dict['subimages']
        # x should be (batch, h, w, channel)

        conv1 = tf.layers.conv2d(x, 128, 1, activation=None)
        conv1 = tf.layers.average_pooling2d(conv1, 2, 1, padding='same')
        conv1 = tf.nn.relu(conv1)

        conv2 = tf.layers.conv2d(conv1, 64, 1, activation=None)
        conv2 = tf.layers.average_pooling2d(conv2, 2, 1, padding='same')
        conv2 = tf.nn.relu(conv2)

        conv3 = tf.layers.conv2d(conv2, n_classes, 1, activation=None)
        out = tf.layers.average_pooling2d(conv3, 5, 1)

    return tf.squeeze(out)

def fst_net(x_dict, dropout, reuse, is_training, n_classes):
    """Network to follow ST preprocessing.
    
    x should be (batch, h, w, channel)
    """
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = x_dict['subimages']

        ### 64convfc
        conv2 = tf.layers.conv2d(x, 64, 1, activation=None)
        conv2 = tf.layers.batch_normalization(conv2)
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.layers.dropout(conv2, rate=dropout, training=is_training)
        
        fc1 = tf.contrib.layers.flatten(conv2)
        
        out = tf.layers.dense(fc1, n_classes)
        


        # conv0 = tf.layers.conv2d(x, 256, 1, activation=None)
        # #conv0 = tf.layers.batch_normalization(conv0)
        # conv0 = tf.nn.relu(conv0)
        # conv0 = tf.layers.dropout(conv0, rate=dropout, training=is_training)

        # conv1 = tf.layers.conv2d(x, 128, 1, activation=None)
        # conv1 = tf.layers.batch_normalization(conv1)
        # conv1 = tf.nn.relu(conv1)
        # conv1 = tf.layers.dropout(conv1, rate=dropout, training=is_training)

        

        # conv2 = tf.layers.conv2d(x, 64, 1, activation=None)
        # conv2 = tf.layers.batch_normalization(conv2)
        # conv2 = tf.nn.relu(conv2)
        # conv2 = tf.layers.dropout(conv2, rate=dropout, training=is_training)
        
        # fc1 = tf.contrib.layers.flatten(conv2)
        # fc1 = tf.layers.dense(fc1, 100)
        # fc1 = tf.nn.relu(fc1)
        # fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        
        # out = tf.layers.dense(fc1, n_classes)

        # conv3 = tf.layers.conv2d(conv2, n_classes, 1, activation=None)
        # conv3 = tf.nn.relu(conv3)
        # out = tf.layers.average_pooling2d(conv3, 3, 1, name="out")

    return tf.squeeze(out)


def cube_iter(data, batch_size, addl_padding=(4,4,0)):
    """Yields iterator through data for predition
    """
    
    [height, width, nbands] = data.shape
    
    all_pixels = np.array(list(itertools.product(range(width),range(height))))
    
    ap = np.array(addl_padding)
    assert np.all(ap % 2 == 0), 'Assymetric is not supported'
    
    padded_data = np.pad(data, ((ap[0]//2,ap[0]//2),(ap[1]//2,ap[1]//2),(ap[2]//2,ap[2]//2)), PAD_TYPE)
    
    batch_item_shape = tuple(map(operator.add, addl_padding, (1,1,data.shape[2])))
    
    batchX = np.zeros((batch_size,) + batch_item_shape, dtype=np.float32)
    
    for pixel_i, pixel in enumerate(all_pixels):
        batch_i = pixel_i % batch_size
        # this iterates through columns first
        [pixel_x, pixel_y] = pixel
        batchX[batch_i,:,:,:] = padded_data[pixel_y:(pixel_y+ap[0]+1), pixel_x:(pixel_x+ap[1]+1), :]
        if batch_i == (batch_size - 1):
            yield batchX
    
    leftover = len(all_pixels) % batch_size
    if leftover != 0:
        yield batchX[:leftover]

st_net_spec_struct = namedtuple('st_net_spec_struct', ['psi1', 'psi2', 'phi'])
paviaU_spec = st_net_spec_struct([9,9,9],[9,9,9],[9,9,9])
spec_7 = st_net_spec_struct([7,7,7],[7,7,7],[7,7,7])
def net_addl_padding_from_spec(spec):
    """
    Here is a potential for huge confusion:
    padding is (h,w,b)
    BUT
    specs are (b,h,w)
    """
    b, h, w = list(tupsum(tuple(spec.psi1), tuple(spec.psi2), tuple(spec.phi), (-3,-3,-3)))
    return (h,w,b)
def spec_to_str(spec):
    a = spec.psi1
    b = spec.psi2
    c = spec.phi
    return '%i-%i-%i_%i-%i-%i_%i-%i-%i' % (a[0],a[1],a[2],b[0],b[1],b[2],c[0],c[1],c[2])

def dlgrf_filter(data, kernel_size=[21,21,21], sigmas=[2,2,2], patch_size=101):
    """
    """
    s = time.time()
    filter_obj = win.dlrgf_factory(kernel_size, sigmas)
    layer_params = layerO((1,1,1), 'valid')
    
    [height, width, nbands] = data.shape
    hyper_pixel_shape = (1, 1,data.shape[2])
    
    padding = (kernel_size[0]-1, kernel_size[1]-1, 0)
    ap = np.array(padding)
    assert np.all(ap[:2] % 2 == 0), 'Assymetric padding is not supported'
    
    padded_data = np.pad(data, ((ap[0]//2,ap[0]//2),(ap[1]//2,ap[1]//2),(ap[2]//2,ap[2]//2)), PAD_TYPE)
    
    # cover the data with patches
    patch_xs = [max(0,width - (x*patch_size)) for x in range(1, width // patch_size + 2)]
    patch_ys = [max(0,height - (y*patch_size)) for y in range(1, height // patch_size + 2)]
    patch_ul_corners = itertools.product(patch_xs, patch_ys) # upper left corners

    addl_spatial_pad = (patch_size-1, patch_size-1, 0)
    batch_item_shape = tupsum(hyper_pixel_shape, padding, addl_spatial_pad)
    
    x = tf.placeholder(tf.float32, shape=batch_item_shape)
    feat = fst.conv3dfeat(x, filter_obj, layer_params, patch_size)
    feat_shape = tuple([int(d) for d in feat.shape])
    
    new_data = np.zeros((height,width,feat_shape[2]))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for pixel_i, pixel in enumerate(tqdm(patch_ul_corners, desc='Performing DLGRF: ', total=len(patch_xs)*len(patch_ys))):
            [pixel_x, pixel_y] = pixel
            subimg = padded_data[pixel_y:(patch_size+pixel_y+ap[0]), pixel_x:(patch_size+pixel_x+ap[1]), :]
            feed_dict = {x: subimg}
            new_data[pixel_y:(patch_size+pixel_y), pixel_x:(patch_size+pixel_x)] = sess.run(feat, feed_dict)

    tf.reset_default_graph()
    print('DLGRF preprocessing finished in %is.' % int(time.time() - s))
    return new_data
    

def preprocess_data(data, st_net_spec, patch_size=51):
    """ST preprocess the whole data cube.
    
    Pad data, then pass in as few subsets of this data.
    """
    s = time.time()
    reuse = tf.AUTO_REUSE
    
    # Network info
    layer_params = layerO((1,1,1), 'valid')
    preprocessing_mode = 'ST'
    if st_net_spec.psi1 and st_net_spec.psi2 and st_net_spec.phi:
        print('Will perform Scattering...')
        psi1 = win.fst3d_psi_factory(st_net_spec.psi1)
        psi2 = win.fst3d_psi_factory(st_net_spec.psi2)
        psis=[psi1,psi2]
        phi = win.fst3d_phi_window_3D(st_net_spec.phi)
        net_addl_padding = net_addl_padding_from_spec(st_net_spec)
        layer_params=[layer_params, layer_params, layer_params]
    elif st_net_spec.psi1 and st_net_spec.psi2 is None and st_net_spec.phi is None: # just one layer gabor
        print('Will perform Gabor filtering...')
        psis = [win.gabor_psi_factory(st_net_spec.psi1)]
        b, h, w = list(tupsum(tuple(st_net_spec.psi1), (-1,-1,-1)))
        net_addl_padding = (h,w,b)
        layer_params=[layer_params]
        preprocessing_mode = 'Gabor'
    else:
        raise ValueError('This ST spec is not supported')

    # END Network info
    

    [height, width, nbands] = data.shape
    hyper_pixel_shape = (1, 1,data.shape[2])

    all_pixels = np.array(list(itertools.product(range(width),range(height))))
    
    ap = np.array(net_addl_padding)
    assert np.all(ap[:2] % 2 == 0), 'Assymetric padding is not supported'
    
    padded_data = np.pad(data, ((ap[0]//2,ap[0]//2),(ap[1]//2,ap[1]//2),(ap[2]//2,ap[2]//2)), PAD_TYPE)
    
    # cover the data with patches
    patch_xs = [max(0,width - (x*patch_size)) for x in range(1, width // patch_size + 2)]
    patch_ys = [max(0,height - (y*patch_size)) for y in range(1, height // patch_size + 2)]
    patch_ul_corners = itertools.product(patch_xs, patch_ys) # upper left corners

    addl_spatial_pad = (patch_size-1, patch_size-1, 0)
    batch_item_shape = tupsum(hyper_pixel_shape, net_addl_padding, addl_spatial_pad)
    
    x = tf.placeholder(tf.float32, shape=batch_item_shape)
    print('Compiling Graph...')
    compile_start = time.time()
    if preprocessing_mode == 'ST':
        feat = scat3d_to_3d_nxn_2layer(x, reuse, psis, phi, layer_params, final_size=patch_size)
    elif preprocessing_mode == 'Gabor':
        feat = gabor_mag_filter(x, reuse, psis, layer_params, final_size=patch_size)
    compile_time = time.time() - compile_start
    feat_shape = tuple([int(d) for d in feat.shape])
    print('Graph Compiled %is. Feature dimension per pixel is now %i.' % (int(compile_time), feat_shape[2]))
    assert feat_shape[0] == feat_shape[1], 'ST spatial output is not square!'
    assert feat_shape[0] == patch_size, 'ST spatial output size is %i, expected %i!' % (feat_shape[0], patch_size)

    new_data = np.zeros((height,width,feat_shape[2]))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for pixel_i, pixel in enumerate(tqdm(patch_ul_corners, desc=('Performing %s: ' % preprocessing_mode), total=len(patch_xs)*len(patch_ys))):
            [pixel_x, pixel_y] = pixel
            subimg = padded_data[pixel_y:(patch_size+pixel_y+ap[0]), pixel_x:(patch_size+pixel_x+ap[1]), :]
            feed_dict = {x: subimg}
            new_data[pixel_y:(patch_size+pixel_y), pixel_x:(patch_size+pixel_x)] = sess.run(feat, feed_dict)

    tf.reset_default_graph()
    print('ST preprocessing finished in %is.' % int(time.time() - s))
    return new_data


def load_or_preprocess_data(data, write_path, backup_write_root, st_net_spec=paviaU_spec, st_patch_size=51):
    """
    If write_path is none then no saving/loading will be done.
    backup_write_root: only used when write_path is not None and is invalid
    """
    if (write_path is not None) and os.path.isfile(write_path):
        data = np.load(write_path)['data']
        print('Loaded %s Successfully.' % write_path)
    else:
        data = preprocess_data(data, st_net_spec, patch_size=st_patch_size)
        if (write_path is not None):
            # save it
            try:
                np.savez(write_path, data=data)
                print('Saved %s' % write_path)
            except:
                print('Could not save %s' % write_path)
                npz_path = os.path.join(backup_write_root, 'preprocess_data.npz')
                np.savez(npz_path, data=data)
                print('Saved %s' % npz_path)
    return data

# hyper_3x3_net
network_dict = {
    'fst_net': fst_net,
    'deng': deng_cnn,
    'yu': yu_net,
    'DFFN_3tower_5depth': DFFN.DFFN_3tower_5depth,
    'DFFN_3tower_4depth': DFFN.DFFN_3tower_4depth,
    'DFFN_3tower_3depth': DFFN.DFFN_3tower_3depth,
    'DFFN_3tower_2depth': DFFN.DFFN_3tower_2depth,
    'DFFN_3tower_1depth': DFFN.DFFN_3tower_1depth,
    'aptoula': aptoula_net,
}
sts_dict = {
    'paviaU': st_net_spec_struct([9,9,9],[9,9,9],[9,9,9]),
    '7': st_net_spec_struct([7,7,7],[7,7,7],[7,7,7]),
    '9': st_net_spec_struct([9,9,9],[9,9,9],[9,9,9]),
    '5': st_net_spec_struct([5,5,5],[5,5,5],[5,5,5]),
    '3': st_net_spec_struct([3,3,3],[3,3,3],[3,3,3]),
    'KSC': st_net_spec_struct([5,9,9],[5,7,7],[5,7,7]),
    'PU_SSS': st_net_spec_struct([9,7,7],[9,3,3],[9,3,3]),
    'IP_SSS': st_net_spec_struct([5,9,9],[5,5,5],[5,5,5]),
    'IP_gabor': st_net_spec_struct([5,9,9],None,None),
}

def many_svm_evals(args):
    """
    
    Like svm_predict but without the full image prediction, and for many masks.
    """
    bs = args.batch_size
    n_classes = nclass_dict[args.dataset]
    trainimgname, trainlabelname = dset_filenames_dict[args.dataset]
    trainimgfield, trainlabelfield = dset_fieldnames_dict[args.dataset]
    st_net_spec = sts_dict[args.st_type]
    
    mask_list_f = open(args.svm_multi_mask_file_list, "r")
    masks = [line.strip() for line in mask_list_f.readlines() if line != "\n"]
    mask_list_f.close()
    valid_masks = [m for m in masks if m and os.path.exists(m)]
    print("%i/%i masks valid in provided file." % (len(valid_masks), len(masks)))
    
    data, labels = load_data(trainimgname, trainimgfield, trainlabelname, trainlabelfield, dataset_path=args.data_root)
        
    if args.fst_preprocessing:
        data = load_or_preprocess_data(data, args.preprocessed_data_path, args.model_root, st_net_spec=st_net_spec, st_patch_size=args.st_patch_size)
    elif args.dlgrf_preprocessing:
        data = dlgrf_filter(data)
    
    height, width, bands = dset_dims[trainimgname]
    
    results = {}
    
    for mask_path in valid_masks:
        # 2 mask roots are not currently supported, if datasets for train/val are different
        train_mask = multiversion_matfile_get_field(mask_path, 'train_mask')
        val_mask = multiversion_matfile_get_field(mask_path, 'test_mask')
           
        s = args.network_spatial_size - 1
        trainX, trainY, valX, valY = get_train_val_splits(data, labels, train_mask, val_mask, (s,s,0))
        
        print('starting training')
        start = time.time()
        clf = SVC(kernel='linear')
        clf.fit(trainX.squeeze(), trainY)
        end = time.time()
        print('Training done. Took %is' % int(end - start))
        
        n_correct = 0
        for i in tqdm(range(0,valY.shape[0],bs), desc='Getting Val Accuracy'):
            p_label = clf.predict(valX.squeeze()[i:i+bs]);
            n_correct += (p_label == valY[i:i+bs]).sum()
        acc = float(n_correct) / valY.shape[0]
        print('Done with %s' % mask_path )
        print('SVM has validation accuracy %.2f' % (acc*100) )
        results[mask_path] = acc
    
    npz_path = os.path.join(args.model_root, 'SVM_results_%i.npz' % (random.randint(0,1e10)))
    np.savez(npz_path, results=results)
    print('Saved %s' % npz_path)

def svm_predict(args):
    """Train and Predict using an SVM features on features.
    
    Features can be the raw HSI cube or the scattering features.
    """
    bs = args.batch_size
    n_classes = nclass_dict[args.dataset]
    trainimgname, trainlabelname = dset_filenames_dict[args.dataset]
    trainimgfield, trainlabelfield = dset_fieldnames_dict[args.dataset]
    st_net_spec = sts_dict[args.st_type]
    
    # 2 mask roots are not currently supported, if datasets for train/val are different
    train_mask = multiversion_matfile_get_field(args.mask_root, 'train_mask')
    val_mask = multiversion_matfile_get_field(args.mask_root, 'test_mask')
    
    data, labels = load_data(trainimgname, trainimgfield, trainlabelname, trainlabelfield, dataset_path=args.data_root)
    
    if args.fst_preprocessing:
        data = load_or_preprocess_data(data, args.preprocessed_data_path, args.model_root, st_net_spec=st_net_spec, st_patch_size=args.st_patch_size)
    
    height, width, bands = dset_dims[trainimgname]
       
    s = args.network_spatial_size - 1
    trainX, trainY, valX, valY = get_train_val_splits(data, labels, train_mask, val_mask, (s,s,0))
    
    print('starting training')
    start = time.time()
    clf = SVC(kernel='linear')
    clf.fit(trainX.squeeze(), trainY)
    end = time.time()
    print('Training done. Took %is' % int(end - start))
    
    n_correct = 0
    for i in tqdm(range(0,valY.shape[0],bs), desc='Getting Val Accuracy'):
        p_label = clf.predict(valX.squeeze()[i:i+bs]);
        n_correct += (p_label == valY[i:i+bs]).sum()
    acc = float(n_correct) / valY.shape[0]
    print('SVM has validation accuracy %.2f' % (acc*100) )

    # test everything
    y_predicted = []
    nbatches = (height * width // bs) + 1
    s = args.network_spatial_size - 1
    for bi, batchX in enumerate(tqdm(cube_iter(data, bs, addl_padding=(s,s,0)), desc='Predicting', total=nbatches)):
        y_predicted += clf.predict(batchX.squeeze()).astype(int).tolist()

    pred_image = np.array(y_predicted).reshape((width, height)).T
    
    imgmatfiledata = {}
    imgmatfiledata[u'imgHat'] = pred_image
    groundtruthfilename = os.path.splitext(trainlabelname)[0]
    imgmatfiledata[u'groundtruthfilename'] = '%s_%s.mat' % (groundtruthfilename, args.network)
    hdf5storage.write(imgmatfiledata,
        filename=os.path.join(args.model_root, imgmatfiledata[u'groundtruthfilename']),
        matlab_compatible=True)
    
    print('Saved %s' % os.path.join(args.model_root, imgmatfiledata[u'groundtruthfilename']))
    
    npz_path = os.path.join(args.model_root, '%s_SVM.npz' % (groundtruthfilename))
    np.savez(npz_path, pred_image=pred_image)
    print('Saved %s' % npz_path)

def predict(args):
    bs = args.batch_size
    network = network_dict[args.network]
    n_classes = nclass_dict[args.dataset]
    trainimgname, trainlabelname = dset_filenames_dict[args.dataset]
    trainimgfield, trainlabelfield = dset_fieldnames_dict[args.dataset]
    st_net_spec = sts_dict[args.st_type]
    
    data, labels = load_data(trainimgname, trainimgfield, trainlabelname, trainlabelfield, dataset_path=args.data_root)
    
    if args.fst_preprocessing:
        data = load_or_preprocess_data(data, args.preprocessed_data_path, args.model_root, st_net_spec=st_net_spec, st_patch_size=args.st_patch_size)
    elif args.npca_components is not None:    
        data = pca_embedding(data, n_components=args.npca_components)
        if args.attribute_profile:
            data = build_profile(data) 

    height, width, bands = dset_dims[trainimgname]
    
    # if there are multiple saved *pb files get the newest
    best_models_dir = os.path.join(args.model_root, PB_EXPORT_DIR)
    subdirs = [x for x in os.listdir(best_models_dir) if os.path.isdir(os.path.join(best_models_dir, x)) and 'temp' not in str(x)]
    latest = sorted(subdirs)[-1]
    full_model_dir = os.path.join(best_models_dir, latest)
    
    with tf.Session() as sess:
        
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], full_model_dir)
        predictor   = tf.contrib.predictor.from_saved_model(full_model_dir)
        
        y_predicted = []
        nbatches = (height * width // bs) + 1
        # TODO do this for fst preprocessed data
        s = args.network_spatial_size - 1
        for bi, batchX in enumerate(tqdm(cube_iter(data, bs, addl_padding=(s,s,0)), desc='Predicting', total=nbatches)):
            y_predicted += list(predictor({"subimages": batchX })['output'])
        
        pred_image = np.array(y_predicted).reshape((width, height)).T
        
        imgmatfiledata = {}
        imgmatfiledata[u'imgHat'] = pred_image
        groundtruthfilename = os.path.splitext(trainlabelname)[0]
        imgmatfiledata[u'groundtruthfilename'] = '%s_%s.mat' % (groundtruthfilename, args.network)
        hdf5storage.write(imgmatfiledata,
            filename=os.path.join(args.model_root, imgmatfiledata[u'groundtruthfilename']),
            matlab_compatible=True)
        
        print('Saved %s' % os.path.join(args.model_root, imgmatfiledata[u'groundtruthfilename']))
        
        npz_path = os.path.join(args.model_root, '%s_%s.npz' % (groundtruthfilename, args.network))
        np.savez(npz_path, pred_image=pred_image)
        print('Saved %s' % npz_path)


def train(args):
    bs = args.batch_size
    network = network_dict[args.network]
    n_classes = nclass_dict[args.dataset]
    trainimgname, trainlabelname = dset_filenames_dict[args.dataset]
    trainimgfield, trainlabelfield = dset_fieldnames_dict[args.dataset]
    st_net_spec = sts_dict[args.st_type]
    
    # 2 mask roots are not currently supported, if datasets for train/val are different
    train_mask = multiversion_matfile_get_field(args.mask_root, 'train_mask')
    val_mask = multiversion_matfile_get_field(args.mask_root, 'test_mask')
    
    data, labels = load_data(trainimgname, trainimgfield, trainlabelname, trainlabelfield, dataset_path=args.data_root)
    
    if args.fst_preprocessing:
        data = load_or_preprocess_data(data, args.preprocessed_data_path, args.model_root, st_net_spec=st_net_spec, st_patch_size=args.st_patch_size)
    elif args.npca_components is not None:    
        data = pca_embedding(data, n_components=args.npca_components)
        if args.attribute_profile:
            data = build_profile(data)
            
    s = args.network_spatial_size - 1
    trainX, trainY, valX, valY = get_train_val_splits(data, labels, train_mask, val_mask, (s,s,0))
    nlabeled = len(trainY)
    bs = min(nlabeled, bs)
    n_eval = args.n_eval
    if n_eval < 1:
        n_eval = int(nlabeled * n_eval)
    else:
        n_eval = int(n_eval)   
    
    best_loss = float("inf")
    best_acc = 0
    acc_at_best_loss = 0
    train_set_size = trainX.shape[0]
    val_set_size = valX.shape[0]
    steps_per_epoch = train_set_size // bs
    max_steps = args.num_epochs * steps_per_epoch
    test_acc_at_best_eval_loss = 0
    test_acc_at_best_eval_acc = 0
    
    ############### END OF SETUP
    
    # Define the model function (following TF Estimator Template)
    def model_fn(features, labels, mode):
        
    
        # Build the neural network
        # Because Dropout have different behavior at training and prediction time, we
        # need to create 2 distinct computation graphs that still share the same weights.
        logits_train = network(features, args.dropout, reuse=False,
                                is_training=True, n_classes=n_classes)
        logits_val = network(features, args.dropout, reuse=True,
                                is_training=False, n_classes=n_classes)
    
        # Predictions
        pred_classes = tf.argmax(logits_val, axis=1)
        pred_probas = tf.nn.softmax(logits_val)
    
        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)
    
            # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        train_op = optimizer.minimize(loss_op,
                                      global_step=tf.train.get_global_step())
    
        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    
        # tf.summary.scalar('min', loss_op)
    
        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op})
    
        return estim_specs
    
    ###############################

    model = tf.estimator.Estimator(model_fn, model_dir=args.model_root)
    
    def identity_serving_input_receiver_fn():
        """
        This function is supposed to translate what the user gives the model
        to what should actually be given to the model.
        In our case this is the identity function.
        
        A useful way to use a 'serving_input_receiver_fn' would be to provide a
        string of image bytes and read/convert it into a tensor of numbers for
        the model.
        """
        serialized_tf_example = tf.placeholder(dtype=tf.float32, shape=[None, trainX.shape[1], trainX.shape[2], trainX.shape[3]]) # , name='input_tensors'
        user_input = {'subimages': serialized_tf_example }
        model_input = user_input
        return tf.estimator.export.ServingInputReceiver(model_input, user_input)
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'subimages': trainX[:,:,:,:]}, y=trainY[:],
        batch_size=bs, num_epochs=args.eval_period, shuffle=True)
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'subimages': valX[:n_eval,:,:,:]}, y=valY[:n_eval],
        batch_size=bs, shuffle=False)
        
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'subimages': valX[n_eval:,:,:,:]}, y=valY[n_eval:],
        batch_size=bs, shuffle=False)
    
    if args.eval_and_exit:
        e = model.evaluate(eval_input_fn)
        print("Validation Accuracy: {:.4f}".format(e['accuracy']))
        return 0
    
    n_nondecreasing_evals = 0
    for i in range(args.num_epochs // args.eval_period):
        model.train(train_input_fn)

        e = model.evaluate(eval_input_fn, name='eval')
        
        if e['accuracy'] > best_acc:
            tf.logging.info("{:06d}: High Accuracy. Saving model with Validation Accuracy: {:.4f}".format(i*args.eval_period, e['accuracy']))
            if os.path.isdir(os.path.join(args.model_root, PB_EXPORT_DIR2)):
                shutil.rmtree(os.path.join(args.model_root, PB_EXPORT_DIR2))
            model.export_savedmodel(os.path.join(args.model_root, PB_EXPORT_DIR2), identity_serving_input_receiver_fn)
            
        if e['loss'] < best_loss:
            tf.logging.info("{:06d}: Low Loss. Saving model with Validation Accuracy: {:.4f}".format(i*args.eval_period, e['accuracy']))
            if os.path.isdir(os.path.join(args.model_root, PB_EXPORT_DIR)):
                shutil.rmtree(os.path.join(args.model_root, PB_EXPORT_DIR))
            model.export_savedmodel(os.path.join(args.model_root, PB_EXPORT_DIR), identity_serving_input_receiver_fn)
            acc_at_best_loss = e['accuracy']
            n_nondecreasing_evals = 0
        else:
            n_nondecreasing_evals += 1
            tf.logging.info("Eval Loss did not decrease %i/%i times." % (n_nondecreasing_evals, args.terminate_if_n_nondecreasing_evals))
        
        if e['accuracy'] > best_acc or e['loss'] < best_loss:
            
            test_e = model.evaluate(test_input_fn, name='test')
            if e['accuracy'] > best_acc:
                test_acc_at_best_eval_acc = test_e['accuracy']
            if e['loss'] < best_loss:
                test_acc_at_best_eval_loss = test_e['accuracy']
            
        best_loss = min(best_loss, e['loss'])
        best_acc = max(best_acc, e['accuracy'])
        tf.logging.info("{:06d}: Validation Accuracy: {:.4f} (At lowest loss: {:.4f}) (Best Ever: {:.4f})".format(i*args.eval_period, e['accuracy'], acc_at_best_loss, best_acc))
        tf.logging.info("{:06d}: Test Accuracy: Best by Eval Acc: {:.4f}. Best by Eval Loss: {:.4f}".format(i*args.eval_period, test_acc_at_best_eval_acc, test_acc_at_best_eval_loss))
        
        tp = float(n_eval) / nlabeled # test percentage
        overall_acc_at_best_eval_acc = tp*test_acc_at_best_eval_acc + (1-tp)*best_acc
        overall_acc_at_best_eval_loss = tp*test_acc_at_best_eval_loss + (1-tp)*acc_at_best_loss
        tf.logging.info("{:06d}: Overall Accuracy: Best by Eval Acc: {:.4f}. Best by Eval Loss: {:.4f}".format(i*args.eval_period, overall_acc_at_best_eval_acc, overall_acc_at_best_eval_loss))
        
        npz_path = os.path.join(args.model_root, 'results.npz')
        np.savez(npz_path, results={args.mask_root: overall_acc_at_best_eval_acc})
        print('Saved %s' % npz_path)
        
        if n_nondecreasing_evals >= args.terminate_if_n_nondecreasing_evals:
            tf.logging.info("Terminating at epoch %i because eval loss did not decrease for %i consecutive evals (%i epochs)" % (i * args.eval_period, n_nondecreasing_evals, n_nondecreasing_evals*args.eval_period))
            return 0

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--dataset', required=True,
                      help='Name of dataset to run on')
    parser.add_argument('--model_root', required=True,
                      help='Full path of where to output the results of training.')
    parser.add_argument('--mask_root', default=None,
                      help='Required unless supplying --predict. Full path to mask to use to generate train/val set.')
    ### Optional args
    
    # Modes of operation
    parser.add_argument(
        '--svm_predict', action='store_true', default=False,
        help='If true predict on the whole HSI image using an SVM on data features (default: %(default)s)')
    parser.add_argument(
        '--predict', action='store_true', default=False,
        help='If true predict on the whole HSI image (default: %(default)s)')
    parser.add_argument(
        '--eval_and_exit', action='store_true', default=False,
        help='If true ...')
    parser.add_argument(
        '--svm_multi_mask_file_list', type=str, default=None,
        help='If not None, run eval using an SVM on data features. This is a path to a txt file with masks to use (default: %(default)s)')
    # Important optionals
    parser.add_argument(
        '--data_root', type=str, default='/scratch0/ilya/locDoc/data/hyperspec/datasets',
        help='Where to find the HSI data cube .mat files (default: %(default)s)')
    parser.add_argument(
        '--network', type=str, default=None,
        help='Name of network to run')
    parser.add_argument(
        '--network_spatial_size', type=int, default=5,
        help='The spatial size of the patches that the network expects as input.')
    parser.add_argument(
        '--fst_preprocessing', action='store_true', default=False,
        help='If true load data with fourier scattering preprocessing (default: %(default)s)')
    parser.add_argument(
        '--dlgrf_preprocessing', action='store_true', default=False,
        help='If true load data with DLGRF preprocessing (default: %(default)s)')
    parser.add_argument(
        '--st_type', default='paviaU',
        help='Used when supplying --fst_preprocessing.')
    # Hyperparams
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Learning rate to use (default: %(default)s)')
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Batch Size')
    parser.add_argument(
        '--dropout', type=float, default=0.6,
        help='Dropout rate.')
    parser.add_argument(
        '--npca_components', type=int, default=None,
        help='....')
    # Other
    parser.add_argument(
        '--num_epochs', type=int, default=10000,
        help='Number of epochs to run training for.')
    parser.add_argument(
        '--eval_period', type=int, default=50,
        help='Eval after every N epochs.')
    parser.add_argument(
        '--n_eval', type=float, default=0.5,
        help='Restrict size of the eval set during training. If it is less than 1, it is treated as a percentage. If it is greater than 1, it is treated like an integer.')
    parser.add_argument(
        '--st_patch_size', type=int, default=51,
        help='Size of patches that FST will break the dataset into and process separately. ')
    parser.add_argument('--preprocessed_data_path', default=None,
      help='Supply this to save/load st data from a fully qualified path.')
    parser.add_argument(
        '--terminate_if_n_nondecreasing_evals', type=int, default=10,
        help='If Eval Loss does not decrease for N consecutive eval periods, then terminate training.')
    parser.add_argument(
        '--attribute_profile', action='store_true', default=False,
        help='If true ...')

    args = parser.parse_args()

    if not os.path.isdir(args.model_root):
        os.makedirs(args.model_root)
        
    # get TF logger
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)
    
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(args.model_root, 'tensorflow.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    if args.predict:
        predict(args)
    elif args.svm_predict:
        assert args.network_spatial_size is 1, 'For SVM prediction --network_spatial_size should be 1?'
        assert args.mask_root is not None, 'For SVM prediction --mask_root must be specified.'
        svm_predict(args)
    elif args.svm_multi_mask_file_list is not None:
        assert os.path.exists(args.svm_multi_mask_file_list), 'SVM mask list txt file not found.'
        many_svm_evals(args)
    else:
        assert args.mask_root is not None, 'In training mode --mask_root must be specified.'
        train(args)

if __name__ == '__main__':
    main()
