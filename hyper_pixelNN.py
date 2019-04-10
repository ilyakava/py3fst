"""FST-CNN network
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

import numpy as np
from PIL import Image
import scipy.io as sio
import tensorflow as tf
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import windows as win
from rle import myrlestring
import salt_baseline as sb
import salt_data as sd
import fst3d_feat as fst

# import matplotlib.pyplot as plt
import pdb

DATA_PATH = '/scratch0/ilya/locDoc/data/hyperspec'
DATASET_PATH = '/scratch0/ilya/locDoc/data/hyperspec/datasets'

# Training Parameters
learning_rate = 0.0001
batch_size = 8
num_steps = 300
N_TEST = 128
N_TRAIN = 256

# Network Parameters
n_classes=0
dropout = 0.25 # Dropout, probability to drop a unit

layerO = namedtuple('layerO', ['strides', 'padding'])

def scat3d_to_3d_3x3_2layer(x, reuse=tf.AUTO_REUSE, psis=None, phi=None, layer_params=None):
    """Computes features for a specific pixel.

    Args:
        x: image in (height, width, bands) format
        psis: array of winO struct, filters are in (bands, height, width) format!
        phi: winO struct, filters are in (bands, height, width) format!
    Output:
        center pixel feature vector
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


        # swap channels with batch
        U1 = tf.transpose(U1, [4, 1, 2, 3, 0])
        # U1 is (lambda1, bands, h, w, 1)
        
        U2s = []
        # only procede with increasing frequency paths
        for res_i, used_params in enumerate(psis[0].filter_params):
            increasing_psi = win.fst3d_psi_factory(psis[1].kernel_size, used_params)
            if increasing_psi.nfilt > 0:
                U2s.append(fst.scat3d(U1[res_i:(res_i+1),:,:,:,:], increasing_psi, layer_params[1]))

        U2 = tf.concat(U2s, 4)
        # swap channels with batch
        U2 = tf.transpose(U2, [4, 1, 2, 3, 0])
        # U2 is (lambda2, bands, h, w, 1)

        # convolve with phis
        S2 = fst.scat3d(U2, phi, layer_params[2])

        def slice_idxs(sig_size, kernel_size):
            def slice_idx(s, k, f):
                if k % 2 == 0:
                    raise('not implemented even padding')
                else:
                    return int((s - k - f)//2)
            final_size = [1,3,3]
            return [slice_idx(s,k,f-1) for s,k,f in zip(sig_size, kernel_size,final_size)]

        [p1b, p1h, p1w] = slice_idxs(U1.shape[1:4], psis[1].kernel_size)
        [p2b, p2h, p2w] = slice_idxs(x.shape[1:4], psis[0].kernel_size)

        S1 = fst.scat3d(U1[:,(p1b):-(p1b),(p1h):-(p1h), (p1w):-(p1w), :], phi, layer_params[2])
        S0 = fst.scat3d(x[:, (p2b):-(p2b),(p2h):-(p2h), (p2w):-(p2w), :], phi, layer_params[2])

        # average across bands
        S0 = tf.reduce_mean(S0,1,keepdims=True)
        S1 = tf.reduce_mean(S1,1,keepdims=True)
        S2 = tf.reduce_mean(S2,1,keepdims=True)

        SX = tf.squeeze(tf.concat([S0,S1,S2], 0))
        # SX is (channels, h, w)

    return SX

# Create the neural network
def hyper_3x3_net(x_dict, dropout, reuse, is_training, n_classes):
    """
    x should be (batch, channel, h, w)
    """
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, like in MNIST example
        x = x_dict['subimages']

        # 1x1 conv replaces PCA step
        # x should be (batch, channel, h, w)
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

# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    

    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = hyper_3x3_net(features, dropout, reuse=False,
                            is_training=True, n_classes=n_classes)
    logits_test = hyper_3x3_net(features, dropout, reuse=True,
                            is_training=False, n_classes=n_classes)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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


def get_train_test_data():
    global n_classes
    reuse = tf.AUTO_REUSE

    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Pavia_center_right'))
    data = mat_contents['Pavia_center_right'].astype(np.float32)
    data /= np.max(np.abs(data))
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Pavia_center_right_gt.mat'))
    labels = mat_contents['Pavia_center_right_gt']
    traintestfilename = 'Pavia_center_right_gt_traintest_coarse_128px128p.mat'

    netO = fst.Pavia_net()
    # from pavia net
    psi = win.fst3d_psi_factory([7,7,7])
    phi = win.fst3d_phi_window_3D([7,7,7])
    layer_params = layerO((3,1,1), 'valid')
    psis=[psi,psi]
    layer_params=[layer_params, layer_params, layer_params]
    # addl_padding = (18,18,18)
    addl_padding = (20,20,18)
    

    [height, width, nbands] = data.shape

    all_pixels = np.array(list(itertools.product(range(width),range(height))))
    flat_labels = labels.transpose().reshape(height*width)
    n_classes = len(set(flat_labels.tolist())) - 1

    ap = np.array(addl_padding)
    assert np.all(ap % 2 == 0), 'Assymetric is not supported'
    
    padded_data = np.pad(data, ((ap[0]//2,ap[0]//2),(ap[1]//2,ap[1]//2),(ap[2]//2,ap[2]//2)), 'wrap')
    

    mat_contents = None
    try:
        mat_contents = sio.loadmat(os.path.join(DATA_PATH, traintestfilename))
    except:
        mat_contents = hdf5storage.loadmat(os.path.join(DATA_PATH, traintestfilename))
    train_mask = mat_contents['train_mask'].astype(int).squeeze()
    test_mask = mat_contents['test_mask'].astype(int).squeeze()
    
    train_pixels = np.array(filter(lambda (x,y): labels[y,x]*train_mask[x*height+y] != 0, all_pixels))
    test_pixels = np.array(filter(lambda (x,y): labels[y,x]*test_mask[x*height+y] != 0, all_pixels))
    
    train_pixels_list = train_pixels.tolist()
    random.shuffle(train_pixels_list)
    train_pixels = np.array(train_pixels_list)
    

    test_pixels_list = test_pixels.tolist()
    random.shuffle(test_pixels_list)
    test_pixels = np.array(test_pixels_list)

    print("Train / Test split is %i / %i" % (train_pixels.shape[0], test_pixels.shape[0]))

    batch_item_shape = tuple(map(operator.add, netO.addl_padding, (3,3,data.shape[2])))

    x = tf.placeholder(tf.float32, shape=batch_item_shape)
    feat = scat3d_to_3d_3x3_2layer(x, reuse, psis, phi, layer_params)
    feat_shape = tuple([int(d) for d in feat.shape])

    trainX = np.zeros((train_mask.sum(),) + feat_shape)
    trainY = np.zeros((train_mask.sum(),))
    valX = np.zeros((test_mask.sum(),) + feat_shape)
    valY = np.zeros((test_mask.sum(),))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for pixel_i, pixel in enumerate(tqdm(train_pixels[:N_TRAIN,:], desc='Loading train data: ')):
            # this iterates through columns first
            [pixel_x, pixel_y] = pixel
            subimg = padded_data[pixel_y:(pixel_y+ap[0]+1), pixel_x:(pixel_x+ap[1]+1), :]
            feed_dict = {x: subimg}
            trainX[pixel_i,:,:,:] = sess.run(feat, feed_dict)
            trainY[pixel_i] = labels[pixel_y,pixel_x] - 1

        for pixel_i, pixel in enumerate(tqdm(test_pixels[:N_TEST,:], desc='Loading test data: ')):
            # this iterates through columns first
            [pixel_x, pixel_y] = pixel
            subimg = padded_data[pixel_y:(pixel_y+ap[0]+1), pixel_x:(pixel_x+ap[1]+1), :]
            feed_dict = {x: subimg}
            valX[pixel_i,:,:,:] = sess.run(feat, feed_dict)
            valY[pixel_i] = labels[pixel_y,pixel_x] - 1

    return trainX, trainY, valX, valY

def main():
    trainX, trainY, valX, valY = get_train_test_data()
    
    model_dir = '/scratch0/ilya/locDoc/data/hypernet/models/threexthree'
    model = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    for i in range(100000):
        # Define the input function for training
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'subimages': trainX[:N_TRAIN,:,:,:]}, y=trainY[:N_TRAIN],
            batch_size=batch_size, num_epochs=1, shuffle=True)
        # Train the Model
        tf.logging.set_verbosity(tf.logging.INFO)
        model.train(input_fn, steps=(1 * N_TRAIN // batch_size)) # 

        # Evaluate the Model
        # Define the input function for evaluating
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'subimages': valX[:N_TEST,:,:,:]}, y=valY[:N_TEST],
            batch_size=batch_size, shuffle=False)
        # Use the Estimator 'evaluate' method
        e = model.evaluate(input_fn)

        print("Testing Accuracy:", e['accuracy'])


if __name__ == '__main__':
    main()
