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
batch_size = 32
num_steps = 300

# Network Parameters
num_classes = 2 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit

layerO = namedtuple('layerO', ['strides', 'padding'])

def scat3d_to_3d_3x3_2layer(x, reuse=tf.AUTO_REUSE, psis=None, phi=None, layer_params=None):
    """Computes features for a specific pixel.

    Args:
        x: image in (1, height, width, bands) format
        psis: array of winO struct, filters are in (bands, height, width) format!
        phi: winO struct, filters are in (bands, height, width) format!
    Output:
        center pixel feature vector
    """
    assert len(layer_params) == 3, 'this network is 2 layers only'
    assert len(psis) == 2, 'this network is 2 layers only'

    
    with tf.variable_scope('Hyper3DNet', reuse=reuse):
        x = tf.transpose(x, [0, 3, 1, 2])

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
                pdb.set_trace()
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

        S0 = tf.reduce_mean(S0,1,keepdims=True)
        S1 = tf.reduce_mean(S1,1,keepdims=True)
        S2 = tf.reduce_mean(S2,1,keepdims=True)

        SX = tf.squeeze(tf.concat([S0,S1,S2], 0))
        # SX is (channels, h, w)

    return SX

# Create the neural network
def hyper_3x3_net(x_dict, dropout, reuse, is_training, psis, phi, layer_params):
    """
    x should be (batch, channel, h, w)
    """
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, like in MNIST example
        x = x_dict['subimages']

        SX = scat3d_to_3d_3x3_2layer(x, reuse, psis, phi, layer_params)

        SX_batch = tf.expand_dims(SX,0)


        # 1x1 conv replaces PCA step
        # x should be (batch, channel, h, w)
        conv1 = tf.layers.conv2d(SX_batch, 1024, 1, data_format='channels_first')
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
    # from pavia net
    psi = win.fst3d_psi_factory([7,7,7])
    phi = win.fst3d_phi_window_3D([7,7,7])
    layer_params = layerO((3,1,1), 'valid')
    psis=[psi,psi]
    layer_params=[layer_params, layer_params, layer_params]

    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = hyper_3x3_net(features, dropout, reuse=False,
                            is_training=True, psis=psis, phi=phi, layer_params=layer_params)
    logits_test = hyper_3x3_net(features, dropout, reuse=True,
                            is_training=False, psis=psis, phi=phi, layer_params=layer_params)

    pdb.set_trace()
    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    notflat_pred_classes = tf.reshape(pred_classes, [batch_size, 101**2])

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'mask': notflat_pred_classes
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Define loss and optimizer
    flat_labels = tf.reshape(tf.cast(labels, dtype=tf.int32), [-1])
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=flat_labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=flat_labels, predictions=pred_classes)
    iou_op = tf.metrics.mean_iou(labels=flat_labels, predictions=pred_classes, num_classes=2)

    # kaggle spec
    threshes = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    labels = tf.cast(labels, dtype=tf.float32)
    notflat_pred_classes = tf.cast(notflat_pred_classes, dtype=tf.float32)
    intx = tf.reduce_sum(tf.multiply(labels, notflat_pred_classes), axis=1)
    union = tf.reduce_sum(tf.add(labels, notflat_pred_classes), axis=1)
    iou = tf.divide(intx, tf.add(union,1))
    mask_present_gt = tf.minimum(tf.reduce_sum(labels, axis=1),1)

    TP, TP_op = tf.metrics.true_positives_at_thresholds(mask_present_gt, iou, threshes) #  IoU above the threshold.
    FP, FP_op = tf.metrics.false_positives_at_thresholds(mask_present_gt, iou, threshes) # predicted something, no gt.
    FN, FN_op = tf.metrics.false_negatives_at_thresholds(mask_present_gt, iou, threshes) # gt but no prediction
    prec, prec_op = tf.metrics.precision_at_thresholds(mask_present_gt, iou, threshes)

    summarize_metrics(TP_op, 'TP', threshes)
    summarize_metrics(FP_op, 'FP', threshes)
    summarize_metrics(FN_op, 'FN', threshes)
    summarize_metrics(prec_op, 'prec', threshes)

    myevalops = {'2pxwise_accuracy': acc_op,
    '2pxwise_iou': iou_op}

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops=myevalops)

    return estim_specs


def get_train_test_data():
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Pavia_center_right'))
    data = mat_contents['Pavia_center_right'].astype(np.float32)
    data /= np.max(np.abs(data))
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Pavia_center_right_gt.mat'))
    labels = mat_contents['Pavia_center_right_gt']
    traintestfilename = 'Pavia_center_right_gt_traintest_coarse_128px128p.mat'
    netO = fst.Pavia_net()

    

    [height, width, nbands] = data.shape

    all_pixels = np.array(list(itertools.product(range(width),range(height))))
    flat_labels = labels.transpose().reshape(height*width)
    nlabels = len(set(flat_labels.tolist())) - 1

    ap = np.array(netO.addl_padding)
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

    batch_item_shape = tuple(map(operator.add, netO.addl_padding, (1,1,data.shape[2])))
    trainX = np.zeros((train_mask.sum(),) + batch_item_shape)
    for pixel_i, pixel in enumerate(tqdm(train_pixels[:8,:], desc='Loading train data: ')):
        # this iterates through columns first
        [pixel_x, pixel_y] = pixel
        trainX[pixel_i,:,:,:] = padded_data[pixel_y:(pixel_y+ap[0]+1), pixel_x:(pixel_x+ap[1]+1), :]

    batch_item_shape = tuple(map(operator.add, netO.addl_padding, (1,1,data.shape[2])))
    valX = np.zeros((test_mask.sum(),) + batch_item_shape)
    for pixel_i, pixel in enumerate(tqdm(test_pixels[:8,:], desc='Loading test data: ')):
        # this iterates through columns first
        [pixel_x, pixel_y] = pixel
        valX[pixel_i,:,:,:] = padded_data[pixel_y:(pixel_y+ap[0]+1), pixel_x:(pixel_x+ap[1]+1), :]

    trainY = flat_labels[train_mask==1]
    valY = flat_labels[test_mask==1]

    return trainX, trainY, valX, valY

def main():
    trainX, trainY, valX, valY = get_train_test_data()
    
    model_dir = '/scratch0/ilya/locDoc/data/hypernet/models/threexthree'
    model = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    for i in range(100000):
        # Define the input function for training
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'subimages': trainX[:,:,:,:]}, y=trainY[:],
            batch_size=1, num_epochs=1, shuffle=True)
        # Train the Model
        # tf.logging.set_verbosity(tf.logging.INFO)
        model.train(input_fn, steps=None)

        if i % 25 == 0:
            # Evaluate the Model
            # Define the input function for evaluating
            input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'subimages': valX[:,:,:,:]}, y=valY[:],
                batch_size=batch_size, shuffle=False)
            # Use the Estimator 'evaluate' method
            e = model.evaluate(input_fn)

            # print("Testing Accuracy:", e['accuracy'])


if __name__ == '__main__':
    main()

    # lets look at the result images with the scroll thru vis
    # then do the mnist like network on binary and see results (with PCA layer in between)
    # and research what they do for semantic segmentation, u net like stuff

    # later concatenate in 2d wavelet features
