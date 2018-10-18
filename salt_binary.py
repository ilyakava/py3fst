""" Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

This example is using TensorFlow layers API, see 'convolutional_network_raw' 
example for a raw implementation with variables.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

from collections import namedtuple
import itertools
import glob
import logging
import os

import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

import windows as win
import salt_data as sd
import salt_pixelNN as sNN
import salt_baseline as sb
import tang_feat as otf
from kaggle_prec import kaggle_prec

import matplotlib.pyplot as plt
import pdb

# Training Parameters
learning_rate = 0.00001
batch_size = 8
num_steps = 300

# Network Parameters
num_classes = 2 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit

layerO = namedtuple('layerO', ['strides', 'padding'])
winO = namedtuple('winO', ['nfilt', 'filters', 'filter_params', 'kernel_size'])

def scat2d(x, win_params, layer_params):
    """Single layer of 2d scattering
    Args:
        x is input with dim (batch, height, width, 1)
        win_params.filters is complex with dim ......... (depth, height, width, channels)
    """
    real1 = tf.layers.conv2d(
        x,
        win_params.nfilt,
        win_params.kernel_size,
        strides=layer_params.strides,
        padding=layer_params.padding,
        dilation_rate=(1,1),
        activation=None,
        use_bias=False,
        kernel_initializer=tf.constant_initializer(win_params.filters.real, dtype=tf.float32),
        trainable=False,
        name=None
    )

    imag1 = tf.layers.conv2d(
        x,
        win_params.nfilt,
        win_params.kernel_size,
        strides=layer_params.strides,
        padding=layer_params.padding,
        dilation_rate=(1,1),
        activation=None,
        use_bias=False,
        kernel_initializer=tf.constant_initializer(win_params.filters.imag, dtype=tf.float32),
        trainable=False,
        name=None
    )

    return tf.abs(tf.complex(real1, imag1))

def wst2d_to_2d_2layer(x, reuse=tf.AUTO_REUSE):
    bs = batch_size
    kernel_size = [1,7,7]
    max_scale = 3
    K = 3

    psis = [None,None]
    layer_params = [None,None,None]

    psiO = win.tang_psi_factory(max_scale, K, kernel_size)
    psiO = winO(psiO.nfilt, psiO.filters.squeeze(), psiO.filter_params, kernel_size[1:])
    phiO = win.tang_phi_window_3D(max_scale, kernel_size)
    phiO = winO(phiO.nfilt, phiO.filters.squeeze(), phiO.filter_params, kernel_size[1:])

    with tf.variable_scope('TangNet2d', reuse=reuse):
        psis[0] = psiO
        layer_params[0] = layerO((1,1), 'valid')

        # 107, 107
        U1 = scat2d(x, psis[0], layer_params[0])

        layer_params[1] = layerO((1,1), 'valid')
        U2s = []
        # only procede with increasing frequency paths
        for res_i, used_params in enumerate(psis[0].filter_params):
            if used_params[0] < max_scale - 1:
                psiO_inc = win.tang_psi_factory(max_scale, K, kernel_size, used_params[0]+1)
                psiO_inc = winO(psiO_inc.nfilt, psiO_inc.filters.squeeze(), psiO_inc.filter_params, kernel_size[1:])
                U2s.append(scat2d(U1[:,:,:,res_i:(res_i+1)], psiO_inc, layer_params[1]))
        

        # 101, 101
        U2 = tf.concat(U2s, 3)
        # swap to (batch, chanU2, h, w)
        U2 = tf.transpose(U2, [0,3,1,2])
        # reshape to (batch, h,w, 1)
        U2os = U2.get_shape()
        U2 = tf.reshape(U2, (bs*U2.get_shape()[1], U2.get_shape()[2],U2.get_shape()[3],1))

        # swap to (batch, chanU1, h, w)
        U1 = tf.transpose(U1, [0,3,1,2])
        # reshape to (batch, h,w, 1)
        U1os = U1.get_shape()
        U1 = tf.reshape(U1, (bs*U1.get_shape()[1], U1.get_shape()[2], U1.get_shape()[3], 1))

        # now lo-pass

        # each layer lo-passed differently so that (h,w) align bc we
        # want to be able to do 2d convolutions afterwards again
        layer_params[2] = layerO((1,1), 'valid')
        phi = win.fst2d_phi_factory([5,5])

        # filter and separate by original batch via old shape
        S0 = scat2d(x[:,6:-6, 6:-6, :], phi, layer_params[2])
        S0 = tf.reshape(S0, (bs, 1, S0.get_shape()[1], S0.get_shape()[2]))
        S1 = scat2d(U1[:,3:-3,3:-3,:], phi, layer_params[2])
        S1 = tf.reshape(S1, (bs, U1os[1], S1.get_shape()[1],S1.get_shape()[2]))
        S2 = scat2d(U2, phi, layer_params[2])
        S2 = tf.reshape(S2, (bs, U2os[1], S2.get_shape()[1], S2.get_shape()[2]))

        # (batch, chan, h,w)
        feat2d = tf.concat([S0,S1,S2], 1)

    return tf.transpose(feat2d, [0,2,3,1])

def scat2d_to_2d_2layer(x, reuse=tf.AUTO_REUSE):
    """
    Args:
        x: in (batch, h, w, 1) shape
    Returns
        (batch, h, w, channels)
    """
    psis = [None,None]
    layer_params = [None,None,None]
    with tf.variable_scope('scat2d_to_2d_2layer', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs

        # x = tf.reshape(x, shape=[-1, 113, 113, 1])
        bs = batch_size

        psis[0] = win.fst2d_psi_factory([7, 7], include_avg=False)
        layer_params[0] = layerO((1,1), 'valid')

        # 107, 107
        U1 = scat2d(x, psis[0], layer_params[0])

        psis[1] = win.fst2d_psi_factory([7, 7], include_avg=False)
        layer_params[1] = layerO((1,1), 'valid')

        U2s = []
        # only procede with increasing frequency paths
        for res_i, used_params in enumerate(psis[0].filter_params):
            increasing_psi = win.fst2d_psi_factory(psis[1].kernel_size, used_params)
            if increasing_psi.nfilt > 0:
                U2s.append(scat2d(U1[:,:,:,res_i:(res_i+1)], increasing_psi, layer_params[1]))
        

        # 101, 101
        U2 = tf.concat(U2s, 3)
        # swap to (batch, chanU2, h, w)
        U2 = tf.transpose(U2, [0,3,1,2])
        # reshape to (batch, h,w, 1)
        U2os = U2.get_shape()
        U2 = tf.reshape(U2, (bs*U2.get_shape()[1], U2.get_shape()[2],U2.get_shape()[3],1))

        # swap to (batch, chanU1, h, w)
        U1 = tf.transpose(U1, [0,3,1,2])
        # reshape to (batch, h,w, 1)
        U1os = U1.get_shape()
        U1 = tf.reshape(U1, (bs*U1.get_shape()[1], U1.get_shape()[2], U1.get_shape()[3], 1))

        # now lo-pass

        # each layer lo-passed differently so that (h,w) align bc we
        # want to be able to do 2d convolutions afterwards again
        layer_params[2] = layerO((1,1), 'valid')
        phi = win.fst2d_phi_factory([5,5])

        # filter and separate by original batch via old shape
        S0 = scat2d(x[:,6:-6, 6:-6, :], phi, layer_params[2])
        S0 = tf.reshape(S0, (bs, 1, S0.get_shape()[1], S0.get_shape()[2]))
        S1 = scat2d(U1[:,3:-3,3:-3,:], phi, layer_params[2])
        S1 = tf.reshape(S1, (bs, U1os[1], S1.get_shape()[1],S1.get_shape()[2]))
        S2 = scat2d(U2, phi, layer_params[2])
        S2 = tf.reshape(S2, (bs, U2os[1], S2.get_shape()[1], S2.get_shape()[2]))

        # (batch, chan, h,w)
        feat2d = tf.concat([S0,S1,S2], 1)

    return tf.transpose(feat2d, [0,2,3,1])

# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, like in MNIST example
        x = x_dict['images']

        feat = wst2d_to_2d_2layer(x, reuse)

        mlt = 1
        # 1x1 conv replaces PCA step
        conv1 = tf.layers.conv2d(feat, 16*mlt, 1)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 16*mlt, 3, activation=tf.nn.relu, padding='same')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2) # 52

        conv3 = tf.layers.conv2d(conv2, 32*mlt, 3, activation=tf.nn.relu, padding='same')
        conv3 = tf.layers.max_pooling2d(conv3, 2, 2) # 26

        conv4 = tf.layers.conv2d(conv3, 128*mlt, 3, activation=tf.nn.relu, padding='same')
        conv4 = tf.layers.max_pooling2d(conv4, 2, 2) # 13

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv4)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024*mlt)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        fc2 = tf.layers.dense(fc1, 250*mlt)

        # Output layer, class prediction
        out = tf.layers.dense(fc2, n_classes)

    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

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

    FP_op = tf.metrics.false_positives(labels, pred_classes)
    FN_op = tf.metrics.false_negatives(labels, pred_classes)
    TP_op = tf.metrics.true_positives(labels, pred_classes)
    mask_predicted_rate_op = tf.metrics.mean(pred_classes)
    prec_op = tf.metrics.precision(labels, pred_classes)

    kaggle_op = kaggle_prec(labels, pred_classes)

    myevalops = {
    'accuracy': acc_op,
    'mask_presence/false_positives': FP_op,
    'mask_presence/false_negatives': FN_op,
    'mask_presence/true_positives': TP_op,
    'mask_presence/mask_prediction_rate': mask_predicted_rate_op,
    'mask_presence/mask_rate': tf.metrics.mean(labels),
    'mask_presence/precision': prec_op,
    'mask_presence/kaggle': kaggle_op}

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops=myevalops)

    return estim_specs

from window_plot import ScrollThruPlot

def scat2d_eg():
    x = tf.placeholder(tf.float32, shape=(8,113,113,1))
    feat = scat2d_to_2d_2layer(x)

    egbatch = get_salt_images()
    egbatch = egbatch[:8,:,:,:]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    feed_dict = {x: egbatch}
    myres = sess.run(feat, feed_dict)
    files = glob.glob('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/train/images/*.png')
    # now lets look at them
    X = myres[0,:,:,:]
    fig, ax = plt.subplots(1, 1)
    tracker = ScrollThruPlot(ax, X, fig)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
    pdb.set_trace()

def get_salt_images(folder='mytrain'):
    image_list = []
    for filename in glob.glob('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/%s/images/*.png' % folder): #assuming gif
        im=Image.open(filename).convert('L')
        npim = np.array(im, dtype=np.float32) / 255.0
        npim_padded = np.pad(npim, ((10,9),(10,9)), 'reflect')
        image_list.append(npim_padded)
        im.close()
    image_list = np.array(image_list)
    return np.expand_dims(image_list, -1)

def get_decisions(foldername='test', model_dir=None, outfile=None):
    """
    change the first two vars to run on different sets
    """
    valX = get_salt_images(folder=foldername)
    fileids = sb.clean_glob(glob.glob('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/%s/images/*.png' % foldername))
    
    setsize = len(fileids)
    headsz = int(setsize / float(batch_size)) * batch_size

    input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'images': valX[:headsz,:,:,:]},
                batch_size=batch_size, shuffle=False)

     # '/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/models/binary1'
    bin_model = tf.estimator.Estimator(model_fn, model_dir=model_dir)
    gen = bin_model.predict(input_fn)

    id_to_pred = {}

    for file_i, prediction in enumerate(tqdm(gen, total=headsz)):
        fileid, file_extension = os.path.splitext(fileids[file_i])
        id_to_pred[fileid] = prediction

    # now get the tail
    input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': valX[-batch_size:,:,:,:]},
        batch_size=batch_size, shuffle=False)
    gen = bin_model.predict(input_fn)
    for file_i, prediction in enumerate(gen):
        idx = setsize-batch_size+file_i
        fileid, file_extension = os.path.splitext(fileids[idx])
        
        id_to_pred[fileid] = prediction

    # '/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/models/binary1/test_bin_pred'
    if outfile:
        np.save(outfile, id_to_pred)

    return id_to_pred

def view_mistakes():
    foldername = 'myval'
    id_to_pred = get_decisions(foldername, '/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/models/binary7')
    fileids = sb.clean_glob(glob.glob('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/%s/images/*.png' % foldername))

    imgs = sd.get_salt_labels(folder='myval')
    imgs = np.reshape(imgs, (imgs.shape[0], 101,101))
    # imgs = sd.get_salt_images(folder='myval')
    val_pix_num = sd.salt_pixel_num(folder='myval')
    valY = np.array(val_pix_num > 0).astype(int)

    mistake_i = 0
    for file_id_i, file_id in enumerate(fileids):
        fileid, file_extension = os.path.splitext(file_id)
        if valY[file_id_i] != id_to_pred[fileid]:
            imgs[mistake_i,:,:] = imgs[file_id_i,:,:]
            mistake_i += 1
    
    print('made %i mistakes' % mistake_i)
    X = np.swapaxes(np.swapaxes(imgs[:mistake_i,:,:],0,2),0,1).astype(float)
    fig, ax = plt.subplots(1, 1)
    tracker = ScrollThruPlot(ax, X, fig)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


def main():

    trainX = get_salt_images(folder='mytrain')
    train_pix_num = sd.salt_pixel_num(folder='mytrain')
    trainY = np.array(train_pix_num > 0).astype(int)

    valX = get_salt_images(folder='myval')
    val_pix_num = sd.salt_pixel_num(folder='myval')
    valY = np.array(val_pix_num > 0).astype(int)
    # Build the Estimator
    model_dir = '/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/models/binary4b'
    model = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    for i in range(100000):
        # Define the input function for training
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': trainX[:3584,:,:,:]}, y=trainY[:3584],
            batch_size=batch_size, num_epochs=1, shuffle=True)
        # Train the Model
        tf.logging.set_verbosity(tf.logging.INFO)
        model.train(input_fn, steps=None)

        if i % 5 == 0:
            # Evaluate the Model
            # Define the input function for evaluating
            input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'images': valX[:384,:,:,:]}, y=valY[:384],
                batch_size=batch_size, shuffle=False)
            # Use the Estimator 'evaluate' method
            e = model.evaluate(input_fn)

            print("Testing Accuracy:", e['accuracy'])

if __name__ == '__main__':
    view_mistakes()

    # lets look at the result images with the scroll thru vis
    # then do the mnist like network on binary and see results (with PCA layer in between)
    # and research what they do for semantic segmentation, u net like stuff

    # later concatenate in 2d wavelet features
