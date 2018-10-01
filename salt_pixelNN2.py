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

import numpy as np
from PIL import Image
import tensorflow as tf

import windows as win
import salt_data as sd

# import matplotlib.pyplot as plt
import pdb

# Training Parameters
learning_rate = 0.001
batch_size = 32
num_steps = 300

# Network Parameters
num_classes = 2 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit

layerO = namedtuple('layerO', ['strides', 'padding'])

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
def pixel_net(x_dict, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, like in MNIST example
        x = x_dict['images']
        # x = tf.reshape(x, shape=[-1, 101, 101, 1])

        # (batch, h, w, chan)
        feat = scat2d_to_2d_2layer(x, reuse)
        fs = feat.get_shape()
        feat = tf.reshape(feat, (fs[0]*fs[1]*fs[2], fs[3]))

        fc1 = tf.layers.dense(feat, 256)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu)

        fc3 = tf.layers.dense(fc2, 64)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc3 = tf.layers.dropout(fc3, rate=dropout, training=is_training)

        fc4 = tf.layers.dense(fc3, 32, activation=tf.nn.relu)
        out = tf.layers.dense(fc4, 2)

    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = pixel_net(features, dropout, reuse=False,
                            is_training=True)
    logits_test = pixel_net(features, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    labels = tf.reshape(tf.cast(labels, dtype=tf.int32), [-1])
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

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

def pixel_eg():
    x = tf.placeholder(tf.float32, shape=(8,117,117,1))
    feat = pixel_net({'images': x}, dropout, reuse=False, is_training=True)

    egbatch = np.random.rand(8,117,117,1)
    # egbatch = egbatch[:8,:,:,:]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    feed_dict = {x: egbatch}
    myres = sess.run(feat, feed_dict)

def get_salt_images(folder='mytrain'):
    image_list = []
    for filename in glob.glob('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/%s/images/*.png' % folder): #assuming gif
        im=Image.open(filename).convert('L')
        npim = np.array(im, dtype=np.float32) / 255.0
        npim_padded = np.pad(npim, ((8,8),(8,8)), 'reflect')
        image_list.append(npim_padded)
        im.close()
    image_list = np.array(image_list)
    return np.expand_dims(image_list, -1)

def get_salt_labels(folder='mytrain'):
    image_list = []
    for filename in glob.glob('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/%s/masks/*.png' % folder): #assuming gif
        im=Image.open(filename).convert('L')
        npim = np.array(im, dtype=int) / 255
        image_list.append(npim.reshape(101**2))
        im.close()
    return np.array(image_list)

def main():

    trainX = get_salt_images(folder='mytrain')
    trainY = get_salt_labels(folder='mytrain')

    valX = get_salt_images(folder='myval')
    valY = get_salt_labels(folder='myval')
    # Build the Estimator
    model_dir = '/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/models/binarypix1'
    model = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    for i in range(100000):
        # Define the input function for training
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': trainX[:3584,:,:,:]}, y=trainY[:3584,:],
            batch_size=batch_size, num_epochs=1, shuffle=True)
        # Train the Model
        # tf.logging.set_verbosity(tf.logging.INFO)
        model.train(input_fn, steps=None)

        if i % 25 == 0:
            # Evaluate the Model
            # Define the input function for evaluating
            input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'images': valX[:384,:,:,:]}, y=valY[:384,:],
                batch_size=batch_size, shuffle=False)
            # Use the Estimator 'evaluate' method
            e = model.evaluate(input_fn)

            print("Testing Accuracy:", e['accuracy'])

if __name__ == '__main__':
    main()

    # lets look at the result images with the scroll thru vis
    # then do the mnist like network on binary and see results (with PCA layer in between)
    # and research what they do for semantic segmentation, u net like stuff

    # later concatenate in 2d wavelet features
