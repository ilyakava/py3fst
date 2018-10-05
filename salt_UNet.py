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
import sys

import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import windows as win
from rle import myrlestring
import salt_baseline as sb
import salt_data as sd
import salt_pixelNN as pxNN

sys.path.append('lib')
import lib.tf_unet.unet as unet
import lib.tf_unet.layers as unetlay

# import matplotlib.pyplot as plt
import pdb

# Training Parameters
learning_rate = 0.0001
batch_size = 32
num_steps = 300

# Network Parameters
num_classes = 2 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit

layerO = namedtuple('layerO', ['strides', 'padding'])

def scatU(x_dict, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    keep_prob = dropout
    if not is_training:
        keep_prob = 1
    with tf.variable_scope('scatU', reuse=reuse):
        # TF Estimator input is a dict, like in MNIST example
        x = x_dict['images']
        # x = tf.reshape(x, shape=[-1, 101, 101, 1])

        # (batch, h, w, chan)
        feat = pxNN.scat2d_to_2d_2layer(x, reuse, batch_size)
        # 1x1 conv replaces PCA step
        pcadim = 64
        pcaed = tf.layers.conv2d(feat, pcadim, 1)

        logits, variables, offset = unet.create_conv_net(
            pcaed, keep_prob, pcadim, 2,
            layers=3, features_root=16, filter_size=3, pool_size=2, summaries=True)

    return logits

def _get_cost(logits, y, cost_name="cross_entropy", regularizer=None):
    """
    Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
    Optional arguments are:
    class_weights: weights for the different classes in case of multi-class imbalance
    regularizer: power of the L2 regularizers added to the loss function
    """
    n_class = 2
    y = tf.stack([tf.cast(tf.equal(y, 0), dtype=tf.int32), tf.cast(tf.equal(y, 1), dtype=tf.int32)], -1)
    with tf.name_scope("cost"):
        flat_logits = tf.reshape(logits, [-1, n_class])
        flat_labels = tf.reshape(y, [-1, n_class])
        if cost_name == "cross_entropy":
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                                 labels=flat_labels))
        elif cost_name == "dice_coefficient":
            eps = 1e-5
            prediction = unetlay.pixel_wise_softmax(logits)
            intersection = tf.reduce_sum(prediction * y)
            union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(y)
            loss = -(2 * intersection / (union))

        else:
            raise ValueError("Unknown cost function: " % cost_name)

        if regularizer is not None:
            regularizers = sum([tf.nn.l2_loss(variable) for variable in variables])
            loss += (regularizer * regularizers)

        return loss

# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = scatU(features, dropout, reuse=False,
                            is_training=True)
    logits_test = scatU(features, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_probas = unetlay.pixel_wise_softmax(logits_test)
    pred_classes = tf.cast(tf.argmax(logits_test, axis=3), tf.int32)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'mask': pred_classes
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Define loss and optimizer
    labels = tf.cast(labels, dtype=tf.int32)
    square_labels = tf.reshape(labels, [-1, 104,104])
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=square_labels, logits=logits_train))
    # loss_op = _get_cost(logits_train, square_labels)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=square_labels, predictions=pred_classes)
    iou_op = tf.metrics.mean_iou(labels=square_labels, predictions=pred_classes, num_classes=2)

    mask_present_gt = tf.minimum(tf.reduce_sum(labels, axis=1),1)
    mask_pixel_count = tf.reduce_sum(pred_classes, axis=(1,2))
    mask_present_pred = tf.minimum(mask_pixel_count,1)
    FP_op = tf.metrics.false_positives(mask_present_gt, mask_present_pred)
    FN_op = tf.metrics.false_negatives(mask_present_gt, mask_present_pred)
    mask_predicted_rate_op = tf.metrics.mean(mask_present_pred)
    mask_volume_op = tf.metrics.mean(mask_pixel_count)

    myevalops = {
    'accuracy': acc_op,
    'iou': iou_op,
    'mask_presence/false_positives': FP_op,
    'mask_presence/false_negatives': FN_op,
    'mask_presence/mask_prediction_rate': mask_predicted_rate_op,
    'mask_presence/avg_mask_size': mask_volume_op}

    # kaggle spec
    threshes = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    intx = tf.reduce_sum(tf.multiply(square_labels, pred_classes), axis=(1,2))
    union = tf.reduce_sum(tf.add(square_labels, pred_classes), axis=(1,2))
    iou = tf.divide(intx, tf.add(union,1))
    for thresh in threshes:
        hit_at_thresh = tf.cast(tf.greater(iou, thresh), tf.int32)
        myevalops['true_positive/%.2f' % thresh] = tf.metrics.true_positives(mask_present_gt, hit_at_thresh)
        myevalops['precision/%.2f' % thresh] = tf.metrics.precision(mask_present_gt, hit_at_thresh)

    with tf.name_scope("summaries"):
        for k in range(4):
            tf.summary.image('mask/train_true_%02d' % k, 
                unet.get_image_summary(tf.expand_dims(tf.cast(square_labels[k:k+1,:,:], dtype=tf.float32), -1)))
            tf.summary.image('mask/train_pred_%02d' % k, 
                unet.get_image_summary(tf.expand_dims(tf.cast(pred_classes[k:k+1,:,:], dtype=tf.float32), -1)))
            tf.summary.image('mask/train_prob_0_%02d' % k, 
                unet.get_image_summary(tf.cast(pred_probas[k:k+1,:,:,:1], dtype=tf.float32)))
            tf.summary.image('mask/train_prob_1_%02d' % k, 
                unet.get_image_summary(tf.cast(pred_probas[k:k+1,:,:,1:], dtype=tf.float32)))


    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops=myevalops)

    return estim_specs

def get_salt_images(folder='mytrain'):
    image_list = []
    for filename in glob.glob('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/%s/images/*.png' % folder): #assuming gif
        im=Image.open(filename).convert('L')
        npim = np.array(im, dtype=np.float32) / 255.0
        npim_padded = np.pad(npim, ((30,29),(30,29)), 'reflect')
        image_list.append(npim_padded)
        im.close()
    image_list = np.array(image_list)
    return np.expand_dims(image_list, -1)

def get_salt_labels(folder='mytrain'):
    image_list = []
    for filename in glob.glob('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/%s/masks/*.png' % folder): #assuming gif
        im=Image.open(filename).convert('L')
        npim = np.array(im, dtype=int) / 255
        npim_padded = np.pad(npim, ((2,1),(2,1)), 'reflect')
        image_list.append(npim_padded.flatten())
        im.close()
    return np.array(image_list)

def main():

    trainX = get_salt_images(folder='mytrain')
    trainY = get_salt_labels(folder='mytrain')

    valX = get_salt_images(folder='myval')
    valY = get_salt_labels(folder='myval')
    
    trainX = trainX[trainY.sum(axis=1) > 0]
    trainY = trainY[trainY.sum(axis=1) > 0]
    ts = batch_size * (trainY.shape[0] // batch_size)

    valX = valX[valY.sum(axis=1) > 0]
    valY = valY[valY.sum(axis=1) > 0]
    vs = batch_size * (valY.shape[0] // batch_size)

    model_dir = '/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/models/unet2k'
    model = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    for i in range(100000):
        # Define the input function for training
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': trainX[:ts,:,:,:]}, y=trainY[:ts,:],
            batch_size=batch_size, num_epochs=1, shuffle=True)
        # Train the Model
        # tf.logging.set_verbosity(tf.logging.INFO)
        model.train(input_fn, steps=None)

        if i % 25 == 0:
            # Evaluate the Model
            # Define the input function for evaluating
            input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'images': valX[:vs,:,:,:]}, y=valY[:vs,:],
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
