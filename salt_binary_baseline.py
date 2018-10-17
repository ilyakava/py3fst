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

import matplotlib.pyplot as plt
import pdb

# Training Parameters
learning_rate = 0.0001
batch_size = 8
num_steps = 300

# Network Parameters
num_classes = 2 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit

# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, like in MNIST example
        x = x_dict['images']

        # 1x1 conv replaces PCA step
        conv1 = tf.layers.conv2d(x, 64, 1)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu, padding='same')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2) # 52

        conv3 = tf.layers.conv2d(conv2, 128, 3, activation=tf.nn.relu, padding='same')
        conv3 = tf.layers.max_pooling2d(conv3, 2, 2) # 26

        conv4 = tf.layers.conv2d(conv3, 256, 3, activation=tf.nn.relu, padding='same')
        conv4 = tf.layers.max_pooling2d(conv4, 2, 2) # 13

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv4)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 4096)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        fc2 = tf.layers.dense(fc1, 1000)

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

    myevalops = {
    'accuracy': acc_op,
    'mask_presence/false_positives': FP_op,
    'mask_presence/false_negatives': FN_op,
    'mask_presence/true_positives': TP_op,
    'mask_presence/mask_prediction_rate': mask_predicted_rate_op,
    'mask_presence/mask_rate': tf.metrics.mean(labels),
    'mask_presence/precision': prec_op}

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
        npim_padded = np.pad(npim, ((2,1),(2,1)), 'reflect')
        image_list.append(npim_padded)
        im.close()
    image_list = np.array(image_list)
    return np.expand_dims(image_list, -1)

def save_decisions():
    """
    change the first two vars to run on different sets
    """
    valX = get_salt_images(folder='test')
    fileids = sb.clean_glob(glob.glob('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/test/images/*.png'))
    
    setsize = len(fileids)
    headsz = int(setsize / float(batch_size)) * batch_size

    input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'images': valX[:headsz,:,:,:]},
                batch_size=batch_size, shuffle=False)

    model_dir = '/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/models/binary1'
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

    np.save('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/models/binary1/test_bin_pred', id_to_pred)


def main():

    trainX = get_salt_images(folder='mytrain')
    train_pix_num = sd.salt_pixel_num(folder='mytrain')
    trainY = np.array(train_pix_num > 0).astype(int)

    valX = get_salt_images(folder='myval')
    val_pix_num = sd.salt_pixel_num(folder='myval')
    valY = np.array(val_pix_num > 0).astype(int)
    # Build the Estimator
    model_dir = '/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/models/binary_base1'
    model = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    for i in range(100000):
        # Define the input function for training
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': trainX[:3584,:,:,:]}, y=trainY[:3584],
            batch_size=batch_size, num_epochs=1, shuffle=True)
        # Train the Model
        tf.logging.set_verbosity(tf.logging.INFO)
        model.train(input_fn, steps=None)

        if i % 10 == 0:
            # Evaluate the Model
            # Define the input function for evaluating
            input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'images': valX[:384,:,:,:]}, y=valY[:384],
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
