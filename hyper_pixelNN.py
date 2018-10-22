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

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import windows as win
from rle import myrlestring
import salt_baseline as sb
import salt_data as sd
import fst3d_feat as fst

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

# Create the neural network
def hyper_pixel_net(x_dict, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, like in MNIST example
        x = x_dict['images']
        # x = tf.reshape(x, shape=[-1, 101, 101, 1])

        # (batch, h, w, chan)
        feat = fst.hyper3d_net(x, reuse)
        pdb.set_trace()
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
    logits_train = hyper_pixel_net(features, dropout, reuse=False,
                            is_training=True)
    logits_test = hyper_pixel_net(features, dropout, reuse=True,
                           is_training=False)

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

def format_dataset(data, labels, traintestfilenames):
    [height, width, nbands] = data.shape


    all_pixels = np.array(list(itertools.product(range(width),range(height))))
    labelled_pixels = np.array(filter(lambda (x,y): labels[y,x] != 0, all_pixels))
    flat_labels = labels.transpose().reshape(height*width)
    nlabels = len(set(flat_labels.tolist())) - 1

    ap = np.array(netO.addl_padding)
    assert np.all(ap % 2 == 0), 'Assymetric is not supported'
    net_in_shape = ap + np.array([1,1,nbands])
    x = tf.placeholder(tf.float32, shape=net_in_shape)
    feat = netO.model_fn(x)

    padded_data = np.pad(data, ((ap[0]/2,ap[0]/2),(ap[1]/2,ap[1]/2),(ap[2]/2,ap[2]/2)), 'wrap')

    print('requesting %d MB memory' % (labelled_pixels.shape[0] * feat.shape[0] * 4 / 1000000.0))
    labelled_pix_feat = np.zeros((labelled_pixels.shape[0], feat.shape[0]), dtype=np.float32)
    
    def extract_subimgs():
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for pixel_i, pixel in enumerate(tqdm(labelled_pixels)):
            # this iterates through columns first
            [pixel_x, pixel_y] = pixel
            subimg = padded_data[pixel_y:(pixel_y+ap[0]+1), pixel_x:(pixel_x+ap[1]+1), :]
        
            feed_dict = {x: subimg}
            labelled_pix_feat[pixel_i,:] = sess.run(feat, feed_dict)
    compute_features()

    for traintestfilename in traintestfilenames:
        mat_contents = sio.loadmat(os.path.join(DATA_PATH, traintestfilename))
        train_mask = mat_contents['train_mask'].astype(int).squeeze()
        test_mask = mat_contents['test_mask'].astype(int).squeeze()
        # resize train/test masks to labelled pixels
        train_mask_skip_unlabelled = train_mask[flat_labels!=0]
        test_mask_skip_unlabelled = test_mask[flat_labels!=0]

        # get training set
        trainY = flat_labels[train_mask==1]
        trainX = labelled_pix_feat[train_mask_skip_unlabelled==1,:]

        print('training now')
        start = time.time()
        prob  = svm_problem(trainY.tolist(), trainX.tolist())
        param = svm_parameter('-s 0 -t 0 -q')
        m = svm_train(prob, param)
        end = time.time()
        print(end - start)

        if outfilename:
            nextoutfilename = os.path.join(DATA_PATH, outfilename)
        else:
            nextoutfilename = os.path.join(DATA_PATH, traintestfilename+'_pyFST3D_expt.mat')

        # now test
        test_chunk_size = 1000
        testY = flat_labels[test_mask==1]
        testX = labelled_pix_feat[test_mask_skip_unlabelled==1,:]

def get_labels(folder='mytrain'):
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

            # print("Testing Accuracy:", e['accuracy'])


if __name__ == '__main__':
    kaggle_test()

    # lets look at the result images with the scroll thru vis
    # then do the mnist like network on binary and see results (with PCA layer in between)
    # and research what they do for semantic segmentation, u net like stuff

    # later concatenate in 2d wavelet features
