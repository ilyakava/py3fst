from collections import namedtuple
import glob
import logging
import os
import random


import argparse
import numpy as np
import tensorflow as tf

from audio_load import load_audio_from_files, audio2spec
from rgb_pixelNN import scat2d
import windows as win

import pdb

layerO = namedtuple('layerO', ['strides', 'padding'])

def wst_net_v1(x_dict, dropout, reuse, is_training, n_classes):
    """Network to follow ST preprocessing.
    
    x should be (...)
    """
    
    psi = win.fst2d_psi_factory([7, 7], include_avg=False)
    
    layer_params = layerO((1,1), 'valid')
    nfeat = 32
    
    with tf.variable_scope('wst_net_v1', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.expand_dims(x, -1)


        # ..., ...
        U1 = scat2d(x, psi, layer_params)
        
        conv = tf.layers.conv2d(U1, nfeat, (7,1), 1, activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, nfeat, (1,7), 1, activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, nfeat, 5, 1, activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, nfeat, 5, 2, activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, nfeat, 5, 4, activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, nfeat, 5, 8, activation=tf.nn.relu)
        
        fc = tf.contrib.layers.flatten(conv)
        fc = tf.layers.dense(fc, 300)
        out = tf.layers.dense(fc, n_classes)
    return tf.squeeze(out, axis=1)

def load_data(args):
    p_files = glob.glob(args.positive_data_glob)
    n_files = glob.glob(args.negative_data_glob)
    
    print('Loaded %i/%i positive/negative examples' % (len(p_files), len(n_files)))
    
    p_samples = load_audio_from_files(p_files, "wav", 3000, 8000)
    p_specs = audio2spec(p_samples, 160, 80, 640)
    p_labels = np.ones(len(p_samples))
    
    n_samples = load_audio_from_files(n_files, "wav", 3000, 8000)
    n_specs = audio2spec(n_samples, 160, 80, 640)
    n_labels = np.zeros(len(n_samples))
    
    data = np.concatenate([p_specs, n_specs], axis=0).astype(np.float32)
    labels = np.concatenate([p_labels, n_labels])
    
    return data, labels

def get_train_val_splits(data, labels):
    
    
    idxs = list(range(len(labels)))
    random.shuffle(idxs)
    
    n_train = int(0.8 * len(idxs))
    train_i = idxs[:n_train]
    val_i = idxs[n_train:]
    
    return data[train_i], labels[train_i], data[val_i], labels[val_i]

def train(args):
    network = wst_net_v1
    bs = args.batch_size
    n_classes = 1
    
    data, labels = load_data(args)
    trainX, trainY, valX, valY = get_train_val_splits(data, labels)

    
    
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
        pred_classes = logits_val > 0.5
        pred_probas = tf.nn.softmax(logits_val)
    
        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)
    
            # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits_train, labels=tf.cast(labels, dtype=tf.float32)))
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
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'spectrograms': trainX}, y=trainY,
        batch_size=bs, num_epochs=args.eval_period, shuffle=True)
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'spectrograms': valX}, y=valY,
        batch_size=bs, shuffle=False)
        
    
    for i in range(args.num_epochs // args.eval_period):
        model.train(train_input_fn)

        e = model.evaluate(eval_input_fn, name='eval')
        
        tf.logging.info("{:06d}: Validation Accuracy: {:.4f}".format(i*args.eval_period, e['accuracy']))

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
    parser.add_argument('--model_root', required=True,
                      help='Full path of where to output the results of training.')
    
    parser.add_argument(
        '--positive_data_glob', type=str, default='/scratch0/ilya/locDoc/data/alexa/v1/alexa/alexa/*/*.wav',
        help='Where to find the audio data files (default: %(default)s)')
    parser.add_argument(
        '--negative_data_glob', type=str, default='/scratch0/ilya/locDoc/data/alexa/v1/snow/*.wav',
        help='Where to find the audio data files (default: %(default)s)')
    # Hyperparams
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate to use (default: %(default)s)')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch Size')
    parser.add_argument(
        '--dropout', type=float, default=0.6,
        help='Dropout rate.')
    # Other
    parser.add_argument(
        '--num_epochs', type=int, default=10000,
        help='Number of epochs to run training for.')
    parser.add_argument(
        '--eval_period', type=int, default=2,
        help='Eval after every N epochs.')
                      
                      
    args = parser.parse_args()
    
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
    
    train(args)

if __name__ == '__main__':
    main()