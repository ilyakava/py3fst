"""FST+NNs on audio spectrograms.
"""

from collections import namedtuple
import glob
import logging
import os
import random

random.seed(2020)

import argparse
import numpy as np
import tensorflow as tf

from audio_load import load_audio_from_files, audio2spec
from st_2d import scat2d
import windows as win

import pdb

layerO = namedtuple('layerO', ['strides', 'padding'])

def st_net_v1(x, dropout, reuse, is_training, n_classes):
    """Network to follow ST preprocessing.
    
    x should be (...)
    """
    
    sz = 13
    psi = win.fst2d_psi_factory([sz, sz], include_avg=False)
    
    layer_params = layerO((1,1), 'valid')
    nfeat = 32
    
    with tf.variable_scope('wst_net_v1', reuse=reuse):
        # x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,256,256))
        x = tf.expand_dims(x, -1)

        ### bs, h, w, channels
        U1 = scat2d(x, psi, layer_params)
        h = U1.shape[1]
        w = U1.shape[2]
        ### bs, h, w, time varying, frequency varying
        U1 = tf.reshape(U1, [-1, h, w, sz-1, sz-1])
        ### bs, time varying, frequency varying, h, w
        U1 = tf.transpose(U1, [0,3,4,1,2])

        ds = (sz-1)//2
        rategram = tf.layers.max_pooling3d(U1, (1,ds,1), (1,ds,1), padding='same')
        scalegram = tf.layers.max_pooling3d(U1, (ds,1,1), (ds,1,1), padding='same')

        nsz = (sz-1)**2 // ds
        rategram = tf.reshape(rategram, [-1, nsz, h, w])
        scalegram = tf.reshape(scalegram, [-1, nsz, h, w])

        cortical = tf.concat([rategram, scalegram], axis=1)
        cortical = tf.transpose(cortical, [0,2,3,1])
        
        conv = tf.layers.conv2d(cortical, nfeat, (7,1), 1, activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, nfeat, (1,7), 1, activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, nfeat, 5, 1, activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, nfeat, 5, 2, activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, nfeat, 5, 4, activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, nfeat, 5, 8, activation=tf.nn.relu)
        
        fc = tf.contrib.layers.flatten(conv)
        fc = tf.layers.dense(fc, 300)
        out = tf.layers.dense(fc, n_classes)
    return tf.squeeze(out, axis=1)

def parser(serialized_example):
  """Parses a single tf.Example into x,y tensors."""
  features = tf.parse_single_example(
      serialized_example,
      features={
          'spectrogram': tf.FixedLenFeature([256*256], tf.float32),
          'label': tf.FixedLenFeature([], tf.int64),
      })
  spec = features['spectrogram']

  label = tf.cast(features['label'], tf.int32)
  return spec, label

def train(args):
    network = st_net_v1
    bs = args.batch_size
    n_classes = 1
    
    def input_fn(tfrecord_dir):
        tfrecord_files = tf.io.gfile.glob(os.path.join(tfrecord_dir, '*.tfrecord'))
        random.shuffle(tfrecord_files)
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        
        # Map the parser over dataset, and batch results by up to batch_size
        dataset = dataset.map(parser)
        dataset = dataset.batch(args.batch_size)
        dataset = dataset.shuffle(buffer_size=5*args.batch_size)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        
        features, labels = iterator.get_next()
        
        return features, labels
    
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
    
    train_spec_dnn = tf.estimator.TrainSpec(input_fn = lambda: input_fn(args.train_data_root), max_steps=args.num_epochs*args.batch_size)
    eval_spec_dnn = tf.estimator.EvalSpec(input_fn = lambda: input_fn(args.val_data_root), steps=args.eval_period*args.batch_size)
    
    tf.estimator.train_and_evaluate(model, train_spec_dnn, eval_spec_dnn)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
    parser.add_argument('--model_root', required=True,
                      help='Full path of where to output the results of training.')
    
    parser.add_argument(
        '--train_data_root', type=str, default='/scratch0/ilya/locDoc/data/alexa/v2/train/',
        help='Where to find the tfrecord files (default: %(default)s)')
    parser.add_argument(
        '--val_data_root', type=str, default='/scratch0/ilya/locDoc/data/alexa/v2/val/',
        help='Where to find the tfrecord files (default: %(default)s)')
    # Hyperparams
    parser.add_argument(
        '--lr', type=float, default=1e-4,
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