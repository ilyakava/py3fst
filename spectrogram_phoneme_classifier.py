"""Classify Phonemes.
"""

from functools import partial
import logging
import os

import argparse
import numpy as np
import tensorflow as tf

from util.log import write_metadata
from util.misc import mkdirp
from util.phones import load_vocab
from networks.arguments import add_basic_NN_arguments
from networks.spectrogram_networks import amazon_net
from networks.spectrogram_data import parser, time_cut_parser, input_fn, identity_serving_input_receiver_fn

import pdb

sr = 16000

def train(args):
    network = amazon_net
    bs = args.batch_size
    phn2idx, idx2phn, phns = load_vocab()
    n_classes = len(phns)
    spec_h = args.feature_height
    spec_w = args.tfrecord_example_length // args.hop_length
    spec_cut_w = args.network_example_length // args.hop_length
        
    train_parser = partial(time_cut_parser, h=spec_h, in_w=spec_w, out_w=spec_cut_w)    
    eval_parser = partial(parser, h=spec_h, w=spec_cut_w)    
    
    ############### END OF SETUP
    
    # Define the model function (following TF Estimator Template)
    def model_fn(features, labels, mode):
        # downsample by 2 here b/c that's what happens in the network
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            labels = labels[:,::2]
    
        # Build the neural network
        # Because Dropout have different behavior at training and prediction time, we
        # need to create 2 distinct computation graphs that still share the same weights.
        logits_train = network(features, args.dropout, reuse=False,
                                is_training=True, n_classes=n_classes, args=args)
        logits_val = network(features, args.dropout, reuse=True,
                                is_training=False, n_classes=n_classes, args=args)
    
        # Predictions
        pred_probas = tf.nn.softmax(logits_val)
        pred_classes = tf.argmax(logits_val, axis=2)
    
        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_probas)
    
        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_train, labels=labels))
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        train_op = optimizer.minimize(loss_op,
                                      global_step=tf.train.get_global_step())
    
        # Evaluate the accuracy of the masks
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
        
        myevalops = {'phoneme_accuracy': acc_op}
        eval_hooks = []
        
        # phoneme text table
        def phoneme_summary(indexes, sz):
            as_str = tf.gather(phns, indexes)
            phones_sep = tf.split(as_str, num_or_size_splits=sz, axis=1)
            phones_cat = tf.strings.join(phones_sep, separator='-')
            return phones_cat
        true_phones = phoneme_summary(labels, spec_cut_w//2)
        pred_phones = phoneme_summary(pred_classes, spec_cut_w//2)
        phones_table = tf.concat([true_phones, pred_phones], axis=1)
        tf.summary.text('true_vs_pred_phones', phones_table)
        
        # phoneme confusion matrix
        confusion = tf.Variable( tf.zeros([n_classes, n_classes],  dtype=tf.int32 ), name='confusion' )
        batch_confusion = tf.confusion_matrix(tf.reshape(labels, [-1]), tf.reshape(pred_classes, [-1]),
                                             num_classes=n_classes,
                                             name='batch_confusion')
        confusion_op = confusion.assign( confusion + batch_confusion )
        myevalops['phoneme_confusion'] = (confusion, confusion_op)
        confusion_img = tf.reshape( tf.cast( confusion, tf.float32),
                                  [1, n_classes, n_classes, 1])
        tf.summary.image('confusion', confusion_img)
        
        # Create a SummarySaverHook
        eval_summary_hook = tf.estimator.SummarySaverHook(
                                        save_steps=1,
                                        output_dir= args.model_root + "/eval",
                                        summary_op=tf.summary.merge_all())
        # Add it to the evaluation_hook list
        eval_hooks.append(eval_summary_hook)
        
        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops=myevalops,
            evaluation_hooks=eval_hooks)
    
        return estim_specs
    
    ###############################

    model = tf.estimator.Estimator(model_fn, model_dir=args.model_root)
    
    train_spec_dnn = tf.estimator.TrainSpec(input_fn = lambda: input_fn(args.train_data_root, bs, train_parser), max_steps=args.max_steps)
    # 45 steps at example_length=19840 is 1 hour
    eval_spec_dnn = tf.estimator.EvalSpec(input_fn = lambda: input_fn(args.val_data_root, bs, eval_parser), steps=45)
    
    tf.estimator.train_and_evaluate(model, train_spec_dnn, eval_spec_dnn)
    
    # model.evaluate(input_fn = lambda: input_fn(args.val_data_root, bs, eval_parser, infinite=False), steps=None)
    
    # saved_model_serving_input_receiver_fn = partial(identity_serving_input_receiver_fn, spec_h, spec_cut_w)
    # model.export_savedmodel(args.model_root, saved_model_serving_input_receiver_fn)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
    parser = add_basic_NN_arguments(parser)
    
    parser.add_argument('--win_length', default=sr//40, type=int,
                    help='...')
    parser.add_argument('--hop_length', default=sr//100, type=int,
                        help='... Becomes the frame size (One frame per this many audio samples).')
    parser.add_argument('--tfrecord_example_length', default=80000, type=int,
                        help='This is the number of samples in the examples in the tf.record. It should be a multiple of hop_length.')
    parser.add_argument('--network_example_length', default=19840, type=int,
        help='This is the number of samples that should be used when input' + \
        ' into the network. It should be a multiple of hop_length.' + \
        ' Length / hop_length = num contexts * context chunk size')
    parser.add_argument('--feature_height', default=80, type=int,
                    help='...')

    args = parser.parse_args()

    # create model dir if needed
    mkdirp(args.model_root)
    
    # save arguments ran with
    write_metadata(args.model_root, args)

    
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