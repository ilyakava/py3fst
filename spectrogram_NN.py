"""FST+NNs on audio spectrograms.
"""

from functools import partial
import logging
import os

import argparse
import numpy as np
import tensorflow as tf

from util.log import write_metadata
from util.misc import mkdirp
from networks.arguments import add_basic_NN_arguments
from networks.spectrogram_networks import amazon_net
from networks.spectrogram_data import parser, time_cut_parser, input_fn, identity_serving_input_receiver_fn

import pdb

sr = 16000

def train(args):
    network = amazon_net
    bs = args.batch_size
    n_classes = 1
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
        pred_probas = tf.math.sigmoid(logits_val)
        pred_classes = pred_probas > 0.5
    
        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_probas)
    
        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=tf.reshape(logits_train, [-1]), labels=tf.cast(tf.reshape(labels, [-1]), dtype=tf.float32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        train_op = optimizer.minimize(loss_op,
                                      global_step=tf.train.get_global_step())
    
        # Evaluate the accuracy of the masks
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
        
        # show masks that are being made
        def get_image_summary(ts_lab):
            img_h = 5
            img_w = args.network_example_length // args.hop_length
            # repeat twice
            ts_lab_rep2 = tf.tile(tf.expand_dims(ts_lab, -1),(1,1,2))
            ts_lab_rep2 = tf.reshape(ts_lab_rep2, [-1, img_w])
            imgs = tf.tile(tf.expand_dims(ts_lab_rep2,-1), (1,1,img_h))
            imgs = tf.transpose(imgs, [0,2,1])
            imgs = tf.expand_dims(imgs, -1)
            return imgs
        
        # image-ify the features
        img_hat = get_image_summary(pred_probas)
        spec_h = args.feature_height # could cut off higher half frequencies here
        # log_specs = tf.log(features)
        summary_specs = features['spectrograms']
        bm = tf.reduce_min(summary_specs, (1,2), keepdims=True)
        bM = tf.reduce_max(summary_specs, (1,2), keepdims=True)
        img_specs = (summary_specs - bm) / (bM - bm)
        img_specs = tf.expand_dims(img_specs[:,:spec_h,:],-1)
        img_gt = get_image_summary(tf.cast(labels, dtype=tf.float32))
        img_compare = tf.concat([img_hat, img_gt, img_specs], axis=1)
        tf.summary.image('Wakeword_Mask_Predictions', img_compare, max_outputs=5)
        
        myevalops = {'mask_accuracy': acc_op}
        # reduce across time
        clip_probas = tf.reduce_mean(pred_probas, axis=1)
        # clip_probas = tf.clip_by_value(clip_probas, 0, 1)
        clip_gt = tf.reduce_max(labels, axis=1)
        threshes = 1 - (np.arange(1.0,3.5,1.0) / 100.0) # sensitivity = 1 - miss_rate
        for sensitivity in threshes:
            myevalops['whole_clip/specificity_at_sensitivity_%.4f' % sensitivity] = tf.metrics.specificity_at_sensitivity(clip_gt, clip_probas, sensitivity)
        
        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops=myevalops)
    
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