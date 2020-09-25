"""FST+NNs on audio spectrograms.
"""

from functools import partial
import glob
import logging
import os

import argparse
import numpy as np
import tensorflow as tf

from util.log import write_metadata
from util.misc import mkdirp
from networks.arguments import add_basic_NN_arguments
from networks.spectrogram_data import parser, time_cut_parser, input_fn, identity_serving_input_receiver_fn, time_cut_centered_wakeword_parser

import pdb

# from: to
pretrain_assignment_map = {
    "CBHG/prenet/":"CBHBH/prenet/",
    "CBHG/highwaynet_featextractor_0/":"CBHBH/highwaynet_featextractor_0/",
    "CBHG/highwaynet_featextractor_1/":"CBHBH/highwaynet_featextractor_1/",
    "CBHG/highwaynet_featextractor_2/":"CBHBH/highwaynet_featextractor_2/",
    "CBHG/highwaynet_featextractor_3/":"CBHBH/highwaynet_featextractor_3/",
}

sr = 16000

def train(args):
    # dynamically select which model.
    network = getattr(__import__('networks.spectrogram_networks', fromlist=[args.network_name]), args.network_name)
    
    bs = args.batch_size
    n_classes = 3
    spec_h = args.feature_height
    
    n_left = 20
    n_right = 10
    smallest_spec_width = n_left+n_right+1
    inference_width = args.export_feature_width
    if inference_width is None:
        inference_width = smallest_spec_width

    train_parser = partial(parser, h=spec_h, w=args.tfrecord_train_feature_width)    
    eval_parser = partial(parser, h=spec_h, w=args.tfrecord_eval_feature_width)
    
    # setup shift and center for train and val
    assert len(args.train_shift) == len(args.train_center) and \
        len(args.train_center) == len(args.val_center) and \
        len(args.val_center) == len(args.val_shift), "Train/Val Mean/Var normalization vectors must be the same length"
    nc = len(args.train_shift)
    assert spec_h % nc == 0, "Total number of features must be a multiple of number of channels"
    feat_per_channel = spec_h // nc
    if nc > 1:
        train_mean = np.expand_dims(np.repeat(args.train_shift, (feat_per_channel,)),-1)
        train_std = np.expand_dims(np.repeat(args.train_center, (feat_per_channel,)),-1)
        val_mean = np.expand_dims(np.repeat(args.val_shift, (feat_per_channel,)),-1)
        val_std = np.expand_dims(np.repeat(args.val_center, (feat_per_channel,)),-1)
    else:
        train_mean = args.train_shift[0]
        train_std = args.train_center[0]
        val_mean = args.val_shift[0]
        val_std = args.val_center[0]
    
    ############### END OF SETUP
    
    # Define the model function (following TF Estimator Template)
    def model_fn(features, labels, mode):

        detection_labels = None
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            labels = labels[:,n_left:-n_right]
            
            detection_labels = tf.maximum(0,labels)
            
            # this is what happens in the network
            labels = labels + 1
        
        if mode == tf.estimator.ModeKeys.TRAIN:    
            network_spec_w = args.tfrecord_train_feature_width
        elif mode == tf.estimator.ModeKeys.EVAL:
            network_spec_w = args.tfrecord_eval_feature_width
        elif mode == tf.estimator.ModeKeys.PREDICT:
            network_spec_w = inference_width
        else:
            network_spec_w = None
    
        # Build the neural network
        # Because Dropout has different behavior at training and prediction time, we
        # need to create 2 distinct computation graphs that still share the same weights.
        logits_train = network(features, args.dropout, reuse=False,
                                is_training=True, n_classes=n_classes,
                                spec_h=spec_h, spec_w=network_spec_w)
        logits_val = network(features, args.dropout, reuse=True,
                                is_training=False, n_classes=n_classes,
                                spec_h=spec_h, spec_w=network_spec_w)
        
        if args.pretrain_checkpoint is not None:
            # TODO: make sure this is only running on init
            tf.train.init_from_checkpoint(args.pretrain_checkpoint, pretrain_assignment_map)
    
        # Predictions
        pred_probas = tf.nn.softmax(logits_val)
        pred_classes = tf.argmax(logits_val, axis=2, output_type=tf.int32)
        pred_wake = tf.maximum(0, pred_classes-1)
        prob_wake = pred_probas[:,:,2]
    
        # If prediction mode, early return, with logits
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions={'softmax': pred_probas, 'logits': logits_val})
    
        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_train, labels=labels))
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        train_op = optimizer.minimize(loss_op,
                                      global_step=tf.train.get_global_step())
    
        # Evaluate the accuracy of the masks
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
        acc2_op = tf.metrics.accuracy(labels=detection_labels, predictions=pred_wake)
        myevalops = {'mask_accuracy': acc_op, 'detection_accuracy': acc2_op}
        eval_hooks = []
        
        # reduce across time
        clip_probas = tf.reduce_mean(prob_wake, axis=1)
        clip_gt = tf.reduce_max(detection_labels, axis=1)
        n_eval_examples_per_hour = 3600 / (args.tfrecord_eval_feature_width*args.hop_length / float(sr))
        
        if args.eval_only:
            # point of eval only mode is to get the threshold to get a desired
            # performance at.
            thresholds = np.arange(0.0, 1.0, 1.0/1000, dtype=np.float32)
            myevalops['whole_clip/false_positives'] = tf.metrics.false_positives_at_thresholds( clip_gt, clip_probas, thresholds)
            myevalops['whole_clip/false_negatives'] = tf.metrics.false_negatives_at_thresholds( clip_gt, clip_probas, thresholds)
        else:
            sensitivities = 1 - (np.arange(1.0,3.5,1.0) / 100.0) # sensitivity = 1 - miss_rate
            for sensitivity in sensitivities:
                myevalops['whole_clip/specificity_at_sensitivity_%.4f' % sensitivity] = tf.metrics.specificity_at_sensitivity(clip_gt, clip_probas, sensitivity)
            
            specificity = (n_eval_examples_per_hour-1) / n_eval_examples_per_hour
            myevalops['whole_clip/sensitivity_at_1_FA_per_hour'] = tf.metrics.sensitivity_at_specificity(clip_gt, clip_probas, specificity)
            specificity = ((n_eval_examples_per_hour*10) - 1) / (n_eval_examples_per_hour*10)
            myevalops['whole_clip/sensitivity_at_1_FA_per_10_hours'] = tf.metrics.sensitivity_at_specificity(clip_gt, clip_probas, specificity)

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
    
    saved_model_serving_input_receiver_fn = partial(identity_serving_input_receiver_fn, spec_h, inference_width)
    
    warm_starts = sorted(glob.glob(args.warm_start_from)) if args.warm_start_from is not None else None
    if args.warm_start_from is not None:
        assert len(warm_starts), "Provided warm start %s glob that does not resolve to any files" % args.warm_start_from
    if args.export_only_dir is not None:
        # check that warm_start_from exists as a glob directory
        assert warm_starts is not None, "Must provide warm start for exporting"
        assert len(warm_starts), 'Did not find warm start directory: %s' % args.warm_start_from
        # you do not need the checkpoint directory if you have the saved model.
        model = tf.estimator.Estimator(model_fn, model_dir=None, warm_start_from=warm_starts[-1])
    else: # train/eval
        # only works from a fresh start
        warm_start_from = warm_starts[-1] if (warm_starts is not None and len(warm_starts)) else None
        if warm_start_from is not None:
            warm_start_from = os.path.join(warm_start_from, 'variables/variables')
            warm_start_from = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=warm_start_from, vars_to_warm_start=".*")
        model = tf.estimator.Estimator(model_fn, model_dir=args.model_root, warm_start_from=warm_start_from)

    train_spec_dnn = tf.estimator.TrainSpec(input_fn = lambda: input_fn(
        args.train_data_root, bs, train_parser,
        shift=train_mean, center=train_std), max_steps=args.max_steps)
    
    # Export *pb automatically after eval
    def _key_better(best_eval_result, current_eval_result, key, higher_is_better):
        if higher_is_better:
            return best_eval_result[key] < current_eval_result[key]
        else:
            return best_eval_result[key] > current_eval_result[key]
    acc_exporter = tf.estimator.BestExporter(
        name="best_acc_exporter",
        serving_input_receiver_fn=saved_model_serving_input_receiver_fn,
        exports_to_keep=1,
        compare_fn=partial(_key_better, key='mask_accuracy', higher_is_better=True))
    MR_exporter = tf.estimator.BestExporter(
        name="MR_exporter",
        serving_input_receiver_fn=saved_model_serving_input_receiver_fn,
        exports_to_keep=1,
        compare_fn=partial(_key_better, key='whole_clip/sensitivity_at_1_FA_per_hour', higher_is_better=True))
    FAR_exporter = tf.estimator.BestExporter(
        name="FAR_exporter",
        serving_input_receiver_fn=saved_model_serving_input_receiver_fn,
        exports_to_keep=1,
        compare_fn=partial(_key_better, key='whole_clip/specificity_at_sensitivity_0.9900', higher_is_better=True))
    loss_exporter = tf.estimator.BestExporter(
        name="best_loss_exporter",
        serving_input_receiver_fn=saved_model_serving_input_receiver_fn,
        exports_to_keep=1)
    latest_exporter = tf.estimator.LatestExporter(
        name="latest_exporter",
        serving_input_receiver_fn=saved_model_serving_input_receiver_fn,
        exports_to_keep=1)
    exporters = [MR_exporter, FAR_exporter, acc_exporter, loss_exporter, latest_exporter]
    
    # 90 steps * 32 bs at example_length=19840 is 1 hour
    eval_spec_dnn = tf.estimator.EvalSpec(input_fn = lambda: input_fn(
        args.val_data_root, bs, eval_parser, infinite=False,
        shift=val_mean, center=val_std), steps=90, exporters=exporters, throttle_secs=300)
    
    if args.eval_only:
        eval_op_results = model.evaluate(input_fn = lambda: input_fn(
            args.val_data_root, bs, eval_parser, infinite=False,
            shift=val_mean, center=val_std), steps=None)
        fname = os.path.join(args.model_root, 'eval_results.npz')
        np.savez(fname, **eval_op_results)
        print('Saved eval results to: %s' % fname)
    elif args.export_only_dir:
        model.export_savedmodel(args.export_only_dir, saved_model_serving_input_receiver_fn)
    else:
        tf.estimator.train_and_evaluate(model, train_spec_dnn, eval_spec_dnn)
    

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
    parser = add_basic_NN_arguments(parser)
    
    parser.add_argument('--win_length', default=sr//40, type=int,
                    help='...')
    parser.add_argument('--hop_length', default=sr//100, type=int,
                        help='... Becomes the frame size (One frame per this many audio samples).')
    parser.add_argument('--tfrecord_train_feature_width', default=80000//160, type=int,
                        help='This is the width of the 2D features in the train tf.record.')
    parser.add_argument('--tfrecord_eval_feature_width', default=19840//160, type=int,
                        help='This is the width of the 2D features in the eval tf.record.')
    parser.add_argument('--feature_height', default=80, type=int,
                    help='...')
    parser.add_argument('--pretrain_checkpoint', default=None, type=str,
                    help='Looks like: model_root/model.ckpt-0 or \
    model_root/export/best_acc_exporter/1585776862/variables/variables. \
    This is a trained model from which to initialize a subset of weights from \
    accoring to pretrain_assignment_map.')
    parser.add_argument('--export_only_dir', default=None, type=str,
                    help='If this argument is provided the model is only exported. The export is made from the model in model_root')
    parser.add_argument('--export_feature_width', default=None, type=int,
                    help='This is the width of the 2D features to be fed at inference time.')
    parser.add_argument('--warm_start_from', default=None, type=str,
                    help='This is passed into the Estimator initiation. It can \
    be a full trained model from which to initialize all weights from. Looks \
    like model_dir/export/MRFAR_exporter/1587168950. Globing is supported. \
    Only works if the model_root is brand new and has no checkpoints.')
    parser.add_argument('--network_name', default='cortical_net_v0', type=str,
                    help='Name of network to import from networks.spectrogram_networks. e.g. Guo_Li_net')

    args = parser.parse_args()
    
    # create model dir if needed
    mkdirp(args.model_root)
    
    # some argument validation
    if args.warm_start_from is not None:
        assert len(glob.glob(os.path.join(args.model_root, '*model.ckpt*'))) == 0, 'For warm start to work there needs to be no checkpoints in args.model_root'
    
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