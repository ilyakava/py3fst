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
from networks.spectrogram_networks import CBHBH_net, amazon_net
from networks.spectrogram_data import parser, time_cut_parser, input_fn, identity_serving_input_receiver_fn, time_cut_centered_wakeword_parser

import pdb

# from: to
pretrain_assignment_map = {
    "CBHG/prenet/":"CBHBH/prenet/",
    "CBHG/conv_bank_1d/":"CBHBH/conv_bank_1d/",
    "CBHG/highwaynet_featextractor_0/":"CBHBH/highwaynet_featextractor_0/",
    "CBHG/highwaynet_featextractor_1/":"CBHBH/highwaynet_featextractor_1/",
    "CBHG/highwaynet_featextractor_2/":"CBHBH/highwaynet_featextractor_2/",
    "CBHG/highwaynet_featextractor_3/":"CBHBH/highwaynet_featextractor_3/",
}

sr = 16000

def train(args):
    network = CBHBH_net
    bs = args.batch_size
    n_classes = 1
    spec_h = args.feature_height
    spec_w = args.tfrecord_feature_width
    spec_cut_w = args.network_feature_width
        
    # train_parser = partial(time_cut_centered_wakeword_parser, h=spec_h, in_w=spec_w, out_w=spec_cut_w)
    train_parser = partial(time_cut_parser, h=spec_h, in_w=spec_w, out_w=spec_cut_w)    
    eval_parser = partial(parser, h=spec_h, w=spec_cut_w)    
    
    ############### END OF SETUP
    
    # Define the model function (following TF Estimator Template)
    def model_fn(features, labels, mode):

        detection_labels = None
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            # starts with a zero and ends with a zero and has a 1 in it
            istarget = (1-labels[:,0]) * (1-labels[:,-1]) * tf.reduce_max(labels, axis=1)
            # downsample by 2 here b/c that's what happens in the network
            labels = labels[:,::2]
            # idea is that the detection should only happen if the entire word
            # is present
            detection_labels = labels #* tf.expand_dims(istarget, -1)
            
    
        # Build the neural network
        # Because Dropout have different behavior at training and prediction time, we
        # need to create 2 distinct computation graphs that still share the same weights.
        logits_train = network(features, args.dropout, reuse=False,
                                is_training=True, n_classes=n_classes, args=args)
        logits_val = network(features, args.dropout, reuse=True,
                                is_training=False, n_classes=n_classes, args=args)
    
        if args.pretrain_checkpoint is not None:
            tf.train.init_from_checkpoint(args.pretrain_checkpoint, pretrain_assignment_map)
    
        # Predictions
        pred_probas = tf.math.sigmoid(logits_val)
        pred_classes = pred_probas > 0.5
    
        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_probas)
    
        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=tf.reshape(logits_train, [-1]), labels=tf.cast(tf.reshape(detection_labels, [-1]), dtype=tf.float32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        train_op = optimizer.minimize(loss_op,
                                      global_step=tf.train.get_global_step())
    
        # Evaluate the accuracy of the masks
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
        acc2_op = tf.metrics.accuracy(labels=detection_labels, predictions=pred_classes)
        myevalops = {'mask_accuracy': acc_op, 'detection_accuracy': acc2_op}
        eval_hooks = []
        
        # show masks that are being made
        def get_image_summary(ts_lab):
            img_h = 5
            img_w = spec_cut_w
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
        
        summary_specs = features['spectrograms']
        summary_specs = tf.log(1 + summary_specs)
        bm = tf.reduce_min(summary_specs, (1,2), keepdims=True)
        bM = tf.reduce_max(summary_specs, (1,2), keepdims=True)
        img_specs = (summary_specs - bm) / (bM - bm)
        img_specs = tf.expand_dims(img_specs[:,:spec_h,:],-1)
        img_gt = get_image_summary(tf.cast(labels, dtype=tf.float32))
        img_gt_clip = get_image_summary(tf.cast(detection_labels, dtype=tf.float32))
        img_compare = tf.concat([img_hat, img_gt, img_gt_clip, img_specs], axis=1)
        tf.summary.image('Wakeword_Mask_Predictions', img_compare, max_outputs=5)
        
        
        # reduce across time
        clip_probas = tf.reduce_mean(pred_probas, axis=1)
        # clip_probas = tf.clip_by_value(clip_probas, 0, 1)
        clip_gt = tf.reduce_max(detection_labels, axis=1)
        threshes = 1 - (np.arange(1.0,3.5,1.0) / 100.0) # sensitivity = 1 - miss_rate
        for sensitivity in threshes:
            myevalops['whole_clip/specificity_at_sensitivity_%.4f' % sensitivity] = tf.metrics.specificity_at_sensitivity(clip_gt, clip_probas, sensitivity)
        
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
    
    saved_model_serving_input_receiver_fn = partial(identity_serving_input_receiver_fn, spec_h, spec_cut_w)

    model = tf.estimator.Estimator(model_fn, model_dir=args.model_root)
    
    train_spec_dnn = tf.estimator.TrainSpec(input_fn = lambda: input_fn(args.train_data_root, bs, train_parser), max_steps=args.max_steps)
    # Eval Spec, save automatically
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
    MRFA_exporter = tf.estimator.BestExporter(
        name="MRFA_exporter",
        serving_input_receiver_fn=saved_model_serving_input_receiver_fn,
        exports_to_keep=1,
        compare_fn=partial(_key_better, key='whole_clip/specificity_at_sensitivity_0.9700', higher_is_better=True))
    loss_exporter = tf.estimator.BestExporter(
        name="best_loss_exporter",
        serving_input_receiver_fn=saved_model_serving_input_receiver_fn,
        exports_to_keep=1)
    latest_exporter = tf.estimator.LatestExporter(
        name="latest_exporter",
        serving_input_receiver_fn=saved_model_serving_input_receiver_fn,
        exports_to_keep=1)
    exporters = [MRFA_exporter, acc_exporter, loss_exporter, latest_exporter]
    # 45 steps at example_length=19840 is 1 hour
    eval_spec_dnn = tf.estimator.EvalSpec(input_fn = lambda: input_fn(args.val_data_root, bs, eval_parser), steps=45, exporters=exporters)
    
    if args.eval_only:
        model.evaluate(input_fn = lambda: input_fn(args.val_data_root, bs, eval_parser, infinite=False), steps=None)
    else:
        tf.estimator.train_and_evaluate(model, train_spec_dnn, eval_spec_dnn)
    
    
    
    # model.export_savedmodel(args.model_root, saved_model_serving_input_receiver_fn)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
    parser = add_basic_NN_arguments(parser)
    
    parser.add_argument('--win_length', default=sr//40, type=int,
                    help='...')
    parser.add_argument('--hop_length', default=sr//100, type=int,
                        help='... Becomes the frame size (One frame per this many audio samples).')
    parser.add_argument('--tfrecord_feature_width', default=80000//160, type=int,
                        help='This is the width of the 2D features in the train tf.record.')
    parser.add_argument('--network_feature_width', default=19840//160, type=int,
        help='This is the width of the 2D features that should be fed to the network.')
    parser.add_argument('--feature_height', default=80, type=int,
                    help='...')
    parser.add_argument('--pretrain_checkpoint', default=None, type=str,
                    help='Looks like: model_root/model.ckpt-0 or model_root/export/best_acc_exporter/1585776862/variables/variables')

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