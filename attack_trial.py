"""Attacks trained networks in time domain.
"""

import glob
import json
import logging
import os

from art.attacks.evasion import FastGradientMethod, AutoProjectedGradientDescent
from art.estimators.classification import TensorFlowClassifier
import argparse
import librosa
import numpy as np
import pydub
import tensorflow as tf
from tensorflow.python.framework import meta_graph

from augment_audio import scale_to_peak_windowed_dBFS
from util.ft import unnormalize_0_1
from util.librosa import librosa_window_fn
from util.log import write_metadata
from util.misc import mkdirp

import pdb


def attack(args):
    exported_model = sorted(glob.glob(args.exported_model_glob))[0]
    ckpt_file = os.path.join(exported_model, 'mode_resave.ckpt')
    n_fft = 512
    hop_length = 160
    win_length = 400
    
    batch_item_length = n_fft-hop_length + hop_length*31
    
    with tf.Graph().as_default() as g_1:
        with tf.compat.v1.Session(graph=g_1) as sess:
            model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], exported_model)
            predictor = tf.contrib.predictor.from_saved_model(exported_model)
            
            signature = model.signature_def
            signature_key = "serving_default"
            input_key = "spectrograms"
            output_key = "logits"
    
            x_tensor_name = signature[signature_key].inputs[input_key].name
            y_tensor_name = signature[signature_key].outputs[output_key].name
    
            input_ph = sess.graph.get_tensor_by_name(x_tensor_name)
            logits = sess.graph.get_tensor_by_name(y_tensor_name)
            flat_logits = tf.reshape(logits, [-1,3], name='flat_logits') # just gets rid of middle time dimension which is equal to 1 in the output 31 case
            
            saver = tf.train.Saver()
            saver.save(sess, ckpt_file)
    
    def mywindow_fn(argi, dtype):
        """
        argi is the length of the window that is returned. In this case it is
        n_fft. The window returned will be a win_length window zero padded to
        be n_fft long.
        """
        del argi
        return tf.convert_to_tensor(librosa_window_fn(win_length, n_fft), dtype=dtype)
        
    with tf.Graph().as_default() as g_2:
        input_td_placeholder = tf.placeholder(tf.float32, shape=(None, batch_item_length), name='time_input')
        
        spec = tf.signal.stft(
            input_td_placeholder,
            frame_length=n_fft,
            frame_step=hop_length,
            fft_length=n_fft,
            window_fn=mywindow_fn,
            pad_end=False,
            name='STFT'
        )
        spec = tf.abs(spec)
        # Stevens's power law for loudness
        spec = spec**0.3
        spec = tf.transpose(spec, perm=[0,2,1])
        
        # need the global mean/variance normalization here
        spec = tf.subtract(spec, args.data_shift)
        spec = tf.divide(spec, args.data_center, name='spec_from_time')
    
    # combine graphs
    g_combined = tf.get_default_graph()
    combined_sess = tf.Session(graph=g_combined)
    combined_input = tf.placeholder(tf.float32, shape=(None, batch_item_length), name='time_input')
    
    meta_graph2 = tf.train.export_meta_graph(graph=g_2)
    meta_graph.import_scoped_meta_graph(meta_graph2, input_map={'time_input': combined_input}, import_scope='g_2')
    out1 = g_combined.get_tensor_by_name('g_2/spec_from_time:0')
    
    meta_graph1 = tf.train.export_meta_graph(graph=g_1)
    new_saver = meta_graph.import_scoped_meta_graph(meta_graph1, input_map={x_tensor_name: out1}, import_scope=None)
    
    out_final = g_combined.get_tensor_by_name('flat_logits:0')
    
    labels_ph = tf.placeholder(tf.int32, shape=[None, 3])
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=out_final, labels=labels_ph))
    
    restorer = tf.train.Saver()
    
    # ART library wrapper around the tensorflow model
    classifier = TensorFlowClassifier(
        clip_values=(-1, 1),
        input_ph=combined_input,
        output=out_final,
        labels_ph=labels_ph,
        loss=loss_op,
        learning=None,
        sess=combined_sess,
        preprocessing_defences=[],
    )
    
    combined_sess.run(tf.global_variables_initializer())
    
    restorer.restore(combined_sess, ckpt_file)
    
    # create attack
    norm = parse_attack_norm(args.attack_norm)
    eps = args.attack_eps
    eps_step = args.attack_eps_step
    
    attack = AutoProjectedGradientDescent(estimator=classifier, norm=norm, targeted=False, eps=eps, eps_step=eps_step, loss_type='cross_entropy')
    
    # define data generator
    tfrecord_files = glob.glob(args.tfrecord_glob)
    def tfrecord_generator():
        batched_x = []
        batched_y = []
        for tfrecord_file in tfrecord_files:
            for i, example in enumerate(tf.python_io.tf_record_iterator(tfrecord_file)):
                eg_np = tf.train.Example.FromString(example)
                audio_segment = pydub.AudioSegment(
                    eg_np.features.feature["audio"].bytes_list.value[0], 
                    frame_rate=16000,
                    sample_width=2, 
                    channels=1
                )
                x = np.array(audio_segment.get_array_of_samples()) / 2**15
                y = eg_np.features.feature["audio_label"].int64_list.value
                
                for j in range(4):
                    batched_x.append(x[(j*batch_item_length):((j+1)*batch_item_length)])
                    batched_y.append(y[(j*batch_item_length) + 2*(batch_item_length//3)])
                    
                if len(batched_x) == args.batch_size:
                    yield (np.array(batched_x), np.array(batched_y))
                    batched_x = []
                    batched_y = []
        
        while True:
            print('Ran out of data')
            yield (None, None)
            
    # start to loop through data and attack
    labeled_batch = tfrecord_generator()
    data, labels = next(labeled_batch)
    mask_accs0 = []
    det_accs0 = []
    mask_accs1 = []
    det_accs1 = []
    err_snrs = []
    n = 0
    while data is not None:
        predictions_before = classifier.predict(data)
        adv_egs = attack.generate(x=data)
        predictions_after = classifier.predict(adv_egs)
        
        mask_acc_0, det_acc_0 = ww_accuracies(predictions_before, labels)
        mask_acc_1, det_acc_1 = ww_accuracies(predictions_after, labels)
        
        err_snr = snr(data, adv_egs)
        
        mask_accs0.append(mask_acc_0)
        det_accs0.append(det_acc_0)
        mask_accs1.append(mask_acc_1)
        det_accs1.append(det_acc_1)
        err_snrs.append(err_snr)
        n += 1
        
        msg = '[{:03d}] Mask acc {:.4f}/{:.4f} ({:.4f}/{:.4f}). Det acc {:.4f}/{:.4f} ({:.4f}/{:.4f}). SNR {:.2f} ({:.2f})'.format( \
            n, \
            mask_acc_0, mask_acc_1, sum(mask_accs0)/n, sum(mask_accs1)/n, \
            det_acc_0, det_acc_1, sum(det_accs0)/n, sum(det_accs1)/n, \
            err_snr, sum(err_snrs)/n)
        tf.compat.v1.logging.info(msg)
        
        data, labels = next(labeled_batch)

        
        
def snr(clean, noisey):
    Psignal = (clean**2).sum()
    Pnoise = ((clean - noisey)**2).sum()
    snrdb = 10*np.log10(Psignal/Pnoise)
    return snrdb

def ww_accuracies(predictions, labels):
    mask_acc = (np.argmax(predictions, axis=1) == (labels+1)).sum() / len(labels)
    detection_acc = ((np.argmax(predictions, axis=1) == 2) & (labels == 1)).sum() / (labels == 1).sum()
    # other_acc = ((np.argmax(predictions, axis=1) != 2) & (labels != 1)).sum() / (labels != 1).sum()
    # print('Mask acc: %.4f, Detection acc: %.4f, Non-wake acc: %.4f' % (mask_acc, detection_acc, other_acc))
    return (mask_acc, detection_acc)
        
    
def parse_attack_norm(str_norm):
    if str_norm == 'inf':
        return np.inf
    elif str_norm in ['1','2']:
        return int(str_norm)
    else:
        raise('Unsupported attack norm %s' % attack_norm)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
    
    parser.add_argument('--log_dir', default=None, type=str,
        help='..')
    parser.add_argument('--exported_model_glob', default=None, type=str,
        help='Something like /scratch0/ilya/locDownloads/temp_7.15_model/baseline_tfspecv715_bs64_trial2/export/best_loss_exporter/*')
    parser.add_argument('--attack_norm', default='inf', type=str,
        help='..')
    parser.add_argument('--attack_eps', default=None, type=float,
        help='..')
    parser.add_argument('--attack_eps_step', default=None, type=float,
        help='..')
    parser.add_argument('--tfrecord_glob', default=None, type=str,
        help='..')
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch Size')
    parser.add_argument(
        '--data_shift', type=json.loads, default=None,
        help='...')
    parser.add_argument(
        '--data_center', type=json.loads, default=None,
        help='...')

    args = parser.parse_args()
    
    # create log dir if needed
    mkdirp(args.log_dir)
    
    # some argument validation
    msg = 'Need an exported model glob that resolves to at least 1 directory'
    assert args.exported_model_glob is not None, msg
    assert len(glob.glob(os.path.join(args.exported_model_glob))) > 0, msg
    
    # save arguments ran with
    write_metadata(args.log_dir, args)
    
    # get TF logger
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)
    
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(args.log_dir, 'tensorflow.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    
    attack(args)

if __name__ == '__main__':
    main()