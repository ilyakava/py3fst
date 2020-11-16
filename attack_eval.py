"""Attacks trained networks with UniversalPerturbations in time domain.
"""

import glob
import json
import logging
import os
from random import randint, shuffle
import soundfile as sf
import sys

from art.attacks.evasion import UniversalPerturbation
from art.estimators.classification import TensorFlowClassifier
from art.utils import projection
import argparse
import librosa
from librosa.filters import mel
from sklearn.metrics import roc_curve, auc
import numpy as np
import pydub
import tensorflow as tf
from tensorflow.python.framework import meta_graph
from tqdm import tqdm

from augment_audio import scale_to_peak_windowed_dBFS
from util.ft import unnormalize_0_1
from util.librosa import librosa_window_fn
from util.log import write_metadata
from util.misc import mkdirp
from networks.arguments import add_basic_NN_arguments
from sklearn.metrics.ranking import _binary_clf_curve

import pdb

def attack(args):
    exported_model = sorted(glob.glob(os.path.join(args.model_root, '*')))[0]
    ckpt_file = os.path.join(exported_model, 'mode_resave.ckpt')
    n_fft = 1024
    n_mels = 256
    sr = 16000
    hop_length = 160
    win_length = 400
    bs = args.batch_size
    
    batch_item_length = hop_length*(args.tfrecord_eval_feature_width-1)
    # batch_item_length = n_fft-hop_length + hop_length*args.tfrecord_eval_feature_width
    
    with tf.Graph().as_default() as g_1:
        with tf.compat.v1.Session(graph=g_1) as sess:
            model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], exported_model)
            predictor = tf.contrib.predictor.from_saved_model(exported_model)
            
            signature = model.signature_def
            signature_key = "serving_default"
            input_key = "spectrograms"
            output_key = "softmax"
    
            x_tensor_name = signature[signature_key].inputs[input_key].name
            y_tensor_name = signature[signature_key].outputs[output_key].name
    
            input_ph = sess.graph.get_tensor_by_name(x_tensor_name)
            pred_probas = sess.graph.get_tensor_by_name(y_tensor_name)
            prob_wake = pred_probas[:,:,2]
            clip_probas = tf.reduce_mean(prob_wake, axis=1,  name='clip_probas')
            
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
        
        input_td = tf.pad(input_td_placeholder, ((0, 0),(n_fft//2, n_fft//2)), "REFLECT")
        
        spec = tf.signal.stft(
            input_td,
            frame_length=n_fft,
            frame_step=hop_length,
            fft_length=n_fft,
            window_fn=mywindow_fn,
            pad_end=False,
            name='STFT'
        )
        spec = tf.abs(spec)
        B = mel(sr, n_fft, n_mels=n_mels)
        spec = tf.linalg.matmul(spec, B, transpose_b=True)
        # Stevens's power law for loudness
        spec = spec**0.3
        spec = tf.transpose(spec, perm=[0,2,1])
        
        # need the global mean/variance normalization here
        spec = tf.subtract(spec, args.val_shift)
        spec = tf.divide(spec, args.val_center, name='spec_from_time')
    
    # combine graphs
    g_combined = tf.get_default_graph()
    combined_sess = tf.Session(graph=g_combined)
    combined_input = tf.placeholder(tf.float32, shape=(None, batch_item_length), name='time_input')
    
    meta_graph2 = tf.train.export_meta_graph(graph=g_2)
    meta_graph.import_scoped_meta_graph(meta_graph2, input_map={'time_input': combined_input}, import_scope='g_2')
    out1 = g_combined.get_tensor_by_name('g_2/spec_from_time:0')
    
    meta_graph1 = tf.train.export_meta_graph(graph=g_1)
    new_saver = meta_graph.import_scoped_meta_graph(meta_graph1, input_map={x_tensor_name: out1}, import_scope=None)
    
    out_final = g_combined.get_tensor_by_name('clip_probas:0')
    
    restorer = tf.train.Saver()
    
    combined_sess.run(tf.global_variables_initializer())
    
    restorer.restore(combined_sess, ckpt_file)
    
    # define data generator for eval, just gives audio
    eval_tfrecord_files = glob.glob(os.path.join(args.val_data_root, '*.tfrecord'))
    assert len(eval_tfrecord_files), 'Found no val data'
    def batched_tfrecord_generator():
        batched_x = []
        batched_y = []
        for tfrecord_file in eval_tfrecord_files:
            for i, example in enumerate(tf.python_io.tf_record_iterator(tfrecord_file)):
                eg_np = tf.train.Example.FromString(example)
                audio_segment = pydub.AudioSegment(
                    eg_np.features.feature["audio"].bytes_list.value[0], 
                    frame_rate=16000,
                    sample_width=2, 
                    channels=1
                )
                x = np.array(audio_segment.get_array_of_samples()) / 2**15
                y = eg_np.features.feature["spectrogram_label"].int64_list.value
                # clip_gt as in spectrogram_NN
                y = max(0,max(*y)) - 1
                
                batched_x.append(x)
                batched_y.append(y)
                
                if len(batched_x) == args.batch_size:
                    yield (np.array(batched_x), 1+np.array(batched_y))
                    batched_x = []
                    batched_y = []

        while True:
            yield (None, None)
    
    # count number of examples in data, is fast so nbd.
    if args.n_eval_batches is None:
        print('Counting eval examples...')
        eval_dset = batched_tfrecord_generator()
        data, _ = next(eval_dset)
        n_eval_batches = 0
        while data is not None:
            n_eval_batches += 1
            data, _ = next(eval_dset)
        
        print('Found eval batch count to be {}. Include this next time...'.format(n_eval_batches))
    else:
        n_eval_batches = args.n_eval_batches
    
    # placeholder for results
    clip_predictions = np.zeros((args.batch_size*n_eval_batches,))
    clip_gt = np.zeros((args.batch_size*n_eval_batches,),dtype=int)
    
    
    itr = 0
    noise_updates = 0
    minibatch_i = 0
    fooling_rate = 0.0
    # Init universal perturbation
    # https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/universal_perturbation.py#L110
    
    # music noise load
    noise = 0
    noise_name = 'none'
    if args.load_noise is not None:
        noise, _ = sf.read(args.load_noise)
        base = os.path.basename(args.load_noise)
        noise_name, _ = os.path.splitext(base)
        if args.init_noise_rescale is not None:
            print('Rescaling noise ...')
            noise = args.init_noise_rescale*noise
        
        
    # eval
    pbar = tqdm(total=n_eval_batches, desc="Evaluating...")
    eval_i = 0
    eval_dset = batched_tfrecord_generator()
    data, labels = next(eval_dset)
    assert noise_name == 'none' or len(noise) == data.shape[1], 'Data must match noise length'
    err_snrs = []
    
    while data is not None:
        feed_dict = {combined_input: data + noise}
        pred = combined_sess.run(out_final, feed_dict)
        clip_predictions[eval_i*bs:(eval_i+1)*bs] = pred
        clip_gt[eval_i*bs:(eval_i+1)*bs] = labels
        
        err_snr = snr(data, data + noise)

        err_snrs.append(err_snr)
        eval_i += 1
        pbar.update(1)
        
        data, labels = next(eval_dset)
    
    pbar.close()
    
    fpr, tpr, thresholds = roc_curve(clip_gt, clip_predictions)
    auc_ = auc(fpr, tpr)
    fpr, fnr, thresholds = det_curve(clip_gt, clip_predictions)
    
    msg = 'SNR {:.2f} AUC {:.4f}'.format(sum(err_snrs)/eval_i, auc_)
    tf.compat.v1.logging.info(msg)
    
    # save
    fname = ''
    if args.save_suffix:
        fname = '{}_{}.npz'.format(noise_name, args.save_suffix)
    else:
        fname = '.npz'
    fname = os.path.join(args.model_root, fname)
    np.savez(fname, false_positives=fpr, false_negatives=fnr, thresholds=thresholds, noise=noise, snr=sum(err_snrs)/eval_i, auc=auc_)
    print('Saved {}'.format(fname))
        

def det_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """Currently in scikit-learn 0.24.dev0
    https://scikit-learn.org/dev/modules/generated/sklearn.metrics.det_curve.html
    """
    if len(np.unique(y_true)) != 2:
        raise ValueError("Only one class present in y_true. Detection error "
                         "tradeoff curve is not defined in that case.")

    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
    )

    fns = tps[-1] - tps
    p_count = tps[-1]
    n_count = fps[-1]

    # start with false positives zero
    first_ind = (
        fps.searchsorted(fps[0], side='right') - 1
        if fps.searchsorted(fps[0], side='right') > 0
        else None
    )
    # stop with false negatives zero
    last_ind = tps.searchsorted(tps[-1]) + 1
    sl = slice(first_ind, last_ind)

    # reverse the output such that list of false positives is decreasing
    return (
        fps[sl][::-1] / n_count,
        fns[sl][::-1] / p_count,
        thresholds[sl][::-1]
    )

def logit2iswake(logit):
    """
    For logits return
    """
    return int(np.argmax(logit) == 2)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
def targetsnr2targetnoisepower_alp(log_power, target_snr):
    return 10**(log_power - target_snr / 10.0)

def targetsnr2targetnoisepower(signal_power, target_snr):
    return 10**(np.log10(signal_power) - target_snr / 10.0)
    
def scale_to_power(target_power, now_power):
    return (target_power / now_power)**0.5
    

def snr(clean, noisey):
    Psignal = (clean**2).sum()
    Pnoise = ((clean - noisey)**2).sum()
    snrdb = 10*np.log10(Psignal/Pnoise)
    return snrdb

def ww_accuracies(predictions, labels):
    labels = labels - 1
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
        
    parser = add_basic_NN_arguments(parser)
        
    
    parser.add_argument('--load_noise', default=None, type=str,
        help='File to load the noise from')
    parser.add_argument(
        '--tfrecord_eval_feature_width', type=int, default=10,
        help='....')
    parser.add_argument(
        '--n_eval_batches', type=int, default=None,
        help='....')
    parser.add_argument('--save_suffix', default='', type=str,
        help='..')
    parser.add_argument(
        '--init_noise_rescale', type=float, default=None,
        help='Rescale the noise. Useful when testing clean noises.')

    args = parser.parse_args()
    
    # some argument validation
    msg = 'Need an exported model glob that resolves to at least 1 directory'
    assert args.model_root is not None, msg
    assert glob.glob(os.path.join(args.model_root, '*')), msg

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
    
    attack(args)

if __name__ == '__main__':
    main()