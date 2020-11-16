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

import pdb

def attack(attack_eps, args):
    exported_model = sorted(glob.glob(os.path.join(args.model_root, '*')))[0]
    ckpt_file = os.path.join(exported_model, 'mode_resave.ckpt')
    n_fft = 1024
    n_mels = 256
    sr = 16000
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
        B = mel(sr, n_fft, n_mels=n_mels)
        spec = tf.linalg.matmul(spec, B, transpose_b=True)
        # Stevens's power law for loudness
        spec = spec**0.3
        spec = tf.transpose(spec, perm=[0,2,1])
        
        # need the global mean/variance normalization here
        spec = tf.subtract(spec, args.train_shift)
        spec = tf.divide(spec, args.train_center, name='spec_from_time')
    
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
    
    ### classifier is ready
    meta_attack = UniversalPerturbation(classifier=classifier, attacker=args.attacker, norm=parse_attack_norm(args.attack_norm), eps=attack_eps)
    attacker = meta_attack._get_attack(meta_attack.attacker, meta_attack.attacker_params)
    attacker.verbose = False
    
    # define data generator for training
    train_tfrecord_files = glob.glob(os.path.join(args.train_data_root, '*.tfrecord'))
    shuffle(train_tfrecord_files)
    assert len(train_tfrecord_files), 'Found no training data'
    def tfrecord_generator():
        while True:
            for tfrecord_file in train_tfrecord_files:
                for i, example in enumerate(tf.python_io.tf_record_iterator(tfrecord_file)):
                    eg_np = tf.train.Example.FromString(example)
                    audio_segment = pydub.AudioSegment(
                        eg_np.features.feature["audio"].bytes_list.value[0], 
                        frame_rate=16000,
                        sample_width=2, 
                        channels=1
                    )
                    x = np.array([audio_segment.get_array_of_samples()]) / 2**15
                    y = eg_np.features.feature["audio_label"].int64_list.value
                    
                    # pick rand segment and yield
                    start = randint(0,x.shape[1] - batch_item_length)
                    end = start + batch_item_length
                    yield (x[:,start:end], 1+y[start + (2*batch_item_length//3)])
    
    # define data generator for eval
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
                y = eg_np.features.feature["audio_label"].int64_list.value
                
                target_n_segments = 32
                start_idxs = list(range(0,len(y) - batch_item_length))
                dec = len(y) // target_n_segments
                
                for start_idx in start_idxs[::dec][:target_n_segments]:
                    batched_x.append(x[start_idx:(start_idx+batch_item_length)])
                    batched_y.append(y[start_idx + 2*(batch_item_length//3)])
                    
                    if len(batched_x) == args.batch_size:
                        yield (np.array(batched_x), 1+np.array(batched_y))
                        batched_x = []
                        batched_y = []
        
        while True:
            yield (None, None)
            
    # start to loop through data and attack
    train_dset = tfrecord_generator()
    
    det_accs_summary = []
    err_snrs_summary = []
    mask_accs_summary = []
    noise_summary = []
    
    itr = 0
    noise_updates = 0
    minibatch_i = 0
    fooling_rate = 0.0
    desired_fool_rate = args.attack_fool_rate
    # Init universal perturbation
    # https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/universal_perturbation.py#L110
    
    # music noise setup
    assert args.init_noise is not None
    noise, _ = sf.read(args.init_noise)
    noise = noise[:5824]
    assert len(noise) == 5824
    # find how to rescale
    print('Reading eval data for power calculation')
    log_powers = []
    eval_dset = batched_tfrecord_generator()
    data, _ = next(eval_dset)
    while data is not None:
        log_powers.append(np.log10((data**2).mean(axis=0).sum()))
        data, _ = next(eval_dset)
    noise_power_target = targetsnr2targetnoisepower_alp(np.array(log_powers).mean(), args.init_noise_snr)
    noise_power_now = (noise**2).sum()
    a = scale_to_power(noise_power_target, noise_power_now)
    noise = a*noise
    music = noise.copy()
    # check rescaling
    print('Rescaled by {}'.format(a))
    print('Reading eval data for snr calculation')
    err_snrs = []
    eval_dset = batched_tfrecord_generator()
    data, _ = next(eval_dset)
    while data is not None:
        err_snrs.append(snr(data, data + noise))
        data, _ = next(eval_dset)
    err_snrs = np.array(err_snrs)
    print('Starting noise mean snr is {:.02f} +/- {:.02f}'.format(err_snrs.mean(), err_snrs.std()))
    
    
    # start
    while itr < args.max_steps and fooling_rate < desired_fool_rate:
        for i in tqdm(range(args.eval_period), desc="[{:04d}/{:04d}] Training Universal Perturbation".format(minibatch_i, args.max_steps // args.eval_period)):
            x_i, original_label = next(train_dset)
            current_label = np.argmax(meta_attack.estimator.predict(x_i + noise)[0])
            
            if current_label == original_label:
                # Compute adversarial perturbation
                adv_xi = attacker.generate(x_i + noise, y=[original_label])
                new_label = np.argmax(meta_attack.estimator.predict(adv_xi)[0])
        
                # If the class has changed, update v
                if current_label != new_label:
                    noise_only = adv_xi - x_i - music
        
                    # Project on L_p ball
                    noise_only = projection(noise_only, meta_attack.eps, meta_attack.norm)
                    noise = music + noise_only
                    noise_updates += 1
            itr += 1
        minibatch_i += 1
        
        # eval
        print('Beginning evaluation with modified music with MSNR {:.02f}'.format(snr(music, noise)))
        pbar = tqdm(total=args.eval_steps, desc="Evaluating...")
        eval_dset = batched_tfrecord_generator()
        data, labels = next(eval_dset)
        eval_i = 0
        mask_accs0 = []
        det_accs0 = []
        mask_accs1 = []
        det_accs1 = []
        err_snrs = []
        
        while data is not None and eval_i < args.eval_steps:
            predictions_before = classifier.predict(data)
            predictions_after = classifier.predict(data + noise)
            
            mask_acc_0, det_acc_0 = ww_accuracies(predictions_before, labels)
            mask_acc_1, det_acc_1 = ww_accuracies(predictions_after, labels)
            err_snr = snr(data, data + noise)
            
            mask_accs0.append(mask_acc_0)
            det_accs0.append(det_acc_0)
            mask_accs1.append(mask_acc_1)
            det_accs1.append(det_acc_1)
            err_snrs.append(err_snr)
            eval_i += 1
            pbar.update(1)
            
            data, labels = next(eval_dset)
        
        pbar.n = args.eval_steps
        pbar.close()
        msg = '[itr {:05d}] [update {:05d}] Mask acc {:.4f}/{:.4f}. Det acc {:.4f}/{:.4f}. SNR {:.2f}'.format( \
            itr, noise_updates, \
            sum(mask_accs0)/eval_i, sum(mask_accs1)/eval_i, \
            sum(det_accs0)/eval_i, sum(det_accs1)/eval_i, \
            sum(err_snrs)/eval_i)
        tf.compat.v1.logging.info(msg)
        err_snrs_summary.append(sum(err_snrs)/eval_i)
        det_accs_summary.append(sum(det_accs1)/eval_i)
        mask_accs_summary.append(sum(mask_accs1)/eval_i)
        noise_summary.append(noise)
        fooling_rate = 1 - sum(mask_accs1)/eval_i
        
        # save after every eval
        fname = 'ua_{}_0p{}_{}'.format(args.attacker, str(attack_eps)[2:], args.max_steps)
        if args.save_suffix:
            fname += '_%s.npz' % args.save_suffix
        else:
            fname += '.npz'
        fname = os.path.join(args.model_root, fname)
        np.savez(fname, det_acc=det_accs_summary, snr=err_snrs_summary, mask_acc=mask_accs_summary, noise=noise_summary)
        
        
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
        
    parser.add_argument('--attacker', default='deepfool', type=str,
        help='..')
    parser.add_argument('--attack_norm', default='inf', type=str,
        help='..')
    
    parser.add_argument(
        '--attack_eps', type=float, default=0.0001,
        help='list of attack eps to try')
    parser.add_argument('--attack_eps_step', default=0.00001, type=float,
        help='..')
    parser.add_argument('--attack_fool_rate', default=1.0, type=float,
        help='Stop running when this fool rate is reached')
    parser.add_argument('--save_suffix', default='', type=str,
        help='..')
    parser.add_argument('--init_noise', default=None, type=str,
        help='File to initialize the noise with')
    parser.add_argument(
        '--eval_steps', type=int, default=10,
        help='Number of batches to run during eval.')
    parser.add_argument(
        '--init_noise_snr', type=float, default=15.0,
        help='Rescale the noise to have this snr before looking for perturbations')

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
    
    print('Attacking with {}'.format(args.attack_eps))
    attack(args.attack_eps, args)

if __name__ == '__main__':
    main()