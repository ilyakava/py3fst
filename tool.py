import tensorflow as tf
import glob
import os
import numpy as np
import scipy.io.wavfile as wav
import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy import signal
import scipy
import librosa
import pydub

# Mean and Standard deviation of the spectrogram across our dataset
mean = 0.320085779840294
std_dev = 0.17540888436162386

def normalize(specs, tf = True):
    specs = specs**0.3
    if (not tf):
        return (specs-np.mean(specs))/np.std(specs)
    else:
        return (specs-tf.reduce_mean(specs))/tf.reduce_std(specs)

def unnormalize(specs):
    specs = specs * std_dev
    specs = specs + mean
    return specs**(1/0.3)

def load_data(data_dir='./val_19680_easy'):
    tfrecord_files = glob.glob(os.path.join(data_dir, '*.tfrecord'))
    tfrecord_file = tfrecord_files[0]
    
    spec_h = 257 # 80
    # set some variables that are relavant to the network
    # network_example_length = 19840
    # hop_length = 160

    examples = []
    specs = []
    spec_labs = []
    limit = 100

    # for tfrecord_file in tfrecord_files_val:
    for i, example in enumerate(tf.python_io.tf_record_iterator(tfrecord_file)):
        if i < 20:
            continue
        eg_np = tf.train.Example.FromString(example)
        audio_segment = pydub.AudioSegment(
            eg_np.features.feature["audio"].bytes_list.value[0], 
            frame_rate=16000,
            sample_width=2, 
            channels=1
        )
        y = audio_segment.get_array_of_samples()
        examples.append(y)

        spec = eg_np.features.feature["spectrogram"].float_list.value
        spec = np.array(spec).reshape(spec_h,-1)
        spec = spec**(1/0.3) 
        specs.append(spec)
        spec_labs.append(eg_np.features.feature["spectrogram_label"].int64_list.value)
        if i > limit:
            break
    examples = np.array(examples)
    specs = np.array(specs, dtype=np.float32)
    spec_labs = np.array(spec_labs)
    

    # slice it into quarters (hop size of 31)
    # this results in skipping labels
    batched_input = []
    labels = []
    for i in range (specs.shape[0]):
        for j in range(specs.shape[-1]//31):
            start = 31 * j
            end = 31 * (j+1)
            batched_input.append(specs[i,:,start:end])
            labels.append(spec_labs[i,start+20])
    batched_input = np.array(batched_input)
    labels = np.array(labels)
    labels = labels + 1
    
    # compute the masking threshold      
    th_batch = []
    psd_max_batch = []
    
    for i in range(batched_input.shape[0]):
        th, psd_max = generate_th(batched_input[i], 16000)
        th_batch.append(th.T)
        psd_max_batch.append(psd_max) 
     
    th_batch = np.array(th_batch)
    psd_max_batch = np.array(psd_max_batch)
    return batched_input, labels, th_batch, psd_max_batch

"""
    Perceptual Masking, copied from https://github.com/tensorflow/cleverhans/blob/master/examples/adversarial_asr,
    implementation of the paper "Imperceptible, Robust, and Targeted Adversarial Examples for Automatic Speech 
    Recognition" https://arxiv.org/pdf/1903.10346.pdf
"""
def compute_PSD_matrix(specs, window_size):
    """
    First, perform STFT.
    Then, compute the PSD.
    Last, normalize PSD.
    """

    win = np.sqrt(8.0/3.) * specs
    z = abs(win / window_size)
    psd_max = np.max(z*z)
    psd = 10 * np.log10(z * z + 0.0000000000000000001)
    PSD = 96 - np.max(psd) + psd
    return PSD, psd_max   

def Bark(f):
    """returns the bark-scale value for input frequency f (in Hz)"""
    return 13*np.arctan(0.00076*f) + 3.5*np.arctan(pow(f/7500.0, 2))

def quiet(f):
     """returns threshold in quiet measured in SPL at frequency f with an offset 12(in Hz)"""
     thresh = 3.64*pow(f*0.001,-0.8) - 6.5*np.exp(-0.6*pow(0.001*f-3.3,2)) + 0.001*pow(0.001*f,4) - 12
     return thresh

def two_slops(bark_psd, delta_TM, bark_maskee):
    """
    returns the masking threshold for each masker using two slopes as the spread function 
    """
    Ts = []
    for tone_mask in range(bark_psd.shape[0]):
        bark_masker = bark_psd[tone_mask, 0]
        dz = bark_maskee - bark_masker
        zero_index = np.argmax(dz > 0)
        sf = np.zeros(len(dz))
        sf[:zero_index] = 27 * dz[:zero_index]
        sf[zero_index:] = (-27 + 0.37 * max(bark_psd[tone_mask, 1] - 40, 0)) * dz[zero_index:] 
        T = bark_psd[tone_mask, 1] + delta_TM[tone_mask] + sf
        Ts.append(T)
    return Ts
    
def compute_th(PSD, barks, ATH, freqs):
    """ returns the global masking threshold
    """
    # Identification of tonal maskers
    # find the index of maskers that are the local maxima
    length = len(PSD)
    masker_index = signal.argrelextrema(PSD, np.greater)[0]
    
    
    # delete the boundary of maskers for smoothing
    if 0 in masker_index:
        masker_index = np.delete(0)
    if length - 1 in masker_index:
        masker_index = np.delete(length - 1)
    num_local_max = len(masker_index)

    # treat all the maskers as tonal (conservative way)
    # smooth the PSD 
    p_k = pow(10, PSD[masker_index]/10.)    
    p_k_prev = pow(10, PSD[masker_index - 1]/10.)
    p_k_post = pow(10, PSD[masker_index + 1]/10.)
    P_TM = 10 * np.log10(p_k_prev + p_k + p_k_post)
    
    # bark_psd: the first column bark, the second column: P_TM, the third column: the index of points
    _BARK = 0
    _PSD = 1
    _INDEX = 2
    bark_psd = np.zeros([num_local_max, 3])
    bark_psd[:, _BARK] = barks[masker_index]
    bark_psd[:, _PSD] = P_TM
    bark_psd[:, _INDEX] = masker_index
    
    # delete the masker that doesn't have the highest PSD within 0.5 Bark around its frequency 
    for i in range(num_local_max):
        next = i + 1
        if next >= bark_psd.shape[0]:
            break
            
        while bark_psd[next, _BARK] - bark_psd[i, _BARK]  < 0.5:
            # masker must be higher than quiet threshold
            if quiet(freqs[int(bark_psd[i, _INDEX])]) > bark_psd[i, _PSD]:
                bark_psd = np.delete(bark_psd, (i), axis=0)
            if next == bark_psd.shape[0]:
                break
                
            if bark_psd[i, _PSD] < bark_psd[next, _PSD]:
                bark_psd = np.delete(bark_psd, (i), axis=0)
            else:
                bark_psd = np.delete(bark_psd, (next), axis=0)
            if next == bark_psd.shape[0]:
                break        
    
    # compute the individual masking threshold
    delta_TM = 1 * (-6.025  -0.275 * bark_psd[:, 0])
    Ts = two_slops(bark_psd, delta_TM, barks) 
    Ts = np.array(Ts)
    
    # compute the global masking threshold
    theta_x = np.sum(pow(10, Ts/10.), axis=0) + pow(10, ATH/10.) 
 
    return theta_x

def generate_th(specs, fs, window_size=400):
    """
    returns the masking threshold theta_xs and the max psd
    """
    PSD, psd_max= compute_PSD_matrix(specs, window_size)  
    freqs = librosa.core.fft_frequencies(fs, n_fft=512)
    barks = Bark(freqs)

    # compute the quiet threshold 
    ATH = np.zeros(len(barks)) - np.inf
    bark_ind = np.argmax(barks > 1)
    ATH[bark_ind:] = quiet(freqs[bark_ind:])

    # compute the global masking threshold theta_xs 
    theta_xs = []
    # compute the global masking threshold in each window
    for i in range(PSD.shape[1]):
        theta_xs.append(compute_th(PSD[:,i], barks, ATH, freqs))
    theta_xs = np.array(theta_xs)
    return theta_xs, psd_max

class Transform(object):
    '''
    Return: PSD
    '''    
    def __init__(self, window_size):
        self.scale = 8. / 3.
        self.frame_length = int(window_size)
        self.frame_step = int(window_size//4)
        self.window_size = window_size
    
    def __call__(self, x, psd_max_ori):
        z = self.scale *tf.abs(x / self.window_size)
        psd = tf.square(z)
        PSD = tf.pow(10., 9.6) / tf.reshape(psd_max_ori, [-1, 1, 1]) * psd
        return PSD

### Tensorflow implementation of stft ###
def tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
    
def power_to_db(S, ref = 1.0, amin=1e-16, top_db=80.0):
    """Convert a power-spectrogram (magnitude squared) to decibel (dB) units.
    based on http://man.hubwiz.com/docset/LibROSA.docset/Contents/Resources/Documents/generated/librosa.core.power_to_db.html
    """
    
    log_spec = 10.0 * tf_log10(tf.maximum(amin, S))
    log_spec -= 10.0 * tf_log10(tf.maximum(amin, ref))
    log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

    return log_spec

def db_to_power(S_db):
    """Convert a dB-scale spectrogram to a power spectrogram.
     based on http://man.hubwiz.com/docset/LibROSA.docset/Contents/Resources/Documents/_modules/librosa/core/spectrum.html#db_to_power
    """
    return tf.math.pow(10.0, 0.1 * S_db)


def tf_audio2spec(audio, win_length=400, hop_length = 160, n_fft = 512):
    """
        Input: tensor that contains input audio, which has shape (batch_size, audio_length)
        Ouput: spectrogram representation of the input tensor, which is of shape (batch_size, height, width)
    """  
    #audio = tf.pad(audio, ((n_fft//2, n_fft//2),(n_fft//2, n_fft//2)), "REFLECT")
    audio = tf.pad(audio, ((0, 0),(n_fft//2, n_fft//2)), "REFLECT")
    spec = tf.math.abs(tf.signal.stft(signals = audio, frame_length=win_length, frame_step=hop_length, fft_length=n_fft))
    
    spec_db = power_to_db(spec**2)
    spec_db = tf.clip_by_value(spec_db, -55, 65)
    spec = db_to_power(spec_db)**0.5
    
    # Stevens's power law for loudness
    spec = spec**0.3
    spec  = tf.transpose(spec)
    return spec

def warm_start(warm_start_from = "./1594432670"):
    warm_starts = sorted(glob.glob(warm_start_from))
    warm_start_from = warm_starts[-1] if (warm_starts is not None and len(warm_starts)) else None
    warm_start_from = os.path.join(warm_start_from, 'variables/variables')
    id_assignment_map = {'CBHBH/prenet/dense/': 'CBHBH/prenet/dense/', 
                     'CBHBH/prenet/dense_1/': 'CBHBH/prenet/dense_1/', 
                     'CBHBH/highwaynet_featextractor_0/dense1/': 'CBHBH/highwaynet_featextractor_0/dense1/', 
                     'CBHBH/highwaynet_featextractor_0/dense2/': 'CBHBH/highwaynet_featextractor_0/dense2/', 
                     'CBHBH/highwaynet_featextractor_1/dense1/': 'CBHBH/highwaynet_featextractor_1/dense1/', 
                     'CBHBH/highwaynet_featextractor_1/dense2/': 'CBHBH/highwaynet_featextractor_1/dense2/', 
                     'CBHBH/highwaynet_featextractor_2/dense1/': 'CBHBH/highwaynet_featextractor_2/dense1/', 
                     'CBHBH/highwaynet_featextractor_2/dense2/': 'CBHBH/highwaynet_featextractor_2/dense2/', 
                     'CBHBH/highwaynet_featextractor_3/dense1/': 'CBHBH/highwaynet_featextractor_3/dense1/', 
                     'CBHBH/highwaynet_featextractor_3/dense2/': 'CBHBH/highwaynet_featextractor_3/dense2/', 
                     'CBHBH/bottleneck/': 'CBHBH/bottleneck/', 
                     'CBHBH/highwaynet_classifier_0/dense1/': 'CBHBH/highwaynet_classifier_0/dense1/', 
                     'CBHBH/highwaynet_classifier_0/dense2/': 'CBHBH/highwaynet_classifier_0/dense2/', 
                     'CBHBH/highwaynet_classifier_1/dense1/': 'CBHBH/highwaynet_classifier_1/dense1/', 
                     'CBHBH/highwaynet_classifier_1/dense2/': 'CBHBH/highwaynet_classifier_1/dense2/', 
                     'CBHBH/highwaynet_classifier_2/dense1/': 'CBHBH/highwaynet_classifier_2/dense1/', 
                     'CBHBH/highwaynet_classifier_2/dense2/': 'CBHBH/highwaynet_classifier_2/dense2/', 
                     'CBHBH/highwaynet_classifier_3/dense1/': 'CBHBH/highwaynet_classifier_3/dense1/', 
                     'CBHBH/highwaynet_classifier_3/dense2/': 'CBHBH/highwaynet_classifier_3/dense2/', 
                     'CBHBH/highwaynet_classifier_4/dense1/': 'CBHBH/highwaynet_classifier_4/dense1/', 
                     'CBHBH/highwaynet_classifier_4/dense2/': 'CBHBH/highwaynet_classifier_4/dense2/', 
                     'CBHBH/highwaynet_classifier_5/dense1/': 'CBHBH/highwaynet_classifier_5/dense1/', 
                     'CBHBH/highwaynet_classifier_5/dense2/': 'CBHBH/highwaynet_classifier_5/dense2/', 
                     'CBHBH/dense/': 'CBHBH/dense/'}
    return warm_start_from, id_assignment_map

def accuracy(logits, labels):
    return np.sum(np.argmax(logits, axis=-1) == np.argmax(labels, axis=-1))/logits.shape[0]