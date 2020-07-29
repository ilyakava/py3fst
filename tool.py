import tensorflow as tf
import glob
import os
import numpy as np
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

def warm_start(warm_start_from = "./1594667126"):
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