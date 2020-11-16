"""
Builds a TFRecord dataset.
"""

import argparse
import glob
from itertools import product
import io
import json
import os
import random
import sys
import time
from multiprocessing.pool import Pool
from multiprocessing import Lock, Value

from librosa.filters import mel
from librosa.util import pad_center, frame
import numpy as np
from pydub import AudioSegment
from pyrubberband import pyrb
from scipy.ndimage.filters import maximum_filter1d
import soundfile as sf
from tqdm import tqdm

import tensorflow as tf
if not tf.executing_eagerly():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=config)

from util.librosa import librosa_window_fn
from util.log import write_metadata
from util.misc import mkdirp
from util.phones import load_vocab
from augment_audio import augment_audio, extract_example, sample_labels2spectrogam_labels, \
    samples2spectrogam, augment_audio_two_negatives, mix_2_sources, \
    gwn_for_audio, scale_to_peak_windowed_dBFS, augment_audio_with_words, \
    samples2dft, samples2polar, human_hearing_hi_pass, quantize

import pdb

counter = None
random.seed(2020)

sr = 16000

lengths_ms = np.arange(0.4,0.9,0.05)
lengths_probabilities = np.array([16, 31, 80, 73, 76, 46, 26, 12, 4])
lengths_probabilities = lengths_probabilities / lengths_probabilities.sum()

# small room, large room
bedroom = [4.9, 3.6, 3.5]
large_livingroom = [8.5, 6.7, 3.5]
room_dim_opts = [bedroom, large_livingroom]
# absorption small...big
room_absorption_opts = np.arange(0.4,0.9,0.05) # prefer non-echoey rooms

room_sim_opts = list(product(*[room_dim_opts, room_absorption_opts]))


parser = argparse.ArgumentParser()

parser.add_argument("--dataset_type", help="wakeword | TIMIT", default="wakeword")
parser.add_argument("--positive_data_dir", help="Directory of Kaggle Wakewords. Will be separated by speaker for train/val.")
parser.add_argument("--wakeword_metafile", default='data/alexa.annotated.start_ends.json', help="Contains starts and ends of wakewords")
parser.add_argument("--wakeword_pitch_file", default='data/alexa.annotated.pitches.json', help="Contains estimated pitches of wakewords in Hz of fundamental frequency.")
parser.add_argument("--negative_data_dir", help="Directory of Librispeech. Will be separated by speaker for train/val.")
parser.add_argument('--positive_multiplier', default=1, type=int,
                    help='Use each positive raw data example this many times. Each use will be modified by pitch, time, etc.')
parser.add_argument('--max_per_record', default=200, type=int,
                    help='Should be set such that each tfrecord file is ~100MB.')
parser.add_argument("--train_path", help="Path to output training tfrecords.", default=None)
parser.add_argument("--val_path", help="Path to output validation tfrecords.", default=None)
parser.add_argument('--percentage_train', default=0.8, type=float,
                    help='Percentage of raw audio speakers that should be in train set.')
parser.add_argument('--negative_version_percentage', default=0.0, type=float,
                    help='Percentage of outputs to not include a positive data example. Should be non-zero when only 1 word fits in the desired example length.')
parser.add_argument('--win_length', default=sr//40, type=int,
                    help='...')
parser.add_argument('--hop_length', default=sr//100, type=int,
                    help='... Becomes the frame size (One frame per this many audio samples).')
parser.add_argument('--example_length', default=80000, type=int,
                    help='The length in samples of examples to output. It should be a multiple of hop_length.')
parser.add_argument("--noise_type", help="clean | gwn. Used for phoneme dataset", default="clean")
parser.add_argument("--demand_data_dir", help="Noise audio samples.", default=None)
parser.add_argument("--feature_type", help="The type of feature to convert the audio into in the tfrecords. Can be one of: spectrogram", default="spectrogram")

parser.add_argument('--threads', default=8, type=int,
                    help='Number of threads to use')
parser.add_argument('--with_test', action='store_true',
                    help='If set, include a test split')


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def audio2feature(samples, feature_type, **kwargs):
    """
    Returns: flat float list
    """
    # experiment with pre-emph
    
    if feature_type == 'spectrogram':
        feat = samples2spectrogam(samples, **kwargs)
    elif feature_type == 'spectrogram_tf':
        feat = samples2spectrogam_tf(samples, **kwargs)
    elif feature_type == 'LFBE_tf':
        feat = samples2LFBE_tf(samples, **kwargs)
    elif feature_type == 'dft':
        feat = samples2dft(samples, **kwargs)
    elif feature_type == 'polar':
        feat = samples2polar(samples, **kwargs)
    else:
        raise ValueError("audio2feature: Feature type %s is unsupported." % feature_type)
        
    # experiment with dc offset removal
    return tf.train.FloatList(value=feat.reshape(-1))

def samples2spectrogam_tf(samples, win_length, hop_length, n_fft=512):
    """Magnitude of spectrogram.
    For labels use: sample_labels2spectrogam_labels.
    
    Interchangeable with samples2spectrogam.
    
    samples: 1-D array of values in range -1,1
    
    Returns a spectrogram that is n_fft // 2 + 1 high, and
    len(samples) // hop_length + 1 wide
    """
    x = tf.expand_dims(tf.cast(samples, tf.float32), 0) # batch, time
    x = tf.pad(x, ((0, 0),(n_fft//2, n_fft//2)), "REFLECT")
    
    def mywindow_fn(argi, dtype):
        """
        argi is the length of the window that is returned. In this case it is
        n_fft. The window returned will be a win_length window zero padded to
        be n_fft long.
        """
        del argi
        return tf.convert_to_tensor(librosa_window_fn(win_length, n_fft), dtype=dtype)
    
    spec = tf.signal.stft(
        x,
        frame_length=n_fft,
        frame_step=hop_length,
        fft_length=n_fft,
        window_fn=mywindow_fn,
        pad_end=False,
        name='STFT'
    )
    spec = tf.abs(spec)
    # Stevens's power law for loudness, see Dick Lyon's 2017 book, fig 4.2
    spec = spec**0.3
    
    return tf.transpose(tf.squeeze(spec)).numpy()

def samples2LFBE_tf(samples, win_length, hop_length, n_fft=1024, n_mels=256):
    """Magnitude of spectrogram dot product with mel filterbanks
    For labels use: sample_labels2spectrogam_labels.
    
    Interchangeable with samples2spectrogam.
    
    samples: 1-D array of values in range -1,1
    
    Returns a spectrogram that is n_fft // 2 + 1 high, and
    len(samples) // hop_length + 1 wide
    """
    x = tf.expand_dims(tf.cast(samples, tf.float32), 0) # batch, time
    x = tf.pad(x, ((0, 0),(n_fft//2, n_fft//2)), "REFLECT")
    
    def mywindow_fn(argi, dtype):
        """
        argi is the length of the window that is returned. In this case it is
        n_fft. The window returned will be a win_length window zero padded to
        be n_fft long.
        """
        del argi
        return tf.convert_to_tensor(librosa_window_fn(win_length, n_fft), dtype=dtype)
    
    spec = tf.signal.stft(
        x,
        frame_length=n_fft,
        frame_step=hop_length,
        fft_length=n_fft,
        window_fn=mywindow_fn,
        pad_end=False,
        name='STFT'
    )
    spec = tf.abs(spec)
    B = mel(sr, n_fft, n_mels=n_mels)
    lfbe = tf.linalg.matmul(spec, B, transpose_b=True)
    # Stevens's power law for loudness
    lfbe = lfbe**0.3
    
    return tf.transpose(tf.squeeze(lfbe)).numpy()

def audio_labels2feature_labels(samples_label, feature_type, **kwargs):
    """
    Returns: flat int64_list list
    """
    if feature_type in ['spectrogram', 'spectrogram_tf', 'LFBE_tf', 'dft', 'polar']:
        labs = sample_labels2spectrogam_labels(samples_label, **kwargs)
    else:
        raise ValueError("audio_labels2feature_labels: Feature type %s is unsupported." % feature_type)
    return tf.train.Int64List(value=labs)


def serialize_example(samples, samples_label, win_length, hop_length, example_length, feature_type, sample_id=None):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    feature = {}
    
    feature['label'] = _int64_feature(samples_label.max())

    if sample_id is not None:
        feature['id'] = _int64_feature(sample_id)
    
    assert len(samples) == example_length, 'Length of samples is wrong, expected %i received %i.' % (example_length, len(samples))
    assert len(samples_label) == example_length, 'Length of samples_label is wrong, expected %i received %i.' % (example_length, len(samples_label))
    assert example_length % hop_length == 0, 'Example length should be a multiple of hop_length'
    
    example_features = audio2feature(samples, feature_type, win_length=win_length, hop_length=hop_length)
    feature['spectrogram'] = tf.train.Feature(float_list=example_features)
    
    example_feature_labels = audio_labels2feature_labels(samples_label, feature_type, win_length=win_length, hop_length=hop_length)
    feature['spectrogram_label'] = tf.train.Feature(int64_list=example_feature_labels)
    
    samples_as_ints = (samples * 2**15).astype(np.int16)
    audio_segment = AudioSegment(
        samples_as_ints.tobytes(), 
        frame_rate=sr,
        sample_width=2, # samples_as_ints.dtype.itemsize
        channels=1
    )
    audio_feature = audio_segment.raw_data

    feature['audio'] = _bytes_feature(audio_feature)
    feature['audio_label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=samples_label))

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def _update_record_writer(record_writer, next_record, num_examples,
                          output_dir, max_per_record, text=''):
    if record_writer is None or num_examples >= max_per_record:
        if record_writer is not None:
            record_writer.close()

        record_name = '%s_%.4d.tfrecord' % (text, next_record)
        record_writer = tf.io.TFRecordWriter(
            os.path.join(output_dir, record_name))
        next_record += 1
        num_examples = 0
    else:
        num_examples += 1

    return record_writer, next_record, num_examples


def _build_tf_records_from_dataset_wrapper(kwargs):
    return _build_tf_records_from_dataset(**kwargs)

def _build_tf_records_from_dataset(p_files, p_start_ends, p_pitches, n_files, \
    positive_multiplier, output_dir, negative_version_percentage, \
    max_per_record, text, idx_offset, win_length, hop_length, \
    example_length, noise_files, feature_type):
    """
    negative_version_percentage: if True then output an audio file without any wakeword also
    
    Creates 3 classes in the labels: -1 is other speech. +1 is Keyword. 0 is silence.
    """
    global counter
    num_examples = 0
    next_record = 0
    record_writer = None

    os.makedirs(output_dir, exist_ok=True)

    for idx, p_file in enumerate(tqdm(p_files, desc='Positive Recordings Processed')):
        p_start_end = p_start_ends[idx]
        p_pitch = float(p_pitches[idx])
        # experimentally found that +/- 60 Hz was the limit for the rubberband
        # library, this becomes the target pitch
        rand_pitch_range = np.clip([p_pitch - 60, p_pitch + 60], 80, 350)
        for j in range(positive_multiplier):
            # sample_id = idx + idx_offset
            
            n_file = n_files[idx*positive_multiplier + j]
            
            rand_p_duration = lengths_ms[np.random.multinomial(1, lengths_probabilities).tolist().index(1)]
            rand_pitch = np.random.uniform(rand_pitch_range[0],rand_pitch_range[1])
            frequency_multiplier = rand_pitch/p_pitch
            
            try:
                # negative version
                if random.random() < negative_version_percentage:
                    output, labs = augment_audio_with_words(None, None, n_file, rand_p_duration, frequency_multiplier, example_length, target_dBFS=-15.0)
                else:
                    output, labs = augment_audio_with_words(p_file, p_start_end, n_file, rand_p_duration, frequency_multiplier, example_length)
                
                source1 = output.mean(axis=1)
                keyword = labs[:,1]
                voice = np.clip(labs[:,0] + labs[:,2], 0, 1)
                # -1 is other speech. +1 is Keyword. 0 is silence.
                samples_labs = (keyword - voice).astype(int)
                
                if noise_files is None:
                    snr = np.clip(np.random.normal(20,5), 10, 100)
                    source2 = gwn_for_audio(source1, snr=snr)
                else:
                    noise_file = noise_files[idx*positive_multiplier + j]
                    source2, _ = sf.read(noise_file)
                    source2 = scale_to_peak_windowed_dBFS(source2, target_dBFS=-25.0)
                    
                room_dim, room_absorption = random.sample(room_sim_opts, 1)[0]
                source2_distance = 1.0
                source1_distance = np.clip(np.random.normal(0.5, 0.1), 0.1, 1.0)
                
                mix = mix_2_sources(source1, source2, room_dim, room_absorption, source1_distance, source2_distance, target_dBFS=-15.0)
                mix = mix[:example_length]
                
                # observing a lot of low-freq sounds, could be messing up normalization
                mix = human_hearing_hi_pass(mix)
                
                # normalize
                mix = scale_to_peak_windowed_dBFS(mix, target_dBFS=-15.0, rms_window=5)
                
                # quantize
                mix = quantize(mix)
                
                example = serialize_example(mix, samples_labs, win_length, hop_length, example_length, feature_type)
            except Exception as e:
                print("Error augmenting inputs {}, {}\n{}".format(p_file, n_file, e))
                continue
            
            record_writer, next_record, num_examples = _update_record_writer(
                record_writer=record_writer, next_record=next_record,
                num_examples=num_examples, output_dir=output_dir,
                max_per_record=max_per_record, text=text)
            record_writer.write(example)
            if counter:
                with counter.get_lock():
                    counter.value += 1
                    
    record_writer.close()

def _build_tf_records_from_phoneme_dataset_wrapper(kwargs):
    return _build_tf_records_from_phoneme_dataset(**kwargs)

def _build_tf_records_from_phoneme_dataset(files, output_dir, max_per_record, \
    text, idx_offset, win_length, hop_length, example_length, noise_type, \
    feature_type):
    """Builds tfrecords from TIMIT where whole source is used.
    Padded with zeros to be constant length. Segments are extracted
    with half second overlap.
    """
    global counter
    num_examples = 0
    next_record = 0
    record_writer = None

    os.makedirs(output_dir, exist_ok=True)

    for idx, wav_file in enumerate(tqdm(files, desc='Recordings Processed')):
        
        samples, _ = sf.read(wav_file)
    
        # phones (targets)
        phn_file = wav_file.replace("WAV", "PHN")
        phn2idx, _, _ = load_vocab()
        phns = np.zeros(shape=(len(samples),), dtype=int)
        bnd_list = []
        for line in open(phn_file, 'rb').read().splitlines():
            start_sample, _, phn = line.split()
            start_sample = int(start_sample)
            phn = phn.decode(encoding="utf-8")
            phns[start_sample:] = phn2idx[phn]
            bnd_list.append(start_sample)
        
        # example will be split into sections len example_length
        # and overlap of example_overlap
        example_overlap = (sr // 2)
        example_hop = example_length - (sr // 2)
        if (len(samples) < example_length):
            padding = example_length - len(samples)
        else:
            # samples are longer than example_length, so the length traversed
            # when creating the frames will be n * hops + example_length
            padding = example_hop - (len(samples) - example_length) % example_hop

        samples = pad_center(samples, len(samples) + padding)
        phns = pad_center(phns, len(phns) + padding)
        samples = frame(samples, frame_length=example_length, hop_length=example_hop)
        phns = frame(phns, frame_length=example_length, hop_length=example_hop)
        
        for sub_clip_i in range(samples.shape[1]):
            y = np.array(samples[:,sub_clip_i], dtype=float)
            if noise_type == "clean":
                mix = y
            elif noise_type in ("gwn", "GWN"):
                # y = y / max(y.max(), -y.min()) # helps noise
                noise = gwn_for_audio(y, snr=np.random.normal(52,5))
                
                room_dim, room_absorption = random.sample(room_sim_opts, 1)[0]
                source2_distance = 1.0
                source1_distance = np.clip(np.random.normal(0.5, 0.1), 0.1, 1.0)
                
                mix = mix_2_sources(y, noise, room_dim, room_absorption, source1_distance, source2_distance)
                mix = mix[:len(y)]
            else:
                raise ValueError("Unsupported noise_type %s" % noise_type)
                
            mix = scale_to_peak_windowed_dBFS(mix, target_dBFS=-15.0, rms_window=5)
            
            example = serialize_example(mix, phns[:,sub_clip_i], win_length, hop_length, example_length, feature_type)
        
            record_writer, next_record, num_examples = _update_record_writer(
                record_writer=record_writer, next_record=next_record,
                num_examples=num_examples, output_dir=output_dir,
                max_per_record=max_per_record, text=text)
            record_writer.write(example)
            if counter:
                with counter.get_lock():
                    counter.value += 1

    record_writer.close()


def _init_pool(l, c):
    global lock, counter
    lock = l
    counter = c
    
_idx_offset = 0

def build_dataset_randomized(args, p_files, n_files, path, negative_version_percentage=0.0, noise_files=None):
    """Chunking stage.
    
    Chunked by number threads.
    """
    global _idx_offset, counter
    lock = Lock()
    counter = Value('i', 0)
    
    with open(args.wakeword_metafile) as json_file:
        metadata = json.load(json_file)
    with open(args.wakeword_pitch_file) as json_file:
        pitch_data = json.load(json_file)

    pool = Pool(initializer=_init_pool, initargs=(lock, counter),
                processes=args.threads)
    args_list = []
    p_chunk_size = len(p_files) // args.threads
    n_files_pointer = 0
    for i in range(args.threads):
        p_files_chunk = p_files[i*p_chunk_size:((i+1)*p_chunk_size)]
        p_files_chunk_ = [os.path.join(*fname.split('/')[-2:]) for fname in p_files_chunk]
        p_start_ends = [metadata[fname] for fname in p_files_chunk_]
        p_pitches = [pitch_data[fname] for fname in p_files_chunk_]
        n_chunk_size = len(p_files_chunk)*args.positive_multiplier
        
        chunk_noise_files = None
        if noise_files:
            chunk_noise_files = noise_files[:n_chunk_size]
            chunk_noise_files = np.roll(noise_files, -n_chunk_size)
        
        args_list.append({'p_files': p_files_chunk,
                          'p_start_ends': p_start_ends,
                          'p_pitches': p_pitches,
                          'n_files': n_files[n_files_pointer:(n_files_pointer+n_chunk_size)],
                          'positive_multiplier': args.positive_multiplier,
                          'output_dir': path,
                          'negative_version_percentage': negative_version_percentage,
                          'max_per_record': args.max_per_record,
                          'text': '_clean_speech_%.2d' % i,
                          'idx_offset': _idx_offset,
                          'win_length': args.win_length,
                          'hop_length': args.hop_length,
                          'example_length': args.example_length,
                          'noise_files': chunk_noise_files,
                          'feature_type': args.feature_type
        })
        n_files_pointer += n_chunk_size
    
    print('Work scheduled, starting now...')
    start_time = time.time()

    if args.threads > 1:
        for _ in pool.imap_unordered(_build_tf_records_from_dataset_wrapper,
                                     args_list):
            pass
    else:
        for a in args_list:
            _build_tf_records_from_dataset_wrapper(a)

    pool.close()
    pool.join()
    print('Work done, took %.1f min.' % ((time.time() - start_time)/60.0))
    print('Made %i examples.' % counter.value)
    
def build_phoneme_dataset_randomized(args, files, path):
    """Chunking stage.
    
    Chunked by number threads.
    """
    global _idx_offset, counter
    lock = Lock()
    counter = Value('i', 0)
    
    pool = Pool(initializer=_init_pool, initargs=(lock, counter),
                processes=args.threads)
    args_list = []
    files_chunk_size = len(files) // args.threads
    for i in range(args.threads):
        files_chunk = files[i*files_chunk_size:((i+1)*files_chunk_size)]
        files_chunk_ = [os.path.join(*fname.split('/')[-2:]) for fname in files_chunk]
        
        args_list.append({'files': files_chunk,
                          'output_dir': path,
                          'max_per_record': args.max_per_record,
                          'text': '_%s_TIMIT_%.2d' % (args.noise_type, i),
                          'idx_offset': _idx_offset,
                          'win_length': args.win_length,
                          'hop_length': args.hop_length,
                          'example_length': args.example_length,
                          'noise_type': args.noise_type,
                          'feature_type': args.feature_type
        })
    
    print('Work scheduled, starting now...')
    start_time = time.time()

    if args.threads > 1:
        for _ in pool.imap_unordered(_build_tf_records_from_phoneme_dataset_wrapper,
                                     args_list):
            pass
    else:
        for a in args_list:
            _build_tf_records_from_phoneme_dataset_wrapper(a)

    pool.close()
    pool.join()
    print('Work done, took %.1f min.' % ((time.time() - start_time)/60.0))
    print('Made %i examples.' % counter.value)

def train_val_speaker_separation(data_dir, fglob='*.wav', percentage_train=0.8):
    """
    Returns:
        shuffled train paths, shuffled val paths
    """
    p_dirs = np.array(os.listdir(data_dir))
    idxs = list(range(len(p_dirs)))
    random.shuffle(idxs)
    
    n_train = int(percentage_train * len(idxs))
    p_train_dirs = p_dirs[idxs[:n_train]]
    p_val_dirs = p_dirs[idxs[n_train:]]
    
    
    p_train_paths = [os.path.join(data_dir, d, fglob) for d in p_train_dirs]
    p_train_files = [glob.glob(d) for d in p_train_paths]
    p_train_files = [item for sublist in p_train_files for item in sublist]
    
    p_val_paths = [os.path.join(data_dir, d, fglob) for d in p_val_dirs]
    p_val_files = [glob.glob(d) for d in p_val_paths]
    p_val_files = [item for sublist in p_val_files for item in sublist]

    return p_train_files, p_val_files
    
def train_val_speaker_separation_wakeword_metafile(args, percentage_train=0.8):
    """
    Returns:
        shuffled train paths, shuffled val paths
    """
    with open(args.wakeword_metafile) as json_file:
        metadata = json.load(json_file)
    source_files = list(metadata.keys())

    # dir per speaker
    all_speakers = np.array(list(set([path.split('/')[0] for path in source_files])))
    idxs = list(range(len(all_speakers)))
    random.shuffle(idxs)
    
    n_train = int(percentage_train * len(idxs))
    train_speakers = all_speakers[idxs[:n_train]]
    val_speakers = all_speakers[idxs[n_train:]]
    
    train_files = []
    val_files = []
    
    for path in source_files:
        speaker = path.split('/')[0]
        full_path = os.path.join(args.positive_data_dir, path)
        if speaker in train_speakers:
            train_files.append(full_path)
        elif speaker in val_speakers:
            val_files.append(full_path)
        else:
            raise ValueError("%s is neither in train nor val set" % path)

    return train_files, val_files
    
def load_demand_precut(noise_data_dir):
    categories = ['DKITCHEN', 'DLIVING', 'DWASHING', 'OHALLWAY', 'OOFFICE', 'PCAFETER', 'SCAFE']
    category_importance = [1, 1, 0.2, 0.8, 0.8, 0.4, 0.2]
    
    all_files = []
    for i, category in enumerate(categories):
        files = glob.glob(os.path.join(noise_data_dir, category, '*.wav'))
        random.shuffle(files)
        all_files += files[:int(category_importance[i] * len(files))]
    return all_files
        

def create_wakeword_dataset(args):
    n_train_files, n_val_files = train_val_speaker_separation(args.negative_data_dir, fglob='*/*.flac', percentage_train=args.percentage_train)
    p_train_files, p_val_files = train_val_speaker_separation_wakeword_metafile(args, percentage_train=args.percentage_train)
    
    n_train_files_cycles = 1 + (len(p_train_files) * args.positive_multiplier) // len(n_train_files)
    n_val_files_cycles = 1 + (len(p_val_files) * args.positive_multiplier) // len(n_val_files)
    
    if args.train_path is not None and n_train_files_cycles > 1:
        print('Cycling training negative files {} times'.format(n_train_files_cycles))
    if args.val_path is not None and n_val_files_cycles > 1:
        print('Cycling val negative files {} times'.format(n_val_files_cycles))
    
    n_train_files = n_train_files * n_train_files_cycles
    n_val_files = n_val_files * n_val_files_cycles
    
    random.shuffle(n_train_files)
    random.shuffle(n_val_files)
    
    noise_files = None
    if args.demand_data_dir is not None:
        noise_files = load_demand_precut(args.demand_data_dir)
        random.shuffle(noise_files)
    
    if args.train_path is not None:
        build_dataset_randomized(args,
                                p_files=p_train_files,
                                n_files=n_train_files,
                                path=args.train_path,
                                noise_files=noise_files)
    if args.val_path is not None:
        build_dataset_randomized(args,
                                p_files=p_val_files,
                                n_files=n_val_files,
                                path=args.val_path,
                                negative_version_percentage=args.negative_version_percentage,
                                noise_files=noise_files)

def create_phoneme_dataset(args):
    # speaker separation for phonemes
    
    
    if args.train_path is not None:
        train_files = glob.glob(os.path.join(args.positive_data_dir, 'TRAIN/*/*/*.WAV'))
        assert len(train_files)
        build_phoneme_dataset_randomized(args, train_files, args.train_path)
    if args.val_path is not None:
        val_files = glob.glob(os.path.join(args.positive_data_dir, 'TEST/*/*/*.WAV'))
        assert len(val_files)
        build_phoneme_dataset_randomized(args, val_files, args.val_path)

if __name__ == '__main__':
    args = parser.parse_args()
    
    if args.train_path is not None:
        mkdirp(args.train_path)
    if args.val_path is not None:
        mkdirp(args.val_path)

    metadata_path = args.train_path if args.train_path is not None else args.val_path
    write_metadata(metadata_path, args)

    if args.dataset_type == 'wakeword':
        create_wakeword_dataset(args)
    elif args.dataset_type == 'TIMIT':
        create_phoneme_dataset(args)
    else:
        raise ValueError("%s is not a valid --dataset_type" % args.dataset_type)
