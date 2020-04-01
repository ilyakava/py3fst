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

from librosa.util import pad_center, frame
import numpy as np
from pydub import AudioSegment
from pyrubberband import pyrb
from scipy.ndimage.filters import maximum_filter1d
import soundfile as sf
from tqdm import tqdm

import tensorflow as tf

from util.log import write_metadata
from util.misc import mkdirp
from util.phones import load_vocab
from augment_audio import augment_audio, extract_example, samples2mfcc, samples2spectrogam, augment_audio_two_negatives, mix_2_sources, gwn_for_audio

import pdb

counter = None
random.seed(2020)

sr = 16000
pitch_shift_opts = np.arange(-3,3.5,0.5).tolist()
silence_1_opts = np.arange(-0.1, 0.35, 0.05).tolist()
silence_2_opts = np.arange(-0.1, 0.35, 0.05).tolist()
loudness_opts = [1.0]
positive_augment_opts = list(product(*[pitch_shift_opts, silence_1_opts, silence_2_opts, loudness_opts]))

lengths_ms = np.arange(0.4,0.9,0.05)
lengths_probabilities = np.array([16, 31, 80, 73, 76, 46, 26, 12, 4])
lengths_probabilities = lengths_probabilities / lengths_probabilities.sum()

# small room, large room
bedroom = [4.9, 3.6, 3.5]
large_livingroom = [8.5, 6.7, 3.5]
room_dim_opts = [bedroom, large_livingroom]
# absorption small...big
room_absorption_opts = [0.1,0.8,0.85,0.9,0.95,1.0] # prefer non-echoey rooms
# far-ness of wakeword
wakeword_to_mic_rel_distance_opts = [0.25,0.5,1.0]

room_sim_opts = list(product(*[room_dim_opts, room_absorption_opts, wakeword_to_mic_rel_distance_opts]))


parser = argparse.ArgumentParser()

parser.add_argument("--dataset_type", help="wakeword | TIMIT", default="wakeword")
parser.add_argument("--positive_data_dir", help="...")
parser.add_argument("--wakeword_metafile", help="...")
parser.add_argument("--negative_data_dir", help="...")
parser.add_argument('--positive_multiplier', default=1, type=int,
                    help='...')
parser.add_argument('--max_per_record', default=200, type=int,
                    help='Should be set such that each tfrecord file is ~100MB.')
parser.add_argument("--train_path", help="...", default=None)
parser.add_argument("--val_path", help="...", default=None)
parser.add_argument('--percentage_train', default=0.8, type=float,
                    help='...')
parser.add_argument('--win_length', default=sr//40, type=int,
                    help='...')
parser.add_argument('--hop_length', default=sr//100, type=int,
                    help='... Becomes the frame size (One frame per this many audio samples).')
parser.add_argument('--example_length', default=80000, type=int,
                    help='The length in samples of examples to output. It should be a multiple of hop_length.')
parser.add_argument("--noise_type", help="clean | gwn", default="clean")

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

def serialize_example(samples, samples_label, win_length, hop_length, example_length, sample_id=None):
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
    
    spec = samples2spectrogam(samples, win_length, hop_length)
    # spec = samples2mfcc(samples, win_length, hop_length)
    
    feature['spectrogram'] = tf.train.Feature(float_list=tf.train.FloatList(value=spec.reshape(-1)))
    # the labels need to be adapted to the feature size
    samp_max_pool = maximum_filter1d(samples_label, size=args.win_length, mode='constant', cval=0)[::args.hop_length]
    feature['spectrogram_label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=samp_max_pool))
    
    samples_as_ints = (samples * 2**15).astype(np.int16)
    audio_segment = AudioSegment(
        samples_as_ints.tobytes(), 
        frame_rate=sr,
        sample_width=2, # samples_as_ints.dtype.itemsize
        channels=1
    )
    audio_feature = audio_segment.raw_data

    feature['audio'] = _bytes_feature(audio_feature)
    feature['audio_label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=samp_max_pool))

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

def _build_tf_records_from_dataset(p_files, p_start_ends, n_files, positive_multiplier, output_dir, output_negative_version, max_per_record, text, idx_offset, win_length, hop_length, example_length):
    """
    output_negative_version: if True then output an audio file without any wakeword also
    """
    global counter
    num_examples = 0
    next_record = 0
    record_writer = None

    os.makedirs(output_dir, exist_ok=True)

    for idx, p_file in enumerate(tqdm(p_files, desc='Positive Recordings Processed')):
        p_start_end = p_start_ends[idx]
        augment_settings = random.sample(positive_augment_opts, positive_multiplier)
        mix_settings = random.sample(room_sim_opts, positive_multiplier)
        for j in range(positive_multiplier):
            # sample_id = idx + idx_offset
            
            n_file = n_files[idx*positive_multiplier + j]
            
            rand_p_duration = lengths_ms[np.random.multinomial(1, lengths_probabilities).tolist().index(1)]
            
            try:
                output, labs = augment_audio(p_file, p_start_end, n_file, rand_p_duration, *augment_settings[j])
            except:
                print("Error augmenting inputs %s, %s:" % (p_file, n_file))
                print(sys.exc_info()[0])
                continue
            
            source1 = output[:,1]
            source2 = output[:,[0,2]].mean(axis=1)
            
            mix = mix_2_sources(source1, source2, *mix_settings[j])
            
            n_tot_chunks = example_length // hop_length
            n_left_chunks = random.randint(n_tot_chunks//4, 3*n_tot_chunks//4)
            n_right_chunks = n_tot_chunks - n_left_chunks - 1 # left/right exclude the center chunk
            samples = extract_example(mix, labs[1,1], example_length=example_length, n_right_chunks=n_right_chunks, n_left_chunks=n_left_chunks)
            
            mix_labs = np.zeros(mix.shape, dtype=int)
            mix_labs[labs[0,1]:labs[1,1]] = 1
            samples_labs = extract_example(mix_labs, labs[1,1], example_length=example_length, n_right_chunks=n_right_chunks, n_left_chunks=n_left_chunks)

            example = serialize_example(samples, samples_labs, win_length, hop_length, example_length)
            
            record_writer, next_record, num_examples = _update_record_writer(
                record_writer=record_writer, next_record=next_record,
                num_examples=num_examples, output_dir=output_dir,
                max_per_record=max_per_record, text=text)
            record_writer.write(example)
            if counter:
                with counter.get_lock():
                    counter.value += 1
            
            # add purely negative example also
            if output_negative_version:
                _, silence, _, loudness = augment_settings[j]
                output, labs = augment_audio_two_negatives(n_file, n_file, silence, loudness)
                mix = mix_2_sources(output[:,0], output[:,1], *mix_settings[j])
                samples = extract_example(mix, labs[0,1], example_length=example_length, n_right_chunks=n_right_chunks, n_left_chunks=n_left_chunks)
                samples_labs = np.zeros(example_length, dtype=int)
                
                example = serialize_example(samples, samples_labs, win_length, hop_length, example_length)
            
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

def _build_tf_records_from_phoneme_dataset(files, output_dir, max_per_record, text, idx_offset, win_length, hop_length, example_length, noise_type):
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
        
        mix_settings = random.sample(room_sim_opts, samples.shape[1])
        for sub_clip_i in range(samples.shape[1]):
            y = np.array(samples[:,sub_clip_i])
            if noise_type == "clean":
                mix = y
            elif noise_type in ("gwn", "GWN"):
                y = y / max(y.max(), -y.min())
                noise = gwn_for_audio(y, snr=np.random.normal(52,5))
                mix = mix_2_sources(y, noise, *mix_settings[sub_clip_i])
                mix = mix[:len(y)]
            else:
                raise ValueError("Unsupported noise_type %s" % noise_type)
            
            example = serialize_example(mix, phns[:,sub_clip_i], win_length, hop_length, example_length)
        
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

def build_dataset_randomized(args, p_files, n_files, path, output_negative_version=False):
    """Chunking stage.
    
    Chunked by number threads.
    """
    global _idx_offset, counter
    lock = Lock()
    counter = Value('i', 0)
    
    with open(args.wakeword_metafile) as json_file:
        metadata = json.load(json_file)

    pool = Pool(initializer=_init_pool, initargs=(lock, counter),
                processes=args.threads)
    args_list = []
    p_chunk_size = len(p_files) // args.threads
    n_files_pointer = 0
    for i in range(args.threads):
        p_files_chunk = p_files[i*p_chunk_size:((i+1)*p_chunk_size)]
        p_files_chunk_ = [os.path.join(*fname.split('/')[-2:]) for fname in p_files_chunk]
        p_start_ends = [metadata[fname] for fname in p_files_chunk_]
        n_chunk_size = len(p_files_chunk)*args.positive_multiplier
        
        args_list.append({'p_files': p_files_chunk,
                          'p_start_ends': p_start_ends,
                          'n_files': n_files[n_files_pointer:(n_files_pointer+n_chunk_size)],
                          'positive_multiplier': args.positive_multiplier,
                          'output_dir': path,
                          'output_negative_version': output_negative_version,
                          'max_per_record': args.max_per_record,
                          'text': '_clean_speech_%.2d' % i,
                          'idx_offset': _idx_offset,
                          'win_length': args.win_length,
                          'hop_length': args.hop_length,
                          'example_length': args.example_length
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
                          'noise_type': args.noise_type
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
    dirs = np.array(list(set([path.split('/')[0] for path in source_files])))
    idxs = list(range(len(dirs)))
    random.shuffle(idxs)
    
    n_train = int(percentage_train * len(idxs))
    train_dirs = dirs[idxs[:n_train]]
    val_dirs = dirs[idxs[n_train:]]
    
    train_files = []
    val_files = []
    
    for path in source_files:
        speaker = path.split('/')[0]
        full_path = os.path.join(args.positive_data_dir, path)
        if speaker in train_dirs:
            train_files.append(full_path)
        elif speaker in val_dirs:
            val_files.append(full_path)
        else:
            raise ValueError("%s is neither in train nor val set" % path)

    return train_files, val_files

def create_wakeword_dataset(args):
    n_train_files, n_val_files = train_val_speaker_separation(args.negative_data_dir, fglob='*/*.flac', percentage_train=args.percentage_train)
    p_train_files, p_val_files = train_val_speaker_separation_wakeword_metafile(args, percentage_train=args.percentage_train)
    
    assert len(p_train_files) * args.positive_multiplier < len(n_train_files), 'Not enough negative recordings'
    assert len(p_val_files) * args.positive_multiplier < len(n_val_files), 'Not enough negative recordings'
    
    random.shuffle(n_train_files)
    random.shuffle(n_val_files)
    
    if args.train_path is not None:
        build_dataset_randomized(args,
                                p_files=p_train_files,
                                n_files=n_train_files,
                                path=args.train_path)
    if args.val_path is not None:
        build_dataset_randomized(args,
                                p_files=p_val_files,
                                n_files=n_val_files,
                                path=args.val_path,
                                output_negative_version=True)

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
