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

import numpy as np
import librosa.core
import librosa.effects
from pydub import AudioSegment
from pyrubberband import pyrb
import soundfile as sf
from tqdm import tqdm

import tensorflow as tf

from util.ft import next_power_of_2, closest_power_of_2
from util.log import write_metadata
from augment_audio import augment_audio, extract_example, augment_audio_two_negatives

import pdb

counter = None
random.seed(2020)

sr = 16000
pitch_shift_opts = np.arange(-3,3.5,0.5).tolist()
silence_1_opts = np.arange(-0.1, 0.2, 0.05).tolist()
silence_2_opts = np.arange(-0.1, 0.2, 0.05).tolist()
loudness_opts = [0.25,0.5,1.0]
positive_augment_opts = list(product(*[pitch_shift_opts, silence_1_opts, silence_2_opts, loudness_opts]))

lengths_ms = np.arange(0.4,0.9,0.05)
lengths_probabilities = np.array([ 2,  7, 31, 20, 24, 21, 13,  4,  0])
lengths_probabilities = lengths_probabilities / lengths_probabilities.sum()

negative_silence_opts = np.arange(-0.1, 0.2, 0.05).tolist()
loudness_opts = [0.25,0.5,1.0,1.25, 1.5]
negative_augment_opts = list(product(*[negative_silence_opts, loudness_opts]))


parser = argparse.ArgumentParser()

parser.add_argument("--positive_data_dir", help="...")
parser.add_argument("--wakeword_metafile", help="...")
parser.add_argument("--negative_data_dir", help="...")
parser.add_argument('--positive_multiplier', default=1, type=int,
                    help='...')
parser.add_argument('--max_per_record', default=1000, type=int,
                    help='...')
parser.add_argument("--train_path", help="...")
parser.add_argument("--val_path", help="...")
parser.add_argument('--percentage_train', default=0.8, type=float,
                    help='...')
parser.add_argument('--win_length', default=sr//40, type=int,
                    help='...')
parser.add_argument('--hop_length', default=sr//100, type=int,
                    help='... Becomes the frame size (One frame per this many audio samples).')
parser.add_argument('--example_length', default=19840, type=int,
                    help='... Should be a multiple of hop_length')


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
    assert example_length % hop_length == 0, 'Example length should be a multiple of hop_length'
    spec = np.abs(librosa.core.stft(samples,
        win_length=win_length,
        hop_length=hop_length,
        n_fft=win_length))
    height = win_length // 2
    width = example_length // hop_length
    spec = spec[:height,:width]
    
    feature['spectrogram_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(spec.shape)))
    feature['spectrogram'] = tf.train.Feature(float_list=tf.train.FloatList(value=spec.reshape(-1)))
    # TODO make into spectrogram label (downsample)
    feature['spectrogram_label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=samples_label))
    
    samples_as_ints = (samples * 2**15).astype(np.int16)
    audio_segment = AudioSegment(
        samples_as_ints.tobytes(), 
        frame_rate=sr,
        sample_width=2, # samples_as_ints.dtype.itemsize
        channels=1
    )
    audio_feature = audio_segment.raw_data

    feature['audio'] = _bytes_feature(audio_feature)

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

def _build_tf_records_from_dataset(p_files, p_start_ends, n_files, positive_multiplier, output_dir,
                                   label, max_per_record, text, idx_offset, win_length, hop_length, example_length):
    global counter, counter2
    num_examples = 0
    next_record = 0
    record_writer = None

    os.makedirs(output_dir, exist_ok=True)

    if label == 1:
        for idx, p_file in enumerate(tqdm(p_files, desc='Positive Examples Processed')):
            p_start_end = p_start_ends[idx]
            augment_settings = random.sample(positive_augment_opts, positive_multiplier)
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
                
                mix = output.mean(axis=1)
                
                # TODO abstract away this option too, which is num segments to extract from a single mix
                # TODO when the wakeword is delayed/advanced advanced too much, output examples with label 0?
                for k, advance in enumerate((np.arange(0.05,rand_p_duration-0.05,0.01) * sr).astype(int).tolist()):
                        samples = extract_example(mix, labs[1,1]-advance, example_length=example_length)
                        
                        mix_labs = np.zeros(mix.shape, dtype=int)
                        mix_labs[labs[0,1]:labs[1,1]] = 1
                        samples_labs = extract_example(mix_labs, labs[1,1]-advance, example_length=example_length)
            
                        example = serialize_example(samples, samples_labs, win_length, hop_length, example_length)
                        
                        record_writer, next_record, num_examples = _update_record_writer(
                            record_writer=record_writer, next_record=next_record,
                            num_examples=num_examples, output_dir=output_dir,
                            max_per_record=max_per_record, text=text)
                        record_writer.write(example)
                        with counter.get_lock():
                            counter.value += 1
                        


    else:
        for idx in tqdm(range(len(n_files)//2), desc='Negative Examples Processed'):
            augment_setting = random.sample(negative_augment_opts, 1)[0]
            rand_p_duration = lengths_ms[np.random.multinomial(1, lengths_probabilities).tolist().index(1)]

            try:
                output, labs = augment_audio_two_negatives(n_files[2*idx], n_files[2*idx + 1], example_length, *augment_setting)
            except:
                print("Error augmenting inputs %s, %s:" % (n_files[2*idx], n_files[2*idx + 1]))
                print(sys.exc_info()[0])
                continue
            
            mix = output.mean(axis=1)
            # sample_id = idx + idx_offset
            # do the same kind of multiplying in negative set for balance
            for k, advance in enumerate((np.arange(0.05,rand_p_duration-0.05,0.01) * sr).astype(int).tolist()):
                samples = extract_example(mix, labs[1,0]+int(0.45*sr)-advance, example_length=example_length)
                
                samples_labs = np.zeros(samples.shape, dtype=int)
                example = serialize_example(samples, samples_labs, win_length, hop_length, example_length)
                
                record_writer, next_record, num_examples = _update_record_writer(
                    record_writer=record_writer, next_record=next_record,
                    num_examples=num_examples, output_dir=output_dir,
                    max_per_record=max_per_record, text=text)
                record_writer.write(example)
                with counter.get_lock():
                    counter2.value += 1
            
    
    record_writer.close()


def _init_pool(l, c, c2):
    global lock, counter, counter2
    lock = l
    counter = c
    counter2 = c2
    
_idx_offset = 0

def build_dataset_randomized(args, p_files, n_files, path):
    """Chunking stage.
    
    Chunked by number threads.
    """
    global _idx_offset
    lock = Lock()
    counter = Value('i', 0)
    counter2 = Value('i', 0)
    
    with open(args.wakeword_metafile) as json_file:
        metadata = json.load(json_file)

    pool = Pool(initializer=_init_pool, initargs=(lock, counter, counter2),
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
                          'label': 1,
                          'max_per_record': args.max_per_record,
                          'text': '_positive_%.2d' % i,
                          'idx_offset': _idx_offset,
                          'win_length': args.win_length,
                          'hop_length': args.hop_length,
                          'example_length': args.example_length
        })
        n_files_pointer += n_chunk_size
    _idx_offset += len(p_files)*args.positive_multiplier
    
    for i in range(args.threads):
        # 50/50 wakeword to not wakeword for now
        n_chunk_size = 2*len(p_files)*args.positive_multiplier // args.threads
        
        args_list.append({'p_files': None,
                          'p_start_ends': None,
                          'n_files': n_files[n_files_pointer:(n_files_pointer+n_chunk_size)],
                          'positive_multiplier': args.positive_multiplier,
                          'output_dir': path,
                          'label': 0,
                          'max_per_record': args.max_per_record,
                          'text': '_negative_%.2d' % i,
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
    print('Made %i positive spectrograms.' % counter.value)
    print('Made %i negative spectrograms.' % counter2.value)

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

if __name__ == '__main__':
    args = parser.parse_args()

    write_metadata(args.train_path, args)

    n_train_files, n_val_files = train_val_speaker_separation(args.negative_data_dir, fglob='*/*.flac', percentage_train=args.percentage_train)
    p_train_files, p_val_files = train_val_speaker_separation_wakeword_metafile(args, percentage_train=args.percentage_train)
    
    assert len(p_train_files) * args.positive_multiplier < len(n_train_files), 'Not enough negative examples'
    assert len(p_val_files) * args.positive_multiplier < len(n_val_files), 'Not enough negative examples'
    
    random.shuffle(n_train_files)
    random.shuffle(n_val_files)
    
    build_dataset_randomized(args,
                            p_files=p_train_files,
                            n_files=n_train_files,
                            path=args.train_path)
    build_dataset_randomized(args,
                            p_files=p_val_files,
                            n_files=n_val_files,
                            path=args.val_path)
