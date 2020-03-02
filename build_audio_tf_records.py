"""
Builds a TFRecord dataset.
"""

import argparse
import glob
from itertools import product
import io
import os
import random
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

import pdb

AUDIO_OUTPUT_LEN = 32768 # 2s, power of 2

sr = 16000
time_stretch_opts = np.arange(0.8,1.25,.05).tolist()
pitch_shift_opts = np.arange(-3,3.5,0.5).tolist()
b_advance_opts = np.arange(0, sr//2, sr//20, dtype=int).tolist()
c_advance_opts = np.arange(0, sr//4, sr//20, dtype=int).tolist()
output_advance_opts = np.arange(sr//10, sr, sr//10, dtype=int).tolist()
loudness_opts = [0.25,0.5,1.0]
augment_opts = list(product(*[time_stretch_opts, pitch_shift_opts, b_advance_opts, c_advance_opts, output_advance_opts, loudness_opts]))

b_advance_opts = np.arange(0, sr//2, sr//20, dtype=int).tolist()
output_advance_opts = np.arange(sr//10, sr, sr//10, dtype=int).tolist()
loudness_opts = [0.25,0.5,1.0,1.25, 1.5]
negative_augment_opts = list(product(*[b_advance_opts, output_advance_opts, loudness_opts]))


parser = argparse.ArgumentParser()

parser.add_argument("--positive_data_dir", help="...")
parser.add_argument("--negative_data_dir", help="...")
parser.add_argument('--positive_multiplier', default=1, type=int,
                    help='...')
parser.add_argument('--max_per_record', default=1000, type=int,
                    help='...')
parser.add_argument("--train_path", help="...")
parser.add_argument("--val_path", help="...")
parser.add_argument('--percentage_train', default=0.8, type=float,
                    help='...')


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


def serialize_example(samples, label, sample_id=None):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    feature = {}
    
    feature['label'] = _int64_feature(label)

    if sample_id is not None:
        feature['id'] = _int64_feature(sample_id)
    
    assert len(samples) == AUDIO_OUTPUT_LEN, 'Audio is length %i expected %i' % (len(samples), AUDIO_OUTPUT_LEN)
    spec = np.abs(librosa.core.stft(samples,
        win_length=closest_power_of_2(sr//50),
        hop_length=closest_power_of_2(sr // 100),
        n_fft=2*closest_power_of_2(sr//50)))
    spec = spec[:256,:256]
    
    feature['spectrogram_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(spec.shape)))
    feature['spectrogram'] = tf.train.Feature(float_list=tf.train.FloatList(value=spec.reshape(-1)))
    
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

def _build_tf_records_from_dataset(p_files, n_files, positive_multiplier, output_dir,
                                   label, max_per_record, text, idx_offset):
    num_examples = 0
    next_record = 0
    record_writer = None

    os.makedirs(output_dir, exist_ok=True)

    if label == 1:
        for idx, p_file in enumerate(tqdm(p_files, desc='Positive Examples Processed')):
            augment_settings = random.sample(augment_opts, positive_multiplier)
            for j in range(positive_multiplier):
                record_writer, next_record, num_examples = _update_record_writer(
                    record_writer=record_writer, next_record=next_record,
                    num_examples=num_examples, output_dir=output_dir,
                    max_per_record=max_per_record, text=text)
    
                sample_id = idx + idx_offset
                
                n_file = n_files[idx*positive_multiplier + j]
                
                samples = augment_audio(p_file, n_file, *augment_settings[j])
    
                example = serialize_example(samples, label, sample_id=sample_id)
                record_writer.write(example)
    else:
        for idx in tqdm(range(len(n_files)//2), desc='Negative Examples Processed'):
            augment_setting = random.sample(negative_augment_opts, 1)[0]
            record_writer, next_record, num_examples = _update_record_writer(
                record_writer=record_writer, next_record=next_record,
                num_examples=num_examples, output_dir=output_dir,
                max_per_record=max_per_record, text=text)

            sample_id = idx + idx_offset
            
            samples = mix_two_negatives(n_files[2*idx], n_files[2*idx + 1], *augment_setting)
            example = serialize_example(samples, label, sample_id=sample_id)
            record_writer.write(example)
            
    
    record_writer.close()

def mix_two(a_,b, b_advance=0):
    a = np.copy(a_)
    if len(b) <= b_advance:
        a[-b_advance:(len(b)-b_advance)] += b
        return a
    else:
        if b_advance > 0:
            a[-b_advance:] += b[:b_advance]
            return np.concatenate([a, b[b_advance:]], axis=0)
        else:
            return np.concatenate([a, b], axis=0)
    

def mix_three(a, b, c, b_advance=320, c_advance=320, output_advance=0, output_len=32000):
    """
     b_advance
          <---
         ^ default output start is whenever B starts
    |---A-----|---B---|----C-----|
                   <--
             c_advance^
    output_advance advances the default output start.
    """
    ab = mix_two(a,b,b_advance)
    abc = mix_two(ab,c,c_advance)
    s = max(len(a)-b_advance - output_advance, 0)
    e = s + output_len
    out = np.zeros(output_len)
    actual_len = len(abc)-s
    out[:actual_len] = abc[s:e]
    return out

def get_chunk(samples, chunk):
    return np.array(samples[chunk[0]:chunk[1]])

def augment_audio(p_file, n_file, time_stretch, pitch_shifts, b_advance, c_advance, output_advance, loudness):
    sr = 16000
    output_len = AUDIO_OUTPUT_LEN
    samples_bg, _ = sf.read(n_file)
    
    samples_bg, _ = librosa.effects.trim(samples_bg, top_db=60)    
    leading_chunk, trailing_chunk = [[len(samples_bg)//2, len(samples_bg)-1],[0,len(samples_bg)//2]]
    leading = get_chunk(samples_bg, leading_chunk)
    trailing = get_chunk(samples_bg, trailing_chunk)
        
    wakeword, _ = sf.read(p_file)
    wakeword_timealtered = pyrb.time_stretch(wakeword, sr=sr, rate=time_stretch)
    wakeword_pitchaltered = pyrb.pitch_shift(wakeword_timealtered, sr=sr, n_steps=pitch_shifts)
    wakeword_pitchaltered_, _ = librosa.effects.trim(wakeword_pitchaltered, top_db=30)
    k = np.sqrt((trailing**2).sum() / len(trailing)) / np.sqrt((wakeword_pitchaltered_**2).sum() / len(wakeword_pitchaltered_))
    wakeword_pitchaltered_ *= k*loudness
    return mix_three(leading, np.array(wakeword_pitchaltered_), trailing, b_advance=b_advance, c_advance=c_advance, output_advance=output_advance, output_len=output_len)
    
def mix_two_negatives(n_file1, n_file2, b_advance, output_advance, loudness):
    sr = 16000
    output_len = AUDIO_OUTPUT_LEN
    a, _ = sf.read(n_file1)
    b, _ = sf.read(n_file2)
    
    a, _ = librosa.effects.trim(a, top_db=40)
    b, _ = librosa.effects.trim(b, top_db=40)
    leading_chunk, trailing_chunk = [[len(a)//2, len(a)-1],[0,len(b)//2]]
    leading = get_chunk(a, leading_chunk)
    trailing = get_chunk(b, trailing_chunk)

    k = np.sqrt((leading**2).sum() / len(leading)) / np.sqrt((trailing**2).sum() / len(trailing))
    trailing *= k*loudness
    mixed = mix_two(leading, trailing, b_advance=b_advance)
    
    s = max(len(leading)-b_advance - output_advance, 0)
    e = s + output_len
    out = np.zeros(output_len)
    actual_len = len(mixed)-s
    out[:actual_len] = mixed[s:e]
    return out
    


def _init_pool(l, c):
    global lock, counter
    lock = l
    counter = c
    
_idx_offset = 0

def build_dataset_randomized(args, p_files, n_files, path):
    """Chunking stage.
    
    Chunked by number threads.
    """
    global _idx_offset
    lock = Lock()
    counter = Value('i', 0)

    pool = Pool(initializer=_init_pool, initargs=(lock, counter),
                processes=args.threads)
    args_list = []
    p_chunk_size = len(p_files) // args.threads
    n_files_pointer = 0
    for i in range(args.threads):
        p_files_chunk = p_files[i*p_chunk_size:((i+1)*p_chunk_size)]
        n_chunk_size = len(p_files_chunk)*args.positive_multiplier
        
        args_list.append({'p_files': p_files_chunk,
                          'n_files': n_files[n_files_pointer:(n_files_pointer+n_chunk_size)],
                          'positive_multiplier': args.positive_multiplier,
                          'output_dir': path,
                          'label': 1,
                          'max_per_record': args.max_per_record,
                          'text': '_positive_%.2d' % i,
                          'idx_offset': _idx_offset
        })
        n_files_pointer += n_chunk_size
    _idx_offset += len(p_files)*args.positive_multiplier
    
    for i in range(args.threads):
        # 50/50 wakeword to not wakeword for now
        n_chunk_size = 2*len(p_files)*args.positive_multiplier // args.threads
        
        args_list.append({'p_files': None,
                          'n_files': n_files[n_files_pointer:(n_files_pointer+n_chunk_size)],
                          'positive_multiplier': args.positive_multiplier,
                          'output_dir': path,
                          'label': 0,
                          'max_per_record': args.max_per_record,
                          'text': '_negative_%.2d' % i,
                          'idx_offset': _idx_offset
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

if __name__ == '__main__':
    args = parser.parse_args()

    n_train_files, n_val_files = train_val_speaker_separation(args.negative_data_dir, fglob='*/*.flac', percentage_train=args.percentage_train)
    p_train_files, p_val_files = train_val_speaker_separation(args.positive_data_dir, fglob='*.wav', percentage_train=args.percentage_train)
    
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
