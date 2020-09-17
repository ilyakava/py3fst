"""
Example Usage:
python scripts/calculate_dataset_mean_std_dev.py --tfrecord_glob=/scratch1/ilya/locDoc/data/alexa/v7.4/val_19680/*tfrecord

Or pass in number of channels to split the input features when computing stats

python scripts/calculate_dataset_mean_std_dev.py --tfrecord_num_channels=2 --tfrecord_glob=/scratch1/ilya/locDoc/data/alexa/v7.8/val_19680/*tfrecord
"""
import argparse
import glob
import json
import os
from time import time

import numpy as np
import tensorflow as tf

from tqdm import tqdm

def mean_std_dev_tfrecords2(tfrecord_files):
    """
    Not faster than the numpy version below.
    """
    num_examples = 0
    n = 0
    S = 0.0
    m = 0.0
    
    for tfrecord_file in tqdm(tfrecord_files):
        for example in tf.python_io.tf_record_iterator(tfrecord_file):
            num_examples += 1
            eg = tf.train.Example.FromString(example)
            x = eg.features.feature["spectrogram"].float_list.value
            for x_i in x:
                n = n + 1
                m_prev = m
                m = m + (x_i - m) / n
                S = S + (x_i - m) * (x_i - m_prev)
    print('Finished processing %i examples' % num_examples)
    return {'mean': m, 'std': np.sqrt(S/n)}
    
def mean_std_dev_tfrecords(tfrecord_files):
    num_examples = 0
    S_v = []
    m_v = []
    
    for tfrecord_file in tqdm(tfrecord_files):
        for example in tf.python_io.tf_record_iterator(tfrecord_file):
            num_examples += 1
            eg = tf.train.Example.FromString(example)
            x = eg.features.feature["spectrogram"].float_list.value
            x = np.array(x)
            m = x.mean()
            S = x.var()
            
            S_v.append(S)
            m_v.append(m)
    print('Finished processing %i examples' % num_examples)
    S_v = np.array(S_v)
    m_v = np.array(m_v)
    return {'mean': m_v.mean(), 'std': np.sqrt(S_v.mean())}
    
def mean_std_dev_vector_tfrecords(tfrecord_files, nc):
    assert nc > 0, 'Expected number channels to be a positive number'
    num_examples = 0
    S_v = []
    m_v = []
    
    for tfrecord_file in tqdm(tfrecord_files):
        for example in tf.python_io.tf_record_iterator(tfrecord_file):
            
            num_examples += 1
            eg = tf.train.Example.FromString(example)
            x = eg.features.feature["spectrogram"].float_list.value
            # the features are saved row-wise so they can be split while flat
            x = np.split(np.array(x),nc)
            m = [chan.mean() for chan in x]
            S = [chan.var() for chan in x]
            
            S_v.append(S)
            m_v.append(m)
                
    S_v = np.array(S_v)
    m_v = np.array(m_v)
    print('Finished processing %i examples' % num_examples)
    return {'mean': m_v.mean(axis=0), 'std': np.sqrt(S_v.mean(axis=0))}
    
def pretty_print(stats):
    for k, v in stats.items():
        print(k + ': {}'.format(v, ' .8f'))
    
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--tfrecord_num_channels', default=1, type=int,
                        help='Include this arg if each example is a concatenation of features. Current convention for multiple channels is that features are concatenated along height')
    parser.add_argument(
        '--tfrecord_glob', type=str, default='/scratch1/ilya/locDoc/data/alexa/v7.4/val_19680/*tfrecord',
        help='Where to find the input')

    args = parser.parse_args()
    
    files = sorted(glob.glob(args.tfrecord_glob))
    
    if args.tfrecord_num_channels == 1:
        stats = mean_std_dev_tfrecords(files)
    else:
        stats = mean_std_dev_vector_tfrecords(files, args.tfrecord_num_channels)
    pretty_print(stats)
    
    tfrecord_dir = os.path.dirname(os.path.abspath(args.tfrecord_glob))
    savefile = os.path.join(tfrecord_dir, "stats.txt")
    print('Wrote stats to %s' % savefile)
    with open(savefile, "w") as f:
        f.write(str(stats))

if __name__ == '__main__':
    main()