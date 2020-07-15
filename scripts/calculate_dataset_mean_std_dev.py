"""
Example Usage:
python scripts/calculate_dataset_mean_std_dev.py --tfrecord_glob='/scratch1/ilya/locDoc/data/alexa/v7.3/train_79840/*tfrecord'
"""
import argparse
import glob
import os

import numpy as np
import tensorflow as tf

from tqdm import tqdm

spec_h = 257 # expected

positive_examples = []
negative_examples = []
n_to_read = 1000
class LimitExceded(Exception): pass
running_min = 1
running_max = 0

def mean_std_dev_tfrecords(tfrecord_files):
    n = 0
    S = 0.0
    m = 0.0
    
    for tfrecord_file in tqdm(tfrecord_files):
        for example in tf.python_io.tf_record_iterator(tfrecord_file):
            eg_np = tf.train.Example.FromString(example)
            x = eg_np.features.feature["spectrogram"].float_list.value
            for x_i in x:
                n = n + 1
                m_prev = m
                m = m + (x_i - m) / n
                S = S + (x_i - m) * (x_i - m_prev)
    return {'mean': m, 'std': np.sqrt(S/n)}
    
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '--tfrecord_glob', type=str, default='/scratch1/ilya/locDoc/data/alexa/v7.4/val_19680/*tfrecord',
        help='Where to find the input')

    args = parser.parse_args()
    
    files = glob.glob(args.tfrecord_glob)
    
    stats = mean_std_dev_tfrecords(files)
    
    print(stats)
    
    tfrecord_dir = os.path.dirname(os.path.abspath(args.tfrecord_glob))
    savefile = os.path.join(tfrecord_dir, "stats.txt")
    print('Wrote stats to %s' % savefile)
    with open(savefile, "w") as f:
        f.write(str(stats))

if __name__ == '__main__':
    main()