"""
Example Usage:
python scripts/demand.py --nthreads=8
"""
import argparse
import glob
import os
import numpy.random
import math
import multiprocessing as mp
from multiprocessing import Lock, Value

from tqdm import tqdm 

import pdb

# precut deman dataset

def run_cmd_line(cmd):
    global pbar, counter
    if counter:
        with counter.get_lock():
            counter.value += 1
        pbar.n = counter.value
        pbar.refresh()
    
    return os.system(cmd)

def precut_demand_dataset(args):
    """Outputs clips of desired length
    Organized as 5 min clips per environment with 16 channels.
    Because of this redundancy it is fair to stagger the clips randomly.
    """
    global pbar, counter
    counter = Value('i', 0)
    
    sr = args.sr
    categories = ['DKITCHEN', 'DLIVING', 'DWASHING', 'OHALLWAY', 'OOFFICE', 'PCAFETER', 'SCAFE']
    # categories = ['DKITCHEN']
    # demand dataset is 300s
    n_outputs = (sr * 300) // args.example_length
    order_of_outputs = int(math.log10(n_outputs))+1
    
    eg_len_s = round(args.example_length / float(sr), 3) + 0.01
    
    ffmpeg_cmds = []
    for category in categories:
        channel_files = glob.glob(os.path.join(args.data_root, category, '*.wav'))
        out_dir = os.path.join(args.output_dir, category)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for channel_full_path in channel_files:
            channel_name = channel_full_path.split('/')[-1].split('.')[0]
            for i in range(n_outputs):
                # schedule
                output_file = '{}_part{}.wav'.format(channel_name, str(i).zfill(order_of_outputs))
                output_full_path = os.path.join(out_dir, output_file)
                start_samples = i*args.example_length + numpy.random.uniform(-args.example_length//4, args.example_length//4)
                start_sec = round(start_samples / sr, 3)
                ffmpeg_cmds.append('ffmpeg -y -hide_banner -loglevel panic -ss {} -i {} -t {} -acodec copy {}'.format(start_sec, channel_full_path, eg_len_s, output_full_path))
    
    pbar = tqdm(total=len(ffmpeg_cmds))

    with mp.Pool(processes = args.nthreads) as p:
        res = p.map(run_cmd_line, ffmpeg_cmds)
        
    pbar.n = len(ffmpeg_cmds)
    pbar.close()

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--example_length', default=80000, type=int,
                        help='...')
    parser.add_argument('--sr', default=16000, type=int,
                        help='...')
    parser.add_argument('--nthreads', default=1, type=int,
                        help='...')
    parser.add_argument(
        '--data_root', type=str, default='/scratch0/ilya/locDoc/data/demand',
        help='Where to find the input')
    parser.add_argument(
        '--output_dir', type=str, default='/scratch1/ilya/locDoc/data/demand_cut/80000',
        help='Where to find the output')
            
    args = parser.parse_args()
    
    precut_demand_dataset(args)

if __name__ == '__main__':
    main()