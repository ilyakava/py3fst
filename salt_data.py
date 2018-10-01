import glob
import os
from shutil import copyfile

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import pdb

def split_trainval(val_size=0.10):
    myhist = {
        0: [],
        600: [],
        2500: [],
        5000: [],
        8000: [],
        10201: []
    }
    keys = np.array(sorted(myhist.keys()))
    for filename in glob.glob('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/train/masks/*.png'): #assuming gif
        im=Image.open(filename).convert('L')
        npim = np.array(im, dtype=np.float32)
        num_nonzero_pix = np.sum(np.array(im, dtype=np.float32) > 0)
        hist_bin = keys[np.sum(keys < num_nonzero_pix)]
        myhist[hist_bin].append(filename)
        im.close()
    train = []
    val = []
    for mybin in keys:
        bin_train, bin_val =  train_test_split(myhist[mybin], test_size=val_size, random_state=42)
        train += bin_train
        val += bin_val
    return [train, val]

def copy_files(train, val):
    dataset_path = '/scratch0/ilya/locDoc/data/kaggle-seismic-dataset'
    sX = '%s/train/images' % dataset_path # source
    tX = '%s/mytrain/images' % dataset_path
    vX = '%s/myval/images' % dataset_path
    sY = '%s/train/masks' % dataset_path # source
    tY = '%s/mytrain/masks' % dataset_path
    vY = '%s/myval/masks' % dataset_path

    for example in train:
        path, filename = os.path.split(example)
        copyfile('%s/%s' % (sX, filename), '%s/%s' % (tX, filename))
        copyfile('%s/%s' % (sY, filename), '%s/%s' % (tY, filename))
    for example in val:
        path, filename = os.path.split(example)
        copyfile('%s/%s' % (sX, filename), '%s/%s' % (vX, filename))
        copyfile('%s/%s' % (sY, filename), '%s/%s' % (vY, filename))

        

def salt_present_labels(folder='train'):
    out_list = []
    for filename in glob.glob('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/%s/masks/*.png' % folder): #assuming gif
        im=Image.open(filename).convert('L')
        npim = np.array(im, dtype=np.float32)
        out_list.append(int(np.any(np.array(im, dtype=np.float32) > 0)))
        im.close()
    out_list = np.array(out_list)
    return out_list


def salt_pixel_num(folder='mytrain'):
    out_list = []
    for filename in glob.glob('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/%s/masks/*.png' % folder): #assuming gif
        im=Image.open(filename).convert('L')
        npim = np.array(im, dtype=np.float32)
        out_list.append(np.sum(np.array(im, dtype=np.float32) > 0))
        im.close()
    out_list = np.array(out_list)
    return out_list

def salt_pixel_hist():
    a = salt_pixel_num()
    plt.hist(a, bins='auto')  # arguments are passed to np.histogram
    plt.show()



if __name__ == '__main__':
    split_trainval()