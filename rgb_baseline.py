"""ST-svm for rgb images

used for Simmyride.
"""
from collections import namedtuple
import itertools
import glob
import pickle
import os
import time

import h5py
# import hdf5storage
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import scipy.io as sio
from sklearn.metrics import confusion_matrix

from lib.libsvm.python.svmutil import *
import windows as win
from rgb_pixelNN import scat2d_to_2d_2layer
from salt_data import split_trainval
from rle import myrlestring

import pdb


def run_baseline(trainfilename=None, labelfilename=None, outfilename=None):
    """
    Args: data, image in (height, width, nbands) format
    """
    [height, width, nbands] = [2017,2017,3]
    nlabels = 6
    ap = np.array([16,16])

    all_pixels_images_train = np.array(list(itertools.product(range(width),range(height))))

    assert np.all(ap % 2 == 0), 'Assymetric is not supported'
    net_in_shape = ap + np.array([height,width])
    x = tf.placeholder(tf.float32, shape=[1,net_in_shape[0], net_in_shape[0], nbands])
    feat = scat2d_to_2d_2layer(x, bs=1)
    feat_size = feat.shape[3]
    print('feat size is %d' % feat_size)

    print('requesting %d MB memory' % (all_pixels_images_train.shape[0] * feat_size * 4 / 1000000.0))
    pix_feat = np.zeros((all_pixels_images_train.shape[0], feat_size), dtype=np.float32)
    # flat_labels_train = np.zeros((all_pixels_images_train.shape[0],), dtype=int)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    def compute_features():
        im=Image.open(trainfilename)
        npim = np.array(im, dtype=np.float32) / 255.0
        image_padded = np.pad(npim, ((ap[0]/2,ap[0]/2),(ap[1]/2,ap[1]/2),(0,0)), 'reflect')

        subimg = np.expand_dims(image_padded,0)

        feed_dict = {x: subimg}
        net_out = sess.run(feat, feed_dict).reshape((height*width, feat_size))
        pix_feat[:(height*width),:] = net_out

    
    flat_labels = np.load(labelfilename).reshape(height*width)
    
    print('computing train feat')
    compute_features()

    OUT_PATH = '/Users/artsyinc/Documents/simmyride/data/materials/first'
    if outfilename:
        outfilename = os.path.join(OUT_PATH, outfilename)
    else:
        outfilename = os.path.join(OUT_PATH, 'scat_expt')


    print('training now')
    start = time.time()
    prob  = svm_problem(flat_labels[flat_labels != 0].tolist(), pix_feat[flat_labels != 0,:].tolist())
    param = svm_parameter('-s 0 -t 0 -q')
    m = svm_train(prob, param)
    end = time.time()
    print(end - start)

    # pdb.set_trace()

    # with open(outfilename+'.pkl', 'wb') as output:
    #     pickle.dump(m, output, pickle.HIGHEST_PROTOCOL)

    # now test
    pred_labels = flat_labels
    test_chunk_size = 1000

    for i in tqdm(range(0,len(pred_labels),test_chunk_size)):
        p_label, p_acc, p_val = svm_predict(pred_labels[i:i+test_chunk_size].tolist(), pix_feat[i:i+test_chunk_size,:].tolist(), m, '-q');
        pred_labels[i:i+test_chunk_size] = p_label

    np.save(outfilename, pred_labels.reshape(height,width))

import matplotlib.pyplot as plt
def myplot():
    arr = np.load('/Users/artsyinc/Documents/simmyride/data/materials/second/resized/svm.npy')
    # npim = arr.reshape(2017,2017)
    plt.imshow(npim)
    plt.show()
    pdb.set_trace()

def clean_glob(myglob):
    outfiles = []
    for filepath in myglob:
        path, filename = os.path.split(filepath)
        outfiles.append(filename)
    return outfiles

def main():
    # trainpaths, _ = split_trainval(0.87)
    # trainfiles = clean_glob(trainpaths)
    # valfiles = ['1bd1c8c771.png', '01c033e116.png', '1c6237ae58.png', '1d0c2fa004.png']
    testpaths = glob.glob('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/test/images/*.png')
    testfiles = clean_glob(testpaths)

    run_baseline(
        '/Users/artsyinc/Documents/simmyride/data/materials/second/resized/image.png',
        '/Users/artsyinc/Documents/simmyride/data/materials/second/resized/labels.npy',
        '/Users/artsyinc/Documents/simmyride/data/materials/second/resized/svm2')

if __name__ == '__main__':
    main()
