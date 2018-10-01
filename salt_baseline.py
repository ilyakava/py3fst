from collections import namedtuple
import itertools
import glob
import pickle
import os
import time

import h5py
import hdf5storage
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import scipy.io as sio
from sklearn.metrics import confusion_matrix

from lib.libsvm.python.svmutil import *
import windows as win
from salt_binary import scat2d_to_2d_2layer
from salt_data import split_trainval
from rle import myrlestring

import pdb


def run_baseline(model_fn, traindir, testdir, trainfilenames=None, testfilenames=None, outfilename=None):
    """
    Args: data, image in (height, width, nbands) format
    """
    [height, width, nbands] = [101,101,1]
    nlabels = 2
    all_pixels_images_train = np.array(list(itertools.product(trainfilenames, range(width),range(height))))
    all_pixels_images_test = np.array(list(itertools.product(testfilenames, range(width),range(height))))

    ap = np.array([16,16])
    assert np.all(ap % 2 == 0), 'Assymetric is not supported'
    net_in_shape = ap + np.array([101,101])
    x = tf.placeholder(tf.float32, shape=[1,net_in_shape[0], net_in_shape[0], nbands])
    feat = model_fn(x)
    feat_size = feat.shape[3]
    print('feat size is %d' % feat_size)

    print('requesting %d MB memory' % (all_pixels_images_train.shape[0] * feat_size * 4 / 1000000.0))
    train_pix_feat = np.zeros((all_pixels_images_train.shape[0], feat_size), dtype=np.float32)
    flat_labels_train = np.zeros((all_pixels_images_train.shape[0],), dtype=int)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    def compute_features(filedir, filenames, output_arr):

        for file_i, image_name in enumerate(tqdm(filenames)):
            filename = os.path.join(filedir, 'images', image_name)
            im=Image.open(filename).convert('L')
            npim = np.array(im, dtype=np.float32) / 255.0
            image_padded = np.pad(npim, ((ap[0]/2,ap[0]/2),(ap[1]/2,ap[1]/2)), 'reflect')

            subimg = np.expand_dims(image_padded,0)
            subimg = np.expand_dims(subimg,-1)

            feed_dict = {x: subimg}
            net_out = sess.run(feat, feed_dict).reshape((height*width, feat_size))
            output_arr[(file_i*height*width):((file_i+1)*height*width),:] = net_out

    def get_labels(filedir, filenames, output_arr):
        for img_idx, image_name in enumerate(tqdm(filenames)):
            filename = os.path.join(filedir, 'masks', image_name)
            im=Image.open(filename).convert('L')
            labs = np.array(np.array(im, dtype=np.float32) / 255.0, dtype=int).reshape(height*width)
            idx = img_idx*height*width
            output_arr[idx:(idx + height*width)] = labs

    
    print('getting training data')
    get_labels(traindir, trainfilenames, flat_labels_train)
    # print('getting test data')
    # flat_labels_test = np.zeros((all_pixels_images_test.shape[0],), dtype=int)
    # get_labels(testdir, testfilenames, flat_labels_test)

    
    print('computing train feat')
    compute_features(traindir, trainfilenames, train_pix_feat)

    OUT_PATH = '/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/baseline'
    if outfilename:
        outfilename = os.path.join(OUT_PATH, outfilename)
    else:
        outfilename = os.path.join(OUT_PATH, 'salt_expt')

    print('training now')
    start = time.time()
    prob  = svm_problem(flat_labels_train.tolist(), train_pix_feat.tolist())
    param = svm_parameter('-s 0 -t 0 -q')
    m = svm_train(prob, param)
    end = time.time()
    print(end - start)

    with open(outfilename+'.pkl', 'wb') as output:
        pickle.dump(m, output, pickle.HIGHEST_PROTOCOL)

    def test(filedir, filenames, flat_labels=None):
        C = np.zeros((nlabels,nlabels))
        chunk_size = height*width
        print('testing now')
        mat_outdata = {}
        mat_outdata[u'metrics'] = {}

        with open(outfilename+'.csv','a') as fd:
            fd.write('id,rle_mask\n')
            for file_i, image_name in enumerate(tqdm(filenames)):
                filename = os.path.join(filedir, 'images', image_name)
                im=Image.open(filename).convert('L')
                npim = np.array(im, dtype=np.float32) / 255.0
                image_padded = np.pad(npim, ((ap[0]/2,ap[0]/2),(ap[1]/2,ap[1]/2)), 'reflect')

                subimg = np.expand_dims(image_padded,0)
                subimg = np.expand_dims(subimg,-1)

                feed_dict = {x: subimg}
                net_out = sess.run(feat, feed_dict).reshape((height*width, feat_size))

                p_label, p_acc, p_val = svm_predict(np.zeros(height*width).tolist(), net_out.tolist(), m, '-q');
                if flat_labels is not None:
                    C += confusion_matrix(flat_labels[(file_i*chunk_size):((file_i+1)*chunk_size)], p_label, labels=[0,1])
                    mat_outdata[u'metrics'][u'CM'] = C
                    hdf5storage.write(mat_outdata, filename=outfilename+'.mat', matlab_compatible=True)
                else:
                    pred = np.array(p_label).reshape((height,width)).transpose().reshape(chunk_size)
                    fileid, file_extension = os.path.splitext(image_name)
                    fd.write('%s,%s\n' % (fileid, myrlestring(pred)))

    test(testdir, testfilenames)

def clean_glob(myglob):
    outfiles = []
    for filepath in myglob:
        path, filename = os.path.split(filepath)
        outfiles.append(filename)
    return outfiles

def main():
    trainpaths, _ = split_trainval(0.87)
    trainfiles = clean_glob(trainpaths)
    # valfiles = ['1bd1c8c771.png', '01c033e116.png', '1c6237ae58.png', '1d0c2fa004.png']
    testpaths = glob.glob('/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/test/images/*.png')
    testfiles = clean_glob(testpaths)

    run_baseline(scat2d_to_2d_2layer,
        '/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/train',
        '/scratch0/ilya/locDoc/data/kaggle-seismic-dataset/test',
        trainfilenames=trainfiles,
        testfilenames=testfiles,
        outfilename='svm1000')

if __name__ == '__main__':
    main()
