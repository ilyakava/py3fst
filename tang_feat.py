"""Just feature extraction
"""

from collections import namedtuple
import itertools
import time
import os

import h5py
import hdf5storage
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import scipy.io as sio

import windows as win

import pdb

layerO = namedtuple('layerO', ['strides', 'padding'])

def scat3d(x, win_params, layer_params):
    """
    Args:
        x is input with dim (batch, depth, height, width, channels)
        win_params.filters is complex with dim (depth, height, width, channels)
    """
    real1 = tf.layers.conv3d(
        x,
        win_params.nfilt,
        win_params.kernel_size,
        strides=layer_params.strides,
        padding=layer_params.padding,
        dilation_rate=(1, 1, 1),
        activation=None,
        use_bias=False,
        kernel_initializer=tf.constant_initializer(win_params.filters.real, dtype=tf.float32),
        trainable=False,
        name=None
    )

    imag1 = tf.layers.conv3d(
        x,
        win_params.nfilt,
        win_params.kernel_size,
        strides=layer_params.strides,
        padding=layer_params.padding,
        dilation_rate=(1, 1, 1),
        activation=None,
        use_bias=False,
        kernel_initializer=tf.constant_initializer(win_params.filters.imag, dtype=tf.float32),
        trainable=False,
        name=None
    )

    return tf.abs(tf.complex(real1, imag1))

def tang_net(x, reuse=False):
    """
    Args:
        x: image in (heigh, width, bands) format
    Output:
        center pixel feature vector
    """
    kernel_size = [7,7,7]
    max_scale = 3
    K = 3

    psiO = win.tang_psi_factory(max_scale, K, kernel_size)
    phiO = win.tang_phi_window_3D(max_scale, kernel_size)

    with tf.variable_scope('TangNet', reuse=reuse):
        x = tf.transpose(x, [2, 0, 1])

        x = tf.expand_dims(x, 0)
        x = tf.expand_dims(x, -1)

        layer1_params = layerO((1,1,1), 'valid')
        U1 = scat3d(x, psiO, layer1_params)

        # swap channels with batch
        U1 = tf.transpose(U1, [4, 1, 2, 3, 0])

        layer2_params = layerO((1,1,1), 'valid')
        U2 = scat3d(U1, psiO, layer2_params)

        # merge channels with batch
        U2 = tf.transpose(U2, [0, 4, 1, 2, 3])
        U2 = tf.reshape(U2, [U2.shape[0]*U2.shape[1], U2.shape[2], U2.shape[3], U2.shape[4]])
        U2 = tf.expand_dims(U2, -1)

        # convolve with phis
        layer2_lo_params = layerO((1,1,1), 'valid')
        S2 = scat3d(U2, phiO, layer2_lo_params)

        S1 = scat3d(U1[:,3:-3, 3:-3, 3:-3, :], phiO, layer2_lo_params)
        S0 = scat3d(x[:,6:-6, 6:-6, 6:-6, :], phiO, layer2_lo_params)

        S2 = tf.reshape(S2, [S2.shape[0] * S2.shape[1]])
        S1 = tf.reshape(S1, [S1.shape[0] * S1.shape[1]])
        S0 = tf.reshape(S0, [S0.shape[1]])

    return tf.concat([S0,S1,S2],0)

def tang_run_IP(data):
    x = tf.placeholder(tf.float32, shape=(19,19,218))
    feat = tang_net(x)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    padded_data = np.pad(data, ((9,9),(9,9),(9,9)), 'wrap')

    matfiledata = {}
    matfiledata[u'feats'] = np.zeros((145*145, 151400))

    pixels = np.array(list(itertools.product(range(data.shape[1]),range(data.shape[0]))))
    for pixel_i, pixel in enumerate(tqdm(pixels)):
        # this iterates through columns first
        [pixel_y, pixel_x] = pixel
        subimg = padded_data[pixel_x:(pixel_x+19), pixel_y:(pixel_y+19), :]
    
        feed_dict = {x: subimg}
        matfiledata[u'feats'][pixel_i,:] = sess.run(feat, feed_dict)


    start = time.time()
    hdf5storage.write(matfiledata, filename='/scratch0/ilya/locDoc/data/hyperspec/features/tang_IP.mat', matlab_compatible=True)
    end = time.time()
    print(end - start)
    pdb.set_trace()

    mat_contents = sio.loadmat('/scratch0/ilya/locDoc/data/hyperspec/Indian_pines_gt_traintest_1_1abefb.mat' )
    trainX = matfiledata[u'feats'][mat_contents['train_mask'][:,0]==1,:]
    mat_contentsY = sio.loadmat('/scratch0/ilya/locDoc/data/hyperspec/Indian_pines_gt.mat')
    all_labels = mat_contentsY['indian_pines_gt'].transpose().reshape(145*145)
    trainY = all_labels[mat_contents['train_mask'][:,0]==1]

    prob  = svm_problem(trainY.tolist(), trainX.tolist())
    param = svm_parameter('-s 0 -t 0 -q')
    m = svm_train(prob, param)

    testX = matfiledata[u'feats'][mat_contents['test_mask'][:,0]==1,:]
    testY = all_labels[mat_contents['test_mask'][:,0]==1]

    svm_predict(testY.tolist(), testX.tolist(), m)



if __name__ == '__main__':
    
    data_path = '/scratch0/ilya/locDoc/data/hyperspec'
    data_file = os.path.join(data_path, 'Indian_pines_corrected.mat')
    mat_contents = sio.loadmat(data_file)
    data = mat_contents['indian_pines_corrected'].astype(np.float32)
    data /= np.max(np.abs(data))
    tang_run_IP(data)
