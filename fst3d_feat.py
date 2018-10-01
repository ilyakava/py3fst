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
from sklearn.metrics import confusion_matrix

from lib.libsvm.python.svmutil import *
import windows as win

import pdb

DATA_PATH = '/scratch0/ilya/locDoc/data/hyperspec'
DATASET_PATH = '/scratch0/ilya/locDoc/data/hyperspec/datasets'

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
        dilation_rate=(1,1,1),
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
        dilation_rate=(1,1,1),
        activation=None,
        use_bias=False,
        kernel_initializer=tf.constant_initializer(win_params.filters.imag, dtype=tf.float32),
        trainable=False,
        name=None
    )

    return tf.abs(tf.complex(real1, imag1))

def kernel_padding(kernel_size):
    def int_padding(n):
        if n % 2 == 0:
            raise('not implemented even padding')
        else:
            return int((n - 1) / 2)
    return [int_padding(m) for m in kernel_size]

netO = namedtuple('netO', ['model_fn', 'addl_padding'])

def IP_net(reuse=tf.AUTO_REUSE):
    """Fully described network.

    This method is basically "data as code"
        
    Returns:
        struct with fields:
        addl_padding: amount of padding needed for an input to model_fn
        model_fn: function that takes:
            
    """
    psi = win.fst3d_psi_factory([3,9,9])
    phi = win.fst3d_phi_window_3D([3,9,9])
    layer_params = layerO((3,1,1), 'valid')

    def model_fn(x):
        """
        Args:
            x: tf.placeholder in (height, width, nbands) format, shape must be (25,25,nbands+6)
        """
        return hyper3d_net(x, reuse=reuse, psis=[psi,psi],
            phi=phi, layer_params=[layer_params, layer_params, layer_params])

    return netO(model_fn, (24,24,6))

def Bots_net(reuse=tf.AUTO_REUSE):
    psi = win.fst3d_psi_factory([3,7,7])
    phi = win.fst3d_phi_window_3D([3,7,7])
    layer_params = layerO((3,1,1), 'valid')

    def model_fn(x):
        """
        Args:
            x: tf.placeholder in (height, width, nbands) format, shape must be (25,25,nbands+6)
        """
        return hyper3d_net(x, reuse=reuse, psis=[psi,psi],
            phi=phi, layer_params=[layer_params, layer_params, layer_params])

    return netO(model_fn, (18,18,6))

def Smith_net(reuse=tf.AUTO_REUSE):
    """Fully described network.

    This method is basically "data as code"
        
    Returns:
        struct with fields:
        addl_padding: amount of padding needed for an input to model_fn
        model_fn: function that takes:
            
    """
    psi = win.fst3d_psi_factory([5,9,9])
    phi = win.fst3d_phi_window_3D([5,9,9])
    layer_params = layerO((3,1,1), 'valid')

    def model_fn(x):
        """
        Args:
            x: tf.placeholder in (height, width, nbands) format, shape must be (25,25,nbands+6)
        """
        return hyper3d_net(x, reuse=reuse, psis=[psi,psi],
            phi=phi, layer_params=[layer_params, layer_params, layer_params])

    return netO(model_fn, (24,24,12))

def KSC_net(reuse=tf.AUTO_REUSE):
    psi = win.fst3d_psi_factory([7,7,7])
    phi = win.fst3d_phi_window_3D([7,7,7])
    layer_params = layerO((7,1,1), 'valid')

    def model_fn(x):
        """
        Args:
            x: tf.placeholder in (height, width, nbands) format, shape must be (25,25,nbands+6)
        """
        return hyper3d_net(x, reuse=reuse, psis=[psi,psi],
            phi=phi, layer_params=[layer_params, layer_params, layer_params])

    return netO(model_fn, (18,18,12))

def KSC_net2(reuse=tf.AUTO_REUSE):
    psi = win.fst3d_psi_factory([3,9,9])
    phi = win.fst3d_phi_window_3D([3,9,9])
    layer_params = layerO((1,1,1), 'valid')

    def model_fn(x):
        """
        Args:
            x: tf.placeholder in (height, width, nbands) format, shape must be (25,25,nbands+6)
        """
        return hyper3d_net(x, reuse=reuse, psis=[psi,psi],
            phi=phi, layer_params=[layer_params, layer_params, layer_params])
    return netO(model_fn, (24,24,6))

def hyper3d_net(x, reuse=tf.AUTO_REUSE, psis=None, phi=None, layer_params=None):
    """Computes features for a specific pixel.

    Args:
        x: image in (height, width, bands) format
        psis: array of winO struct, filters are in (bands, height, width) format!
        phi: winO struct, filters are in (bands, height, width) format!
    Output:
        center pixel feature vector
    """
    assert len(layer_params) == 3, 'this network is 2 layers only'
    assert len(psis) == 2, 'this network is 2 layers only'

    
    with tf.variable_scope('Hyper3DNet', reuse=reuse):
        x = tf.transpose(x, [2, 0, 1])

        x = tf.expand_dims(x, 0)
        x = tf.expand_dims(x, -1)

        U1 = scat3d(x, psis[0], layer_params[0])


        # swap channels with batch
        U1 = tf.transpose(U1, [4, 1, 2, 3, 0])
        
        U2s = []
        # only procede with increasing frequency paths
        for res_i, used_params in enumerate(psis[0].filter_params):
            increasing_psi = win.fst3d_psi_factory(psis[1].kernel_size, used_params)
            if increasing_psi.nfilt > 0:
                U2s.append(scat3d(U1[res_i:(res_i+1),:,:,:,:], increasing_psi, layer_params[1]))

        U2 = tf.concat(U2s, 4)
        # swap channels with batch
        U2 = tf.transpose(U2, [4, 1, 2, 3, 0])

        # convolve with phis
        S2 = scat3d(U2, phi, layer_params[2])

        [p1h, p1w, p1b] = kernel_padding(psis[0].kernel_size)
        [p2h, p2w, p2b] = kernel_padding(psis[0].kernel_size)
        p2h += p1h; p2w += p1w; p2b += p1b;

        S1 = scat3d(U1[:,(p1h):-(p1h), (p1w):-(p1w), (p1b):-(p1b), :], phi, layer_params[2])
        
        S0 = scat3d(x[:,(p2h):-(p2h), (p2w):-(p2w), (p2b):-(p2b), :], phi, layer_params[2])

        # flatten everything
        S2 = tf.reshape(S2, [S2.shape[0] * S2.shape[1]]) # enforces last 3 dimensions being 1
        S1 = tf.reshape(S1, [S1.shape[0] * S1.shape[1]]) # enforces last 3 dimensions being 1
        S0 = tf.reshape(S0, [S0.shape[1]]) # enforces all but dim1 being 1

    return tf.concat([S0,S1,S2], 0)

def hyper_run_acc(data, labels, netO, traintestfilenames=None, outfilename=None):
    """
    Args: data, image in (height, width, nbands) format
    """
    [height, width, nbands] = data.shape


    all_pixels = np.array(list(itertools.product(range(width),range(height))))
    labelled_pixels = np.array(filter(lambda (x,y): labels[y,x] != 0, all_pixels))
    flat_labels = labels.transpose().reshape(height*width)
    nlabels = len(set(flat_labels.tolist())) - 1

    ap = np.array(netO.addl_padding)
    assert np.all(ap % 2 == 0), 'Assymetric is not supported'
    net_in_shape = ap + np.array([1,1,nbands])
    x = tf.placeholder(tf.float32, shape=net_in_shape)
    feat = netO.model_fn(x)

    padded_data = np.pad(data, ((ap[0]/2,ap[0]/2),(ap[1]/2,ap[1]/2),(ap[2]/2,ap[2]/2)), 'wrap')

    print('requesting %d MB memory' % (labelled_pixels.shape[0] * feat.shape[0] * 4 / 1000000.0))
    labelled_pix_feat = np.zeros((labelled_pixels.shape[0], feat.shape[0]), dtype=np.float32)
    
    def compute_features():
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for pixel_i, pixel in enumerate(tqdm(labelled_pixels)):
            # this iterates through columns first
            [pixel_x, pixel_y] = pixel
            subimg = padded_data[pixel_y:(pixel_y+ap[0]+1), pixel_x:(pixel_x+ap[1]+1), :]
        
            feed_dict = {x: subimg}
            labelled_pix_feat[pixel_i,:] = sess.run(feat, feed_dict)
    compute_features()

    for traintestfilename in traintestfilenames:
        mat_contents = sio.loadmat(os.path.join(DATA_PATH, traintestfilename))
        train_mask = mat_contents['train_mask'].astype(int).squeeze()
        test_mask = mat_contents['test_mask'].astype(int).squeeze()
        # resize train/test masks to labelled pixels
        train_mask_skip_unlabelled = train_mask[flat_labels!=0]
        test_mask_skip_unlabelled = test_mask[flat_labels!=0]

        # get training set
        trainY = flat_labels[train_mask==1]
        trainX = labelled_pix_feat[train_mask_skip_unlabelled==1,:]

        print('training now')
        start = time.time()
        prob  = svm_problem(trainY.tolist(), trainX.tolist())
        param = svm_parameter('-s 0 -t 0 -q')
        m = svm_train(prob, param)
        end = time.time()
        print(end - start)

        if outfilename:
            outfilename = os.path.join(DATA_PATH, outfilename)
        else:
            outfilename = os.path.join(DATA_PATH, traintestfilename+'_pyFST3D_expt.mat')

        # now test
        test_chunk_size = 1000
        testY = flat_labels[test_mask==1]
        testX = labelled_pix_feat[test_mask_skip_unlabelled==1,:]
        C = np.zeros((nlabels,nlabels))
        print('testing now')
        mat_outdata = {}
        mat_outdata[u'metrics'] = {}
        for i in tqdm(range(0,len(testY),test_chunk_size)):
            p_label, p_acc, p_val = svm_predict(testY[i:i+test_chunk_size].tolist(), testX[i:i+test_chunk_size,:].tolist(), m, '-q');
            C += confusion_matrix(testY[i:i+test_chunk_size], p_label, labels=list(range(1,nlabels+1)))

            mat_outdata[u'metrics'][u'CM'] = C
            hdf5storage.write(mat_outdata, filename=outfilename, matlab_compatible=True)

def single_input_example():
    netO = IP_net()
    net_in_shape = np.array(netO.addl_padding) + np.array([1,1,200])
    x = tf.placeholder(tf.float32, shape=net_in_shape)
    feat = netO.model_fn(x)
    
    subimg = np.random.rand(net_in_shape[0], net_in_shape[1], net_in_shape[2])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    feed_dict = {x: subimg}
    myres = sess.run(feat, feed_dict)
    print myres.shape
    pdb.set_trace()

if __name__ == '__main__':
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Indian_pines_corrected.mat'))
    # data = mat_contents['indian_pines_corrected'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Indian_pines_gt.mat'))
    # labels = mat_contents['indian_pines_gt']
    # traintestfilenames = [ 'Indian_pines_gt_traintest_1_1abefb.mat', 'Indian_pines_gt_traintest_2_0bccd7.mat', 'Indian_pines_gt_traintest_3_7b4f69.mat', 'Indian_pines_gt_traintest_4_eeba08.mat', 'Indian_pines_gt_traintest_5_d75e59.mat', 'Indian_pines_gt_traintest_6_3a9ebd.mat', 'Indian_pines_gt_traintest_7_cad093.mat', 'Indian_pines_gt_traintest_8_97b27f.mat', 'Indian_pines_gt_traintest_9_1e4231.mat', 'Indian_pines_gt_traintest_10_6d71a1.mat' ];
    # hyper_run_acc(data, labels, IP_net(), traintestfilenames[:1])

    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Smith.mat'))
    # data = mat_contents['Smith'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Smith_gt.mat'))
    # labels = mat_contents['Smith_gt']

    # traintestfilenames = [ 'Smith_gt_traintest_p05_1_dd77f9.mat', 'Smith_gt_traintest_p05_2_e75152.mat', 'Smith_gt_traintest_p05_3_c8e897.mat', 'Smith_gt_traintest_p05_4_e2bd4d.mat', 'Smith_gt_traintest_p05_5_59815b.mat', 'Smith_gt_traintest_p05_6_316c37.mat', 'Smith_gt_traintest_p05_7_6aef72.mat', 'Smith_gt_traintest_p05_8_c24907.mat', 'Smith_gt_traintest_p05_9_3c2737.mat', 'Smith_gt_traintest_p05_10_75deb4.mat' ];
    # hyper_run_acc(data, labels, Smith_net(), traintestfilenames[:1])

    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Salinas_corrected.mat'))
    # data = mat_contents['salinas_corrected'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Salinas_gt.mat'))
    # labels = mat_contents['salinas_gt']

    # traintestfilenames = [ 'Salinas_gt_traintest_p05_1_4228ee.mat', 'Salinas_gt_traintest_p05_2_eb1804.mat', 'Salinas_gt_traintest_p05_3_fad367.mat', 'Salinas_gt_traintest_p05_4_8cb8a3.mat', 'Salinas_gt_traintest_p05_5_d2384b.mat', 'Salinas_gt_traintest_p05_6_e34195.mat', 'Salinas_gt_traintest_p05_7_249774.mat', 'Salinas_gt_traintest_p05_8_f772c1.mat', 'Salinas_gt_traintest_p05_9_371ee5.mat', 'Salinas_gt_traintest_p05_10_22b46b.mat' ];
    # hyper_run_acc(data, labels, IP_net(), traintestfilenames[:1])


    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'KSC.mat'))
    data = mat_contents['KSC'].astype(np.float32)
    data /= np.max(np.abs(data))
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'KSC_gt.mat'))
    labels = mat_contents['KSC_gt']

    traintestfilenames = [ 'KSC_gt_traintest_1_6061b3.mat', 'KSC_gt_traintest_2_c4043d.mat', 'KSC_gt_traintest_3_db432b.mat', 'KSC_gt_traintest_4_95e0ef.mat', 'KSC_gt_traintest_5_3d7a8e.mat', 'KSC_gt_traintest_6_2a60db.mat', 'KSC_gt_traintest_7_ae63a4.mat', 'KSC_gt_traintest_8_b128c8.mat', 'KSC_gt_traintest_9_9ed856.mat', 'KSC_gt_traintest_10_548b31.mat' ];

    hyper_run_acc(data, labels, KSC_net2(), traintestfilenames[:1], 'kscnet2.mat')
    # hyper_run_acc(data, labels, KSC_net(False), traintestfilenames[:1], 'kscnet.mat')
