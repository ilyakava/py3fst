"""FST-svm
"""

from collections import namedtuple
import itertools
from itertools import product
import time
import os
import random

import h5py
import hdf5storage
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import scipy.io as sio
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from lib.libsvm.python.svmutil import *
import windows as win

import rgb_pixelNN as pxnn

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

def conv3dfeat(x, win_params, layer_params, final_size):
    """3D filters and concats a cube.
    Args:
        x is input with dim (h,w,bands)
        win_params.filters is float32 with dim (depth, height, width, channels)
    Returns:
        cube that is (h,w,bands)
        
        
    Says that the magnitude of the filtering result is the feature
    """
    x = tf.transpose(x, [2, 0, 1])

    x = tf.expand_dims(x, 0)
    x = tf.expand_dims(x, -1)
    conv1 = tf.layers.conv3d(
        x,
        win_params.nfilt,
        win_params.kernel_size,
        strides=layer_params.strides,
        padding=layer_params.padding,
        dilation_rate=(1,1,1),
        activation=None,
        use_bias=False,
        kernel_initializer=tf.constant_initializer(win_params.filters, dtype=tf.float32),
        trainable=False,
        name=None
    )
    # x is now (1,bands,h,w,nfilt)
    conv1 = tf.transpose(conv1, [0,4,1,2,3])
    # x is now (1,nfilt,bands,h,w)
    feat = tf.reshape(conv1, [-1,final_size, final_size])
    feat = tf.transpose(feat, [1,2,0])
    return tf.abs(feat)
    # return feat

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

def custom_net(s1=9, s2=9, e1=3, e2=3):
    psi1 = win.fst3d_psi_factory([e1,s1,s1])
    psi2 = win.fst3d_psi_factory([e2,s2,s2])
    phi = win.fst3d_phi_window_3D([e2,s2,s2])
    lp1 = layerO((min(e1,5),1,1), 'valid')
    lp2 = layerO((min(e2,5),1,1), 'valid')
    lp3 = layerO((min(e2,5),1,1), 'valid')

    def model_fn(x):
        return hyper3d_net(x, reuse=False, psis=[psi1,psi2],
            phi=phi, layer_params=[lp1, lp2, lp3])

    return netO(model_fn, (s1+s2+s2-3,s1+s2+s2-3,e1+e2+e2-3))

def Bots_net(reuse=tf.AUTO_REUSE):
    s = 11
    psi1 = win.fst3d_psi_factory([3,s,s])
    psi2 = win.fst3d_psi_factory([8,s,s])
    phi = win.fst3d_phi_window_3D([8,s,s])
    lp1 = layerO((1,1,1), 'valid')
    lp2 = layerO((8,1,1), 'valid')
    lp3 = layerO((8,1,1), 'valid')

    def model_fn(x):
        """
        Args:
            x: tf.placeholder in (height, width, nbands) format, shape must be (25,25,nbands+6)
        """
        return hyper3d_net(x, reuse=reuse, psis=[psi1,psi2],
            phi=phi, layer_params=[lp1, lp2, lp3])

    return netO(model_fn, ((s-1)*3,(s-1)*3,0))

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
    s = 11
    psi1 = win.fst3d_psi_factory([3,s,s])
    psi2 = win.fst3d_psi_factory([8,s,s])
    phi = win.fst3d_phi_window_3D([8,s,s])
    lp1 = layerO((1,1,1), 'valid')
    lp2 = layerO((8,1,1), 'valid')
    lp3 = layerO((8,1,1), 'valid')

    def model_fn(x):
        """
        Args:
            x: tf.placeholder in (height, width, nbands) format, shape must be (25,25,nbands+6)
        """
        return hyper3d_net(x, reuse=reuse, psis=[psi1,psi2],
            phi=phi, layer_params=[lp1, lp2, lp3])

    return netO(model_fn, ((s-1)*3,(s-1)*3,0))

def Pavia_net(reuse=tf.AUTO_REUSE):
    psi = win.fst3d_psi_factory([7,7,7])
    phi = win.fst3d_phi_window_3D([7,7,7])
    layer_params = layerO((3,1,1), 'valid')

    def model_fn(x):
        """
        Args:
            x: tf.placeholder in (height, width, nbands) format, shape must be (25,25,nbands+6)
        """
        return hyper3d_net(x, reuse=reuse, psis=[psi,psi],
            phi=phi, layer_params=[layer_params, layer_params, layer_params])
    return netO(model_fn, (18,18,18))

def PaviaR_net(reuse=tf.AUTO_REUSE):
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
        # x is (1, bands, h, w, 1)

        U1 = scat3d(x, psis[0], layer_params[0])
        # U1 is (1, bands, h, w, lambda1)


        # swap channels with batch
        U1 = tf.transpose(U1, [4, 1, 2, 3, 0])
        # U1 is (lambda1, bands, h, w, 1)
        
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

        [p1h, p1w, p1b] = kernel_padding(psis[1].kernel_size)
        [p2h, p2w, p2b] = kernel_padding(psis[0].kernel_size)
        p2h += p1h; p2w += p1w; p2b += p1b;

        S1 = scat3d(U1[:,(p1h):-(p1h), (p1w):-(p1w), (p1b):-(p1b), :], phi, layer_params[2])
        
        S0 = scat3d(x[:,(p2h):-(p2h), (p2w):-(p2w), (p2b):-(p2b), :], phi, layer_params[2])

        # flatten everything
        S2 = tf.reshape(S2, [S2.shape[0] * S2.shape[1]]) # enforces last 3 dimensions being 1
        S1 = tf.reshape(S1, [S1.shape[0] * S1.shape[1]]) # enforces last 3 dimensions being 1
        S0 = tf.reshape(S0, [S0.shape[1]]) # enforces all but dim1 being 1

    return tf.concat([S0,S1,S2], 0)

def hyper_run_acc(data, labels, netO, traintestfilenames=None, outfilename=None, test_egs=None, expt_name='lin'):
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
        sess.close()
        tf.reset_default_graph()

    print('computing features now')
    compute_features()

    #svm_params = [(10,-3),(9,-3),(8,-3),(7,-3),(6,-3),(5,-3),(4,-3),(10,-5),(9,-5),(8,-5),(7,-5),(6,-5),(10,-4),(9,-4),(8,-4),(7,-4),(6,-4),(5,-4),(10,-6),(9,-6),(8,-6),(7,-6),(5,-5),(4,-4),(6,-6)]
    #for pow_C, pow_gamma in svm_params:
    exp_OAs = []
    expt_name =  'lin'
    #print('starting experiment: %s' % expt_name)
    for traintestfilename in traintestfilenames:
        mat_contents = None
        try:
            mat_contents = sio.loadmat(os.path.join(DATA_PATH, traintestfilename))
        except:
            mat_contents = hdf5storage.loadmat(os.path.join(DATA_PATH, traintestfilename))
        train_mask = mat_contents['train_mask'].astype(int).squeeze()
        test_mask = mat_contents['test_mask'].astype(int).squeeze()
        # resize train/test masks to labelled pixels
        # we have to index X differently than Y since only labelled feat are computed
        train_mask_skip_unlabelled = train_mask[flat_labels!=0]
        test_mask_skip_unlabelled = test_mask[flat_labels!=0]

        # get training set
        trainY = flat_labels[train_mask==1]
        trainX = labelled_pix_feat[train_mask_skip_unlabelled==1,:]


        nextoutfilename = os.path.join(DATA_PATH, traintestfilename+'_pyFST3D_'+expt_name+'_expt.mat')

        print('starting training')
        start = time.time()
        clf = SVC(kernel='linear')
        clf.fit(trainX, trainY)
        end = time.time()
        print(end - start)

        # now test
        test_chunk_size = 1000
        testY = flat_labels[test_mask==1]

        # we want to shuffle the feat and labels in the same order
        # and be able to unshuffle the pred_labels afterwards
        order = range(testY.shape[0]); random.shuffle(order)
        # shuffle idxs into labelled feat
        labelled_pix_feat_idxs = np.array(range(labelled_pix_feat.shape[0]))
        test_labelled_pix_feat_idxs = labelled_pix_feat_idxs[test_mask_skip_unlabelled==1]
        shuff_test_labelled_pix_feat_idxs = test_labelled_pix_feat_idxs[order]
        # and shuffle test labels
        shuff_test_labels = testY[order]

        shuff_test_pred_pix = np.zeros(testY.shape)

        C = np.zeros((nlabels,nlabels))
        print('testing now')
        mat_outdata = {}
        mat_outdata[u'metrics'] = {}
        test_limit_egs = len(testY)
        if test_egs:
            test_limit_egs = test_egs
        for i in tqdm(range(0,test_limit_egs,test_chunk_size)):
            # populate test X
            this_feat_idxs = shuff_test_labelled_pix_feat_idxs[i:i+test_chunk_size]
            this_labs = shuff_test_labels[i:i+test_chunk_size].tolist()

            p_label = clf.predict(labelled_pix_feat[this_feat_idxs]);
            shuff_test_pred_pix[i:i+test_chunk_size] = p_label
            C += confusion_matrix(this_labs, p_label, labels=list(range(1,nlabels+1)))

            mat_outdata[u'metrics'][u'CM'] = C
            hdf5storage.write(mat_outdata, filename=nextoutfilename, matlab_compatible=True)

        exp_OAs.append(100*np.diagonal(C).sum() / C.sum())
        mat_outdata[u'true_image'] = flat_labels.reshape((width, height)).transpose()
        # unshuffle predictions
        Yhat = np.zeros(testY.shape)
        for i, j in enumerate(order):
            Yhat[j] = shuff_test_pred_pix[i]
        # reshape Yhat to an image, and save for later comparison
        pred_image = np.zeros(flat_labels.shape)
        pred_image[test_mask==1] = Yhat
        mat_outdata[u'pred_image'] = pred_image.reshape((width, height)).transpose()
        hdf5storage.write(mat_outdata, filename=nextoutfilename, matlab_compatible=True)

    print('ACC ACHIEVED ({}): {:.4f}'.format(expt_name, np.array(exp_OAs).mean()))

def run_full_img(data, labels, netO, groundtruthfilename='100p'):
    """
    """
    [height, width, nbands] = data.shape
    
    ap = np.array(netO.addl_padding)
    assert np.all(ap % 2 == 0), 'Assymetric is not supported'
    net_in_shape = ap + np.array([1,1,nbands])
    x = tf.placeholder(tf.float32, shape=net_in_shape)
    feat = netO.model_fn(x)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    padded_data = np.pad(data, ((ap[0]/2,ap[0]/2),(ap[1]/2,ap[1]/2),(ap[2]/2,ap[2]/2)), 'reflect')

    all_pixels = np.array(list(itertools.product(range(width),range(height))))
    labelled_pixels = np.array(filter(lambda (x,y): labels[y,x] != 0, all_pixels))
    flat_labels = labels.transpose().reshape(height*width)
    
    print('requesting %d MB memory' % (labelled_pixels.shape[0] * feat.shape[0] * 4 / 1000000.0))
    labelled_pix_feat = np.zeros((labelled_pixels.shape[0], feat.shape[0]), dtype=np.float32)

    def compute_features():

        for pixel_i, pixel in enumerate(tqdm(labelled_pixels)):
            # this iterates through columns first
            [pixel_x, pixel_y] = pixel
            subimg = padded_data[pixel_y:(pixel_y+ap[0]+1), pixel_x:(pixel_x+ap[1]+1), :]
        
            feed_dict = {x: subimg}
            labelled_pix_feat[pixel_i,:] = sess.run(feat, feed_dict)
    print('computing features now')
    compute_features()

    trainY = flat_labels[flat_labels!=0]
    
    print('starting training')
    start = time.time()
    clf = SVC(kernel='linear')
    clf.fit(labelled_pix_feat, trainY)
    end = time.time()
    print(end - start)

    # now start predicting the full image, 1 column at a time
    col_feat = np.zeros((height, feat.shape[0]), dtype=np.float32)
    pred_image = np.zeros((height,width), dtype=int)
    for pixel_x in tqdm(range(width)):
        # get feat
        for pixel_y in range(height):
            subimg = padded_data[pixel_y:(pixel_y+ap[0]+1), pixel_x:(pixel_x+ap[1]+1), :]
            feed_dict = {x: subimg}
            col_feat[pixel_y,:] = sess.run(feat, feed_dict)
    
        # get pred for feat
        p_label = clf.predict(col_feat);
        pred_image[:,pixel_x] = np.array(p_label).astype(int)

    imgmatfiledata = {}
    imgmatfiledata[u'imgHat'] = pred_image
    imgmatfiledata[u'groundtruthfilename'] = groundtruthfilename+'_100p_3dfst_fullimg.mat'
    hdf5storage.write(imgmatfiledata, filename=imgmatfiledata[u'groundtruthfilename'], matlab_compatible=True)
    print('done making img, run hundredpercent_img_figures.m')



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

def run_all_accs():
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Indian_pines_corrected.mat'))
    # data = mat_contents['indian_pines_corrected'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Indian_pines_gt.mat'))
    # labels = mat_contents['indian_pines_gt']

    # traintestfilenames = ['Indian_pines_gt_traintest_coarse_14px14p.mat', 'Indian_pines_gt_traintest_coarse_6px6p.mat', 'Indian_pines_gt_traintest_coarse_10px10p.mat', 'Indian_pines_gt_traintest_coarse_12x12_add7s9.mat', 'Indian_pines_gt_traintest_coarse_12x12_skip7s9.mat']
    # traintestfilenames = [ 'Indian_pines_gt_traintest_1_1abefb.mat', 'Indian_pines_gt_traintest_2_0bccd7.mat', 'Indian_pines_gt_traintest_3_7b4f69.mat', 'Indian_pines_gt_traintest_4_eeba08.mat', 'Indian_pines_gt_traintest_5_d75e59.mat', 'Indian_pines_gt_traintest_6_3a9ebd.mat', 'Indian_pines_gt_traintest_7_cad093.mat', 'Indian_pines_gt_traintest_8_97b27f.mat', 'Indian_pines_gt_traintest_9_1e4231.mat', 'Indian_pines_gt_traintest_10_6d71a1.mat' ];
    
    # takesome = []

    # traintestfilenames = [ 'Indian_pines_gt_traintest_p04_nozero_1_05cf41.mat', 'Indian_pines_gt_traintest_p04_nozero_2_ce4ce0.mat', 'Indian_pines_gt_traintest_p04_nozero_3_c2fb75.mat', 'Indian_pines_gt_traintest_p04_nozero_4_5d3141.mat', 'Indian_pines_gt_traintest_p04_nozero_5_0d824a.mat', 'Indian_pines_gt_traintest_p04_nozero_6_6e4725.mat', 'Indian_pines_gt_traintest_p04_nozero_7_3e6a00.mat', 'Indian_pines_gt_traintest_p04_nozero_8_957ed5.mat', 'Indian_pines_gt_traintest_p04_nozero_9_9eb6a2.mat', 'Indian_pines_gt_traintest_p04_nozero_10_76cc88.mat' ];
    # takesome += traintestfilenames
    
    # traintestfilenames = [ 'Indian_pines_gt_traintest_p03_nozero_1_c162cc.mat', 'Indian_pines_gt_traintest_p03_nozero_2_2db4c5.mat', 'Indian_pines_gt_traintest_p03_nozero_3_4a0c9f.mat', 'Indian_pines_gt_traintest_p03_nozero_4_b293fe.mat', 'Indian_pines_gt_traintest_p03_nozero_5_40d425.mat', 'Indian_pines_gt_traintest_p03_nozero_6_58f5f9.mat', 'Indian_pines_gt_traintest_p03_nozero_7_c677ec.mat', 'Indian_pines_gt_traintest_p03_nozero_8_f53e55.mat', 'Indian_pines_gt_traintest_p03_nozero_9_3bdfbf.mat', 'Indian_pines_gt_traintest_p03_nozero_10_ef5555.mat' ];
    # takesome += traintestfilenames

    # traintestfilenames = [ 'Indian_pines_gt_traintest_p02_nozero_1_93e12e.mat', 'Indian_pines_gt_traintest_p02_nozero_2_06eda5.mat', 'Indian_pines_gt_traintest_p02_nozero_3_e27f64.mat', 'Indian_pines_gt_traintest_p02_nozero_4_5268bc.mat', 'Indian_pines_gt_traintest_p02_nozero_5_9d0774.mat', 'Indian_pines_gt_traintest_p02_nozero_6_733c26.mat', 'Indian_pines_gt_traintest_p02_nozero_7_4696af.mat', 'Indian_pines_gt_traintest_p02_nozero_8_cc878b.mat', 'Indian_pines_gt_traintest_p02_nozero_9_351667.mat', 'Indian_pines_gt_traintest_p02_nozero_10_f7cbbe.mat' ];
    # takesome += traintestfilenames

    # traintestfilenames = [ 'Indian_pines_gt_traintest_p01_nozero_1_556ea4.mat', 'Indian_pines_gt_traintest_p01_nozero_2_6c358d.mat', 'Indian_pines_gt_traintest_p01_nozero_3_d5e750.mat', 'Indian_pines_gt_traintest_p01_nozero_4_2e12e8.mat', 'Indian_pines_gt_traintest_p01_nozero_5_d6b184.mat', 'Indian_pines_gt_traintest_p01_nozero_6_d9d30c.mat', 'Indian_pines_gt_traintest_p01_nozero_7_f3c39c.mat', 'Indian_pines_gt_traintest_p01_nozero_8_c16774.mat', 'Indian_pines_gt_traintest_p01_nozero_9_b6715b.mat', 'Indian_pines_gt_traintest_p01_nozero_10_8bc7e5.mat' ];
    # takesome += traintestfilenames

    # traintestfilenames = ['Indian_pines_gt_traintest_p05_1_f0b0f8.mat', 'Indian_pines_gt_traintest_p05_2_2c7710.mat', 'Indian_pines_gt_traintest_p05_3_dd1c2c.mat', 'Indian_pines_gt_traintest_p05_4_c44ed3.mat', 'Indian_pines_gt_traintest_p05_5_96acac.mat', 'Indian_pines_gt_traintest_p05_6_c99119.mat', 'Indian_pines_gt_traintest_p05_7_5c222a.mat', 'Indian_pines_gt_traintest_p05_8_a09f39.mat', 'Indian_pines_gt_traintest_p05_9_6e41d3.mat', 'Indian_pines_gt_traintest_p05_10_801219.mat' ]
    # takesome += traintestfilenames

    # traintestfilenames = [ 'Indian_pines_gt_traintest_ma2015_1_9146f0.mat', 'Indian_pines_gt_traintest_ma2015_2_692f24.mat', 'Indian_pines_gt_traintest_ma2015_3_223f7e.mat', 'Indian_pines_gt_traintest_ma2015_4_447c47.mat', 'Indian_pines_gt_traintest_ma2015_5_82c5ad.mat', 'Indian_pines_gt_traintest_ma2015_6_a46a51.mat', 'Indian_pines_gt_traintest_ma2015_7_be4864.mat', 'Indian_pines_gt_traintest_ma2015_8_dacd43.mat', 'Indian_pines_gt_traintest_ma2015_9_962bab.mat', 'Indian_pines_gt_traintest_ma2015_10_f03ef8.mat']
    # takesome += traintestfilenames

    #hyper_run_acc(data, labels, IP_net(), takesome)
    
    #hyper_run_acc(data, labels, custom_net(*(11, 11, 5, 3) ), takesome)

     
    # tf.reset_default_graph()

    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Smith_117chan.mat'))
    # data = mat_contents['Smith'].astype(np.float32)
    # data = pxnn.normalize_channels(data)
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Smith_gt.mat'))
    # labels = mat_contents['Smith_gt']

    # # traintestfilenames = [ 'Smith_gt_traintest_p05_1_dd77f9.mat', 'Smith_gt_traintest_p05_2_e75152.mat', 'Smith_gt_traintest_p05_3_c8e897.mat', 'Smith_gt_traintest_p05_4_e2bd4d.mat', 'Smith_gt_traintest_p05_5_59815b.mat', 'Smith_gt_traintest_p05_6_316c37.mat', 'Smith_gt_traintest_p05_7_6aef72.mat', 'Smith_gt_traintest_p05_8_c24907.mat', 'Smith_gt_traintest_p05_9_3c2737.mat', 'Smith_gt_traintest_p05_10_75deb4.mat' ];
    # traintestfilenames = [ 'Smith_gt_traintest_p05_1_256610.mat', 'Smith_gt_traintest_p05_2_40467b.mat', 'Smith_gt_traintest_p05_3_34ac0b.mat', 'Smith_gt_traintest_p05_4_975f46.mat', 'Smith_gt_traintest_p05_5_7ad5ce.mat', 'Smith_gt_traintest_p05_6_588ff3.mat', 'Smith_gt_traintest_p05_7_be5a75.mat', 'Smith_gt_traintest_p05_8_e931a6.mat', 'Smith_gt_traintest_p05_9_00c835.mat', 'Smith_gt_traintest_p05_10_d8c90f.mat' ];

    # # #traintestfilenames = ['Smith_gt_traintest_coarse_18px18p.mat', 'Smith_gt_traintest_coarse_12px12p.mat']
    # hyper_run_acc(data, labels, Smith_net(), traintestfilenames)

    # tf.reset_default_graph()

    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Salinas_corrected.mat'))
    # data = mat_contents['salinas_corrected'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Salinas_gt.mat'))
    # labels = mat_contents['salinas_gt']

    # traintestfilenames = [ 'Salinas_gt_traintest_ma2015_1_e2c1ec.mat', 'Salinas_gt_traintest_ma2015_2_7ffc47.mat', 'Salinas_gt_traintest_ma2015_3_4a50d1.mat', 'Salinas_gt_traintest_ma2015_4_142526.mat', 'Salinas_gt_traintest_ma2015_5_3cabac.mat', 'Salinas_gt_traintest_ma2015_6_186b23.mat', 'Salinas_gt_traintest_ma2015_7_f7edb0.mat', 'Salinas_gt_traintest_ma2015_8_7491e1.mat', 'Salinas_gt_traintest_ma2015_9_2f2b9b.mat', 'Salinas_gt_traintest_ma2015_10_4cc904.mat' ]
    # traintestfilenames = ['Salinas_gt_traintest_p01_1_648958.mat', 'Salinas_gt_traintest_p01_2_be0673.mat', 'Salinas_gt_traintest_p01_3_45ffc4.mat', 'Salinas_gt_traintest_p01_4_a0aceb.mat', 'Salinas_gt_traintest_p01_5_97aa3b.mat', 'Salinas_gt_traintest_p01_6_d02783.mat', 'Salinas_gt_traintest_p01_7_c24335.mat', 'Salinas_gt_traintest_p01_8_8578df.mat', 'Salinas_gt_traintest_p01_9_0dac3f.mat', 'Salinas_gt_traintest_p01_10_7700db.mat' ];
    # # traintestfilenames = [ 'Salinas_gt_traintest_p05_1_4228ee.mat', 'Salinas_gt_traintest_p05_2_eb1804.mat', 'Salinas_gt_traintest_p05_3_fad367.mat', 'Salinas_gt_traintest_p05_4_8cb8a3.mat', 'Salinas_gt_traintest_p05_5_d2384b.mat', 'Salinas_gt_traintest_p05_6_e34195.mat', 'Salinas_gt_traintest_p05_7_249774.mat', 'Salinas_gt_traintest_p05_8_f772c1.mat', 'Salinas_gt_traintest_p05_9_371ee5.mat', 'Salinas_gt_traintest_p05_10_22b46b.mat' ];
    # traintestfilenames = ['Salinas_gt_traintest_coarse_40px40p.mat', 'Salinas_gt_traintest_coarse_30px30p.mat', 'Salinas_gt_traintest_coarse_20px20p.mat', 'Salinas_gt_traintest_coarse_16x16.mat']
    # hyper_run_acc(data, labels, IP_net(), traintestfilenames[:2])

    # tf.reset_default_graph()

    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'KSC_corrected.mat'))
    # data = mat_contents['KSC'].astype(np.float32)
    # data = pxnn.normalize_channels(data)
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'KSC_gt.mat'))
    # labels = mat_contents['KSC_gt']

    # traintestfilenames = [ 'KSC_gt_traintest_1_6061b3.mat', 'KSC_gt_traintest_2_c4043d.mat', 'KSC_gt_traintest_3_db432b.mat', 'KSC_gt_traintest_4_95e0ef.mat', 'KSC_gt_traintest_5_3d7a8e.mat', 'KSC_gt_traintest_6_2a60db.mat', 'KSC_gt_traintest_7_ae63a4.mat', 'KSC_gt_traintest_8_b128c8.mat', 'KSC_gt_traintest_9_9ed856.mat', 'KSC_gt_traintest_10_548b31.mat' ];
    # hyper_run_acc(data, labels, KSC_net(), traintestfilenames[:1])

    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Botswana.mat'))
    data = mat_contents['Botswana'].astype(np.float32)
    data = pxnn.normalize_channels(data)
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Botswana_gt.mat'))
    labels = mat_contents['Botswana_gt']
    traintestfilenames = [ 'Botswana_gt_traintest_1_e24fae.mat', 'Botswana_gt_traintest_2_518c23.mat', 'Botswana_gt_traintest_3_7b7b6a.mat', 'Botswana_gt_traintest_4_588b5a.mat', 'Botswana_gt_traintest_5_60813e.mat', 'Botswana_gt_traintest_6_05a6b3.mat', 'Botswana_gt_traintest_7_fbba81.mat', 'Botswana_gt_traintest_8_a083a4.mat', 'Botswana_gt_traintest_9_8591e0.mat', 'Botswana_gt_traintest_10_996e67.mat' ];
    # # traintestfilenames = ['Botswana_gt_traintest_coarse_36px36p.mat', 'Botswana_gt_traintest_coarse_12px12p.mat']
    # hyper_run_acc(data, labels, Bots_net(), traintestfilenames[:1])

    params = list(product(*[[9,11,13],[9,11,13],[3,5,7],[5,7,9]]))
    phalf = len(params)//2
    for param in params[:phalf]:
        print('running: {}'.format(param))
        hyper_run_acc(data, labels, custom_net(*param), traintestfilenames[:3])
        tf.reset_default_graph()
    
    # tf.reset_default_graph()

    #mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'PaviaU.mat'))
    #data = mat_contents['paviaU'].astype(np.float32)
    #data /= np.max(np.abs(data))
    #mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'PaviaU_gt.mat'))
    #labels = mat_contents['paviaU_gt']

    # takesome = []
    # traintestfilenames = ['PaviaU_gt_traintest_s50_1_d5e6bd.mat', 'PaviaU_gt_traintest_s50_2_303acb.mat', 'PaviaU_gt_traintest_s50_3_d2eae6.mat', 'PaviaU_gt_traintest_s50_4_25f50b.mat', 'PaviaU_gt_traintest_s50_5_54b9ac.mat', 'PaviaU_gt_traintest_s50_6_66cf3b.mat', 'PaviaU_gt_traintest_s50_7_410a8a.mat', 'PaviaU_gt_traintest_s50_8_c9bc29.mat', 'PaviaU_gt_traintest_s50_9_55d476.mat', 'PaviaU_gt_traintest_s50_10_336ff4.mat' ]
    # takesome += traintestfilenames[2:]
    # # traintestfilenames = ['PaviaU_gt_traintest_p01_1_7d625e.mat', 'PaviaU_gt_traintest_p01_2_a7371a.mat', 'PaviaU_gt_traintest_p01_3_88b44e.mat', 'PaviaU_gt_traintest_p01_4_495c09.mat', 'PaviaU_gt_traintest_p01_5_6cacdd.mat', 'PaviaU_gt_traintest_p01_6_612fad.mat', 'PaviaU_gt_traintest_p01_7_a6a332.mat', 'PaviaU_gt_traintest_p01_8_87c7bc.mat', 'PaviaU_gt_traintest_p01_9_0d17b7.mat', 'PaviaU_gt_traintest_p01_10_1afac4.mat' ]
    # traintestfilenames = [ 'PaviaU_gt_traintest_s03_1_3f6384.mat', 'PaviaU_gt_traintest_s03_2_b67b5f.mat', 'PaviaU_gt_traintest_s03_3_7d8356.mat', 'PaviaU_gt_traintest_s03_4_241266.mat', 'PaviaU_gt_traintest_s03_5_ccbbb1.mat', 'PaviaU_gt_traintest_s03_6_dce186.mat', 'PaviaU_gt_traintest_s03_7_d5cdfe.mat', 'PaviaU_gt_traintest_s03_8_6bcd5a.mat', 'PaviaU_gt_traintest_s03_9_a1ff2b.mat', 'PaviaU_gt_traintest_s03_10_e1dac2.mat' ];
    # takesome += traintestfilenames[3:]
    # traintestfilenames = [ 'PaviaU_gt_traintest_1_334428.mat', 'PaviaU_gt_traintest_2_03ccd1.mat', 'PaviaU_gt_traintest_3_698d0c.mat', 'PaviaU_gt_traintest_4_7b2f96.mat', 'PaviaU_gt_traintest_5_8adc4a.mat', 'PaviaU_gt_traintest_6_b1ef2f.mat', 'PaviaU_gt_traintest_7_844918.mat', 'PaviaU_gt_traintest_8_16b8dc.mat', 'PaviaU_gt_traintest_9_e14191.mat', 'PaviaU_gt_traintest_10_c36f7c.mat' ];
    #traintestfilenames = ['PaviaU_gt_traintest_coarse_16px16p.mat', 'PaviaU_gt_traintest_coarse_32px32p.mat', 'PaviaU_gt_traintest_coarse_64px64p.mat', 'PaviaU_gt_traintest_coarse_128px128p.mat']

    # traintestfilenames = [ 'PaviaU_gt_traintest_s60_1_dd069a.mat', 'PaviaU_gt_traintest_s60_2_ec12ca.mat', 'PaviaU_gt_traintest_s60_3_983688.mat', 'PaviaU_gt_traintest_s60_4_32e811.mat', 'PaviaU_gt_traintest_s60_5_a7a45a.mat', 'PaviaU_gt_traintest_s60_6_6aabaa.mat', 'PaviaU_gt_traintest_s60_7_125045.mat', 'PaviaU_gt_traintest_s60_8_0bcc09.mat', 'PaviaU_gt_traintest_s60_9_4680b3.mat', 'PaviaU_gt_traintest_s60_10_2da9e8.mat' ];
    # takesome += traintestfilenames[2:]

    # traintestfilenames = [ 'PaviaU_gt_traintest_p02_1_9106ec.mat', 'PaviaU_gt_traintest_p02_2_cce574.mat', 'PaviaU_gt_traintest_p02_3_8df071.mat', 'PaviaU_gt_traintest_p02_4_7e94ed.mat', 'PaviaU_gt_traintest_p02_5_a66eb7.mat', 'PaviaU_gt_traintest_p02_6_3ff9dd.mat', 'PaviaU_gt_traintest_p02_7_d2b9c5.mat', 'PaviaU_gt_traintest_p02_8_2f8e11.mat', 'PaviaU_gt_traintest_p02_9_abfbc2.mat', 'PaviaU_gt_traintest_p02_10_ed16d4.mat' ];
    # takesome += traintestfilenames[2:]

    # traintestfilenames = [ 'PaviaU_gt_traintest_s200_1_591636.mat', 'PaviaU_gt_traintest_s200_2_2255d5.mat', 'PaviaU_gt_traintest_s200_3_628d0a.mat', 'PaviaU_gt_traintest_s200_4_26eddf.mat', 'PaviaU_gt_traintest_s200_5_25dd01.mat', 'PaviaU_gt_traintest_s200_6_2430e7.mat', 'PaviaU_gt_traintest_s200_7_409d67.mat', 'PaviaU_gt_traintest_s200_8_f79373.mat', 'PaviaU_gt_traintest_s200_9_dac1e4.mat', 'PaviaU_gt_traintest_s200_10_149f64.mat' ];
    # takesome += traintestfilenames[2:]

    # # traintestfilenames = [ 'PaviaU_gt_traintest_p05_1_a56883.mat', 'PaviaU_gt_traintest_p05_2_1f3a4a.mat', 'PaviaU_gt_traintest_p05_3_b377a5.mat', 'PaviaU_gt_traintest_p05_4_40ac2f.mat', 'PaviaU_gt_traintest_p05_5_cfd19f.mat', 'PaviaU_gt_traintest_p05_6_aadde4.mat', 'PaviaU_gt_traintest_p05_7_b517e8.mat', 'PaviaU_gt_traintest_p05_8_84dd04.mat', 'PaviaU_gt_traintest_p05_9_ed4b5c.mat', 'PaviaU_gt_traintest_p05_10_48b520.mat' ];
    # # takesome += traintestfilenames[2:]

    # traintestfilenames = [ 'PaviaU_gt_traintest_p06_1_a34a5f.mat', 'PaviaU_gt_traintest_p06_2_095dc2.mat', 'PaviaU_gt_traintest_p06_3_9354f7.mat', 'PaviaU_gt_traintest_p06_4_ceaf11.mat', 'PaviaU_gt_traintest_p06_5_f8b8b4.mat', 'PaviaU_gt_traintest_p06_6_7c7f5f.mat', 'PaviaU_gt_traintest_p06_7_4ebcf3.mat', 'PaviaU_gt_traintest_p06_8_4f81aa.mat', 'PaviaU_gt_traintest_p06_9_52736d.mat', 'PaviaU_gt_traintest_p06_10_36f2ec.mat' ];
    # takesome += traintestfilenames[2:]


    # traintestfilenames = [ 'PaviaU_gt_traintest_ma2015_1_0b3591.mat', 'PaviaU_gt_traintest_ma2015_2_88f4ce.mat', 'PaviaU_gt_traintest_ma2015_3_c51f99.mat', 'PaviaU_gt_traintest_ma2015_4_e3a361.mat', 'PaviaU_gt_traintest_ma2015_5_2922fa.mat', 'PaviaU_gt_traintest_ma2015_6_15194e.mat', 'PaviaU_gt_traintest_ma2015_7_df3db2.mat', 'PaviaU_gt_traintest_ma2015_8_ca5afe.mat', 'PaviaU_gt_traintest_ma2015_9_55492c.mat', 'PaviaU_gt_traintest_ma2015_10_a604d2.mat']
    # takesome += traintestfilenames[2:]

    #hyper_run_acc(data, labels, Pavia_net(), list(reversed(traintestfilenames)))

    #mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Pavia_center_right'))
    #data = mat_contents['Pavia_center_right'].astype(np.float32)
    #data /= np.max(np.abs(data))
    #mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Pavia_center_right_gt.mat'))
    #labels = mat_contents['Pavia_center_right_gt']

    # traintestfilenames = ['Pavia_center_right_gt_traintest_s30_1_ac0c37.mat', 'Pavia_center_right_gt_traintest_s30_2_ae70d7.mat', 'Pavia_center_right_gt_traintest_s30_3_b2b41f.mat', 'Pavia_center_right_gt_traintest_s30_4_29ea37.mat', 'Pavia_center_right_gt_traintest_s30_5_ed1aee.mat', 'Pavia_center_right_gt_traintest_s30_6_41a43a.mat', 'Pavia_center_right_gt_traintest_s30_7_17fd46.mat', 'Pavia_center_right_gt_traintest_s30_8_a81d10.mat', 'Pavia_center_right_gt_traintest_s30_9_f2e35e.mat', 'Pavia_center_right_gt_traintest_s30_10_1de379.mat']
    # # traintestfilenames = [ 'Pavia_center_right_gt_traintest_1_c23379.mat', 'Pavia_center_right_gt_traintest_2_555d38.mat', 'Pavia_center_right_gt_traintest_3_436123.mat', 'Pavia_center_right_gt_traintest_4_392727.mat', 'Pavia_center_right_gt_traintest_5_da2b6f.mat', 'Pavia_center_right_gt_traintest_6_9848f9.mat', 'Pavia_center_right_gt_traintest_7_2e4963.mat', 'Pavia_center_right_gt_traintest_8_12c92f.mat', 'Pavia_center_right_gt_traintest_9_7593be.mat', 'Pavia_center_right_gt_traintest_10_30cc68.mat' ];
    #traintestfilenames = ['Pavia_center_right_gt_traintest_coarse_128px128p.mat','Pavia_center_right_gt_traintest_coarse_72px72p.mat','Pavia_center_right_gt_traintest_coarse_36px36p.mat']

    # traintestfilenames = [ 'Pavia_center_right_gt_traintest_p01_1_2ecf33.mat', 'Pavia_center_right_gt_traintest_p01_2_b162bf.mat', 'Pavia_center_right_gt_traintest_p01_3_b199a4.mat', 'Pavia_center_right_gt_traintest_p01_4_a182df.mat', 'Pavia_center_right_gt_traintest_p01_5_403e9e.mat', 'Pavia_center_right_gt_traintest_p01_6_b4cf2f.mat', 'Pavia_center_right_gt_traintest_p01_7_efa5c4.mat', 'Pavia_center_right_gt_traintest_p01_8_a6b7ec.mat', 'Pavia_center_right_gt_traintest_p01_9_725578.mat', 'Pavia_center_right_gt_traintest_p01_10_274170.mat' ];
    # hyper_run_acc(data, labels, Pavia_net(), traintestfilenames)



def run_all_full_imgs():
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Indian_pines_corrected.mat'))
    # data = mat_contents['indian_pines_corrected'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Indian_pines_gt.mat'))
    # labels = mat_contents['indian_pines_gt']
    # run_full_img(data, labels, IP_net(), groundtruthfilename='Indian_pines_gt')

    # tf.reset_default_graph()

    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'PaviaU.mat'))
    # data = mat_contents['paviaU'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'PaviaU_gt.mat'))
    # labels = mat_contents['paviaU_gt']
    # run_full_img(data, labels, Pavia_net(), groundtruthfilename='PaviaU_gt')

    # tf.reset_default_graph()

    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Botswana.mat'))
    # data = mat_contents['Botswana'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Botswana_gt.mat'))
    # labels = mat_contents['Botswana_gt']
    # run_full_img(data, labels, Bots_net(), groundtruthfilename='Botswana_gt')

    # tf.reset_default_graph()
    
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Smith_117chan.mat'))
    data = mat_contents['Smith'].astype(np.float32)
    data = pxnn.normalize_channels(data)
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Smith_gt.mat'))
    labels = mat_contents['Smith_gt']
    run_full_img(data, labels, Smith_net(), groundtruthfilename='Smith_gt')

    tf.reset_default_graph()

    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'KSC.mat'))
    # data = mat_contents['KSC'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'KSC_gt.mat'))
    # labels = mat_contents['KSC_gt']
    # run_full_img(data, labels, KSC_net(), groundtruthfilename='KSC_gt')

    # tf.reset_default_graph()
    
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Pavia_center_right'))
    # data = mat_contents['Pavia_center_right'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Pavia_center_right_gt.mat'))
    # labels = mat_contents['Pavia_center_right_gt']
    # run_full_img(data, labels, PaviaR_net(), groundtruthfilename='Pavia_center_right_gt')

    # tf.reset_default_graph()

    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Salinas_corrected.mat'))
    # data = mat_contents['salinas_corrected'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Salinas_gt.mat'))
    # labels = mat_contents['salinas_gt']
    # run_full_img(data, labels, IP_net(), groundtruthfilename='Salinas_gt')

    # tf.reset_default_graph()

if __name__ == '__main__':
    run_all_accs()
    # run_all_accs()
