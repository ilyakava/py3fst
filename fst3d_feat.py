"""FST-svm
"""

from collections import namedtuple
import itertools
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

def hyper_run_acc(data, labels, netO, traintestfilenames=None, outfilename=None, test_egs=None):
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
    print('computing features now')
    compute_features()

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
        trainY = flat_labels[train_mask==1].tolist()
        trainX = labelled_pix_feat[train_mask_skip_unlabelled==1,:].tolist()

        print('training now')
        start = time.time()
        prob  = svm_problem(trainY, trainX)
        param = svm_parameter('-s 0 -t 0 -q')
        m = svm_train(prob, param)
        end = time.time()
        print(end - start)

        if outfilename:
            nextoutfilename = os.path.join(DATA_PATH, outfilename)
        else:
            nextoutfilename = os.path.join(DATA_PATH, traintestfilename+'_117pyFST3D_expt.mat')

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
        test_flags = '-q'
        if test_egs:
            test_limit_egs = test_egs
            test_flags = ''
        for i in tqdm(range(0,test_limit_egs,test_chunk_size)):
            # populate test X
            this_feat_idxs = shuff_test_labelled_pix_feat_idxs[i:i+test_chunk_size]
            this_labs = shuff_test_labels[i:i+test_chunk_size].tolist()

            p_label, p_acc, p_val = svm_predict(
                this_labs,
                labelled_pix_feat[this_feat_idxs].tolist(), m, test_flags);
            shuff_test_pred_pix[i:i+test_chunk_size] = p_label
            C += confusion_matrix(this_labs, p_label, labels=list(range(1,nlabels+1)))

            mat_outdata[u'metrics'][u'CM'] = C
            hdf5storage.write(mat_outdata, filename=nextoutfilename, matlab_compatible=True)

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
    pdb.set_trace()

def run_all_accs():
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Indian_pines_corrected.mat'))
    # data = mat_contents['indian_pines_corrected'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Indian_pines_gt.mat'))
    # labels = mat_contents['indian_pines_gt']
    # # traintestfilenames = [ 'Indian_pines_gt_traintest_1_1abefb.mat', 'Indian_pines_gt_traintest_2_0bccd7.mat', 'Indian_pines_gt_traintest_3_7b4f69.mat', 'Indian_pines_gt_traintest_4_eeba08.mat', 'Indian_pines_gt_traintest_5_d75e59.mat', 'Indian_pines_gt_traintest_6_3a9ebd.mat', 'Indian_pines_gt_traintest_7_cad093.mat', 'Indian_pines_gt_traintest_8_97b27f.mat', 'Indian_pines_gt_traintest_9_1e4231.mat', 'Indian_pines_gt_traintest_10_6d71a1.mat' ];
    # traintestfilenames = ['Indian_pines_gt_traintest_coarse_14px14p.mat', 'Indian_pines_gt_traintest_coarse_6px6p.mat', 'Indian_pines_gt_traintest_coarse_10px10p.mat', 'Indian_pines_gt_traintest_coarse_12x12_add7s9.mat', 'Indian_pines_gt_traintest_coarse_12x12_skip7s9.mat']
    # hyper_run_acc(data, labels, IP_net(), traintestfilenames[:2])

    # tf.reset_default_graph()

    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Smith_117chan.mat'))
    data = mat_contents['Smith'].astype(np.float32)
    data = pxnn.normalize_channels(data)
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Smith_gt.mat'))
    labels = mat_contents['Smith_gt']

    traintestfilenames = [ 'Smith_gt_traintest_p05_1_dd77f9.mat', 'Smith_gt_traintest_p05_2_e75152.mat', 'Smith_gt_traintest_p05_3_c8e897.mat', 'Smith_gt_traintest_p05_4_e2bd4d.mat', 'Smith_gt_traintest_p05_5_59815b.mat', 'Smith_gt_traintest_p05_6_316c37.mat', 'Smith_gt_traintest_p05_7_6aef72.mat', 'Smith_gt_traintest_p05_8_c24907.mat', 'Smith_gt_traintest_p05_9_3c2737.mat', 'Smith_gt_traintest_p05_10_75deb4.mat' ];
    #traintestfilenames = ['Smith_gt_traintest_coarse_18px18p.mat', 'Smith_gt_traintest_coarse_12px12p.mat']
    hyper_run_acc(data, labels, Smith_net(), traintestfilenames)

    tf.reset_default_graph()

    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Salinas_corrected.mat'))
    # data = mat_contents['salinas_corrected'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Salinas_gt.mat'))
    # labels = mat_contents['salinas_gt']

    # # traintestfilenames = [ 'Salinas_gt_traintest_p05_1_4228ee.mat', 'Salinas_gt_traintest_p05_2_eb1804.mat', 'Salinas_gt_traintest_p05_3_fad367.mat', 'Salinas_gt_traintest_p05_4_8cb8a3.mat', 'Salinas_gt_traintest_p05_5_d2384b.mat', 'Salinas_gt_traintest_p05_6_e34195.mat', 'Salinas_gt_traintest_p05_7_249774.mat', 'Salinas_gt_traintest_p05_8_f772c1.mat', 'Salinas_gt_traintest_p05_9_371ee5.mat', 'Salinas_gt_traintest_p05_10_22b46b.mat' ];
    # traintestfilenames = ['Salinas_gt_traintest_coarse_40px40p.mat', 'Salinas_gt_traintest_coarse_30px30p.mat', 'Salinas_gt_traintest_coarse_20px20p.mat', 'Salinas_gt_traintest_coarse_16x16.mat']
    # hyper_run_acc(data, labels, IP_net(), traintestfilenames[1:3])

    # tf.reset_default_graph()

    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'KSC_corrected.mat'))
    # data = mat_contents['KSC'].astype(np.float32)
    # data = pxnn.normalize_channels(data)
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'KSC_gt.mat'))
    # labels = mat_contents['KSC_gt']

    # traintestfilenames = [ 'KSC_gt_traintest_1_6061b3.mat', 'KSC_gt_traintest_2_c4043d.mat', 'KSC_gt_traintest_3_db432b.mat', 'KSC_gt_traintest_4_95e0ef.mat', 'KSC_gt_traintest_5_3d7a8e.mat', 'KSC_gt_traintest_6_2a60db.mat', 'KSC_gt_traintest_7_ae63a4.mat', 'KSC_gt_traintest_8_b128c8.mat', 'KSC_gt_traintest_9_9ed856.mat', 'KSC_gt_traintest_10_548b31.mat' ];
    # hyper_run_acc(data, labels, KSC_net(), traintestfilenames[:1])

    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Botswana.mat'))
    # data = mat_contents['Botswana'].astype(np.float32)
    # data = pxnn.normalize_channels(data)
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Botswana_gt.mat'))
    # labels = mat_contents['Botswana_gt']
    # # traintestfilenames = [ 'Botswana_gt_traintest_1_e24fae.mat', 'Botswana_gt_traintest_2_518c23.mat', 'Botswana_gt_traintest_3_7b7b6a.mat', 'Botswana_gt_traintest_4_588b5a.mat', 'Botswana_gt_traintest_5_60813e.mat', 'Botswana_gt_traintest_6_05a6b3.mat', 'Botswana_gt_traintest_7_fbba81.mat', 'Botswana_gt_traintest_8_a083a4.mat', 'Botswana_gt_traintest_9_8591e0.mat', 'Botswana_gt_traintest_10_996e67.mat' ];
    # traintestfilenames = ['Botswana_gt_traintest_coarse_36px36p.mat', 'Botswana_gt_traintest_coarse_12px12p.mat']
    # hyper_run_acc(data, labels, Bots_net(), traintestfilenames[:1])

    # tf.reset_default_graph()

    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'PaviaU.mat'))
    # data = mat_contents['paviaU'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'PaviaU_gt.mat'))
    # labels = mat_contents['paviaU_gt']

    # # traintestfilenames = [ 'PaviaU_gt_traintest_1_334428.mat', 'PaviaU_gt_traintest_2_03ccd1.mat', 'PaviaU_gt_traintest_3_698d0c.mat', 'PaviaU_gt_traintest_4_7b2f96.mat', 'PaviaU_gt_traintest_5_8adc4a.mat', 'PaviaU_gt_traintest_6_b1ef2f.mat', 'PaviaU_gt_traintest_7_844918.mat', 'PaviaU_gt_traintest_8_16b8dc.mat', 'PaviaU_gt_traintest_9_e14191.mat', 'PaviaU_gt_traintest_10_c36f7c.mat' ];
    # traintestfilenames = ['PaviaU_gt_traintest_coarse_16px16p.mat', 'PaviaU_gt_traintest_coarse_32px32p.mat', 'PaviaU_gt_traintest_coarse_64px64p.mat', 'PaviaU_gt_traintest_coarse_128px128p.mat']
    # hyper_run_acc(data, labels, Pavia_net(), traintestfilenames[:])

    # tf.reset_default_graph()

    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Pavia_center_right'))
    # data = mat_contents['Pavia_center_right'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Pavia_center_right_gt.mat'))
    # labels = mat_contents['Pavia_center_right_gt']

    # # traintestfilenames = [ 'Pavia_center_right_gt_traintest_1_c23379.mat', 'Pavia_center_right_gt_traintest_2_555d38.mat', 'Pavia_center_right_gt_traintest_3_436123.mat', 'Pavia_center_right_gt_traintest_4_392727.mat', 'Pavia_center_right_gt_traintest_5_da2b6f.mat', 'Pavia_center_right_gt_traintest_6_9848f9.mat', 'Pavia_center_right_gt_traintest_7_2e4963.mat', 'Pavia_center_right_gt_traintest_8_12c92f.mat', 'Pavia_center_right_gt_traintest_9_7593be.mat', 'Pavia_center_right_gt_traintest_10_30cc68.mat' ];
    # traintestfilenames = ['Pavia_center_right_gt_traintest_coarse_128px128p.mat','Pavia_center_right_gt_traintest_coarse_72px72p.mat','Pavia_center_right_gt_traintest_coarse_36px36p.mat']
    # hyper_run_acc(data, labels, PaviaR_net(), traintestfilenames[:])

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
    run_all_full_imgs()
    # run_all_accs()

