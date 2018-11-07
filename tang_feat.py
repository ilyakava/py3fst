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

def tang_net(x, reuse=tf.AUTO_REUSE):
    """Computes tang features for a specific pixel.

    Args:
        x: image in (height, width, bands) format, should be (19,19,nbands+18)
    Output:
        center pixel feature vector

    Example Usage:
        x = tf.placeholder(tf.float32, shape=(19,19,nbands+18))
        feat = tang_net(x)
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        ...

        subimg = padded_data[pixel_x:(pixel_x+19), pixel_y:(pixel_y+19), :]
        
        feed_dict = {x: subimg}
        labelled_pix_feat[pixel_i,:] = sess.run(feat, feed_dict)
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
        
        # separate out different scales
        U1 = tf.reshape(U1, (max_scale, U1.shape[0] / max_scale, U1.shape[1], U1.shape[2], U1.shape[3], U1.shape[4]))

        layer2_params = layerO((1,1,1), 'valid')
        # only continue scattering across increasing scale paths
        U2j0 = scat3d(U1[0,:,:,:,:,:], win.tang_psi_factory(max_scale, K, kernel_size, 1), layer2_params)
        U2j1 = scat3d(U1[1,:,:,:,:,:], win.tang_psi_factory(max_scale, K, kernel_size, 2), layer2_params)

        def merge_channels_with_batch(Uz):
            Uz = tf.transpose(Uz, [0, 4, 1, 2, 3])
            Uz = tf.reshape(Uz, [Uz.shape[0]*Uz.shape[1], Uz.shape[2], Uz.shape[3], Uz.shape[4]])
            return tf.expand_dims(Uz, -1)

        U2j0 = merge_channels_with_batch(U2j0)
        U2j1 = merge_channels_with_batch(U2j1)

        # convolve with phis
        layer2_lo_params = layerO((1,1,1), 'valid')
        S2 = scat3d(tf.concat([U2j0, U2j1], 0), phiO, layer2_lo_params)

        # merge the different scales
        U1 = tf.reshape(U1, (U1.shape[0] * U1.shape[1], U1.shape[2], U1.shape[3], U1.shape[4], U1.shape[5]))
        S1 = scat3d(U1[:,3:-3, 3:-3, 3:-3, :], phiO, layer2_lo_params)
        
        S0 = scat3d(x[:,6:-6, 6:-6, 6:-6, :], phiO, layer2_lo_params)

        # flatten everything
        S2 = tf.reshape(S2, [S2.shape[0] * S2.shape[1]])
        S1 = tf.reshape(S1, [S1.shape[0] * S1.shape[1]])
        S0 = tf.reshape(S0, [S0.shape[1]])

    return tf.concat([S0,S1,S2], 0)

def tang_run_full_img(data, labels, groundtruthfilename='100p'):
    """
    """
    [height, width, nbands] = data.shape
    x = tf.placeholder(tf.float32, shape=(19,19,nbands+18))
    feat = tang_net(x)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    padded_data = np.pad(data, ((9,9),(9,9),(9,9)), 'wrap')



    all_pixels = np.array(list(itertools.product(range(width),range(height))))
    labelled_pixels = np.array(filter(lambda (x,y): labels[y,x] != 0, all_pixels))
    
    print('requesting %d MB memory' % (labelled_pixels.shape[0] * 271*nbands * 4 / 1000000.0))
    labelled_pix_feat = np.zeros((labelled_pixels.shape[0], 271*nbands), dtype=np.float32)

    for pixel_i, pixel in enumerate(tqdm(labelled_pixels)):
        # this iterates through columns first
        [pixel_x, pixel_y] = pixel
        subimg = padded_data[pixel_x:(pixel_x+19), pixel_y:(pixel_y+19), :]
    
        feed_dict = {x: subimg}
        labelled_pix_feat[pixel_i,:] = sess.run(feat, feed_dict)

    flat_labels = labels.transpose().reshape(height*width)
    trainY = flat_labels[flat_labels!=0]
    
    start = time.time()
    prob  = svm_problem(trainY.tolist(), labelled_pix_feat.tolist())
    param = svm_parameter('-s 0 -t 0 -q')
    m = svm_train(prob, param)
    end = time.time()
    print(end - start)

    # now start predicting the full image, 1 column at a time
    col_feat = np.zeros((height, 271*nbands), dtype=np.float32)
    pred_image = np.zeros((height,width), dtype=int)
    for pixel_x in tqdm(range(width)):
        # get feat
        for pixel_y in range(height):
            subimg = padded_data[pixel_x:(pixel_x+19), pixel_y:(pixel_y+19), :]
            feed_dict = {x: subimg}
            col_feat[pixel_y,:] = sess.run(feat, feed_dict)
    
        # get pred for feat
        dontcare = [0] * height
        p_label, p_acc, p_val = svm_predict(dontcare, col_feat.tolist(), m);
        pred_image[:,pixel_x] = np.array(p_label).astype(int)

    imgmatfiledata = {}
    imgmatfiledata[u'imgHat'] = pred_image
    imgmatfiledata[u'groundtruthfilename'] = groundtruthfilename
    hdf5storage.write(imgmatfiledata, filename=groundtruthfilename+'_100p_tang_fullimg.mat', matlab_compatible=True)
    print('done making img, run hundredpercent_img_figures.m')

def tang_run_acc(data, labels, traintestfilenames=None):
    """
    """
    [height, width, nbands] = data.shape


    all_pixels = np.array(list(itertools.product(range(width),range(height))))
    labelled_pixels = np.array(filter(lambda (x,y): labels[y,x] != 0, all_pixels))
    flat_labels = labels.transpose().reshape(height*width)
    nlabels = len(set(flat_labels.tolist())) - 1

    padded_data = np.pad(data, ((9,9),(9,9),(9,9)), 'wrap')

    print('requesting %d MB memory' % (labelled_pixels.shape[0] * 271*nbands * 4 / 1000000.0))
    labelled_pix_feat = np.zeros((labelled_pixels.shape[0], 271*nbands), dtype=np.float32)
    
    def compute_features():
        x = tf.placeholder(tf.float32, shape=(19,19,nbands+18))
        feat = tang_net(x)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for pixel_i, pixel in enumerate(tqdm(labelled_pixels)):
            # this iterates through columns first
            [pixel_x, pixel_y] = pixel
            subimg = padded_data[pixel_y:(pixel_y+19), pixel_x:(pixel_x+19), :]
        
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

        # now test
        test_chunk_size = 1000
        testY = flat_labels[test_mask==1]
        testX = labelled_pix_feat[test_mask_skip_unlabelled==1,:]
        C = np.zeros((nlabels,nlabels))
        print('testing now')
        for i in tqdm(range(0,len(testY),test_chunk_size)):
            p_label, p_acc, p_val = svm_predict(testY[i:i+test_chunk_size].tolist(), testX[i:i+test_chunk_size,:].tolist(), m, '-q');
            C += confusion_matrix(testY[i:i+test_chunk_size], p_label, labels=list(range(1,nlabels+1)))

        mat_outdata = {}
        mat_outdata[u'metrics'] = {}
        mat_outdata[u'metrics'][u'CM'] = C
        hdf5storage.write(mat_outdata, filename=os.path.join(DATA_PATH, traintestfilename+'_WST3D_expt.mat'), matlab_compatible=True)


def tang_run_all_full_imgs():
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Indian_pines_corrected.mat'))
    # data = mat_contents['indian_pines_corrected'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Indian_pines_gt.mat'))
    # labels = mat_contents['indian_pines_gt']
    # tang_run(data, labels, groundtruthfilename='Indian_pines_gt')

    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'PaviaU.mat'))
    data = mat_contents['paviaU'].astype(np.float32)
    data /= np.max(np.abs(data))
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'PaviaU_gt.mat'))
    labels = mat_contents['paviaU_gt']
    tang_run_full_img(data, labels, groundtruthfilename='PaviaU_gt')

    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Botswana.mat'))
    data = mat_contents['Botswana'].astype(np.float32)
    data /= np.max(np.abs(data))
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Botswana_gt.mat'))
    labels = mat_contents['Botswana_gt']
    tang_run_full_img(data, labels, groundtruthfilename='Botswana_gt')
    
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'KSC.mat'))
    # data = mat_contents['KSC'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'KSC_gt.mat'))
    # labels = mat_contents['KSC_gt']
    # tang_run(data, labels, groundtruthfilename='KSC_gt')
    
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Pavia_center_right'))
    data = mat_contents['Pavia_center_right'].astype(np.float32)
    data /= np.max(np.abs(data))
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Pavia_center_right_gt.mat'))
    labels = mat_contents['Pavia_center_right_gt']
    tang_run_full_img(data, labels, groundtruthfilename='Pavia_center_right_gt')

    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Smith.mat'))
    data = mat_contents['Smith'].astype(np.float32)
    data /= np.max(np.abs(data))
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Smith_gt.mat'))
    labels = mat_contents['Smith_gt']
    tang_run_full_img(data, labels, groundtruthfilename='Smith_gt')

    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Salinas_corrected.mat'))
    data = mat_contents['salinas_corrected'].astype(np.float32)
    data /= np.max(np.abs(data))
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Salinas_gt.mat'))
    labels = mat_contents['salinas_gt']
    tang_run_full_img(data, labels, groundtruthfilename='Salinas_gt')


# datasettrainingfiles = [ 'Indian_pines_gt_traintest_1_1abefb.mat', 'Indian_pines_gt_traintest_2_0bccd7.mat', 'Indian_pines_gt_traintest_3_7b4f69.mat', 'Indian_pines_gt_traintest_4_eeba08.mat', 'Indian_pines_gt_traintest_5_d75e59.mat', 'Indian_pines_gt_traintest_6_3a9ebd.mat', 'Indian_pines_gt_traintest_7_cad093.mat', 'Indian_pines_gt_traintest_8_97b27f.mat', 'Indian_pines_gt_traintest_9_1e4231.mat', 'Indian_pines_gt_traintest_10_6d71a1.mat' ];
# datasettrainingfiles = [ 'PaviaU_gt_traintest_1_334428.mat', 'PaviaU_gt_traintest_2_03ccd1.mat', 'PaviaU_gt_traintest_3_698d0c.mat', 'PaviaU_gt_traintest_4_7b2f96.mat', 'PaviaU_gt_traintest_5_8adc4a.mat', 'PaviaU_gt_traintest_6_b1ef2f.mat', 'PaviaU_gt_traintest_7_844918.mat', 'PaviaU_gt_traintest_8_16b8dc.mat', 'PaviaU_gt_traintest_9_e14191.mat', 'PaviaU_gt_traintest_10_c36f7c.mat' ];
# datasettrainingfiles = [ 'Botswana_gt_traintest_1_e24fae.mat', 'Botswana_gt_traintest_2_518c23.mat', 'Botswana_gt_traintest_3_7b7b6a.mat', 'Botswana_gt_traintest_4_588b5a.mat', 'Botswana_gt_traintest_5_60813e.mat', 'Botswana_gt_traintest_6_05a6b3.mat', 'Botswana_gt_traintest_7_fbba81.mat', 'Botswana_gt_traintest_8_a083a4.mat', 'Botswana_gt_traintest_9_8591e0.mat', 'Botswana_gt_traintest_10_996e67.mat' ];
# datasettrainingfiles = [ 'KSC_gt_traintest_1_6061b3.mat', 'KSC_gt_traintest_2_c4043d.mat', 'KSC_gt_traintest_3_db432b.mat', 'KSC_gt_traintest_4_95e0ef.mat', 'KSC_gt_traintest_5_3d7a8e.mat', 'KSC_gt_traintest_6_2a60db.mat', 'KSC_gt_traintest_7_ae63a4.mat', 'KSC_gt_traintest_8_b128c8.mat', 'KSC_gt_traintest_9_9ed856.mat', 'KSC_gt_traintest_10_548b31.mat' ];

# datasettrainingfiles{7} = { 'Salinas_gt_traintest_p05_1_4228ee.mat', 'Salinas_gt_traintest_p05_2_eb1804.mat', 'Salinas_gt_traintest_p05_3_fad367.mat', 'Salinas_gt_traintest_p05_4_8cb8a3.mat', 'Salinas_gt_traintest_p05_5_d2384b.mat', 'Salinas_gt_traintest_p05_6_e34195.mat', 'Salinas_gt_traintest_p05_7_249774.mat', 'Salinas_gt_traintest_p05_8_f772c1.mat', 'Salinas_gt_traintest_p05_9_371ee5.mat', 'Salinas_gt_traintest_p05_10_22b46b.mat' };
# datasettrainingfiles{8} = { 'Smith_gt_traintest_p05_1_dd77f9.mat', 'Smith_gt_traintest_p05_2_e75152.mat', 'Smith_gt_traintest_p05_3_c8e897.mat', 'Smith_gt_traintest_p05_4_e2bd4d.mat', 'Smith_gt_traintest_p05_5_59815b.mat', 'Smith_gt_traintest_p05_6_316c37.mat', 'Smith_gt_traintest_p05_7_6aef72.mat', 'Smith_gt_traintest_p05_8_c24907.mat', 'Smith_gt_traintest_p05_9_3c2737.mat', 'Smith_gt_traintest_p05_10_75deb4.mat' };


def tang_run_accs():
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Pavia_center_right.mat'))
    # data = mat_contents['Pavia_center_right'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Pavia_center_right_gt.mat'))
    # labels = mat_contents['Pavia_center_right_gt']

    # datasettrainingfiles = [ 'Pavia_center_right_gt_traintest_1_c23379.mat', 'Pavia_center_right_gt_traintest_2_555d38.mat', 'Pavia_center_right_gt_traintest_3_436123.mat', 'Pavia_center_right_gt_traintest_4_392727.mat', 'Pavia_center_right_gt_traintest_5_da2b6f.mat', 'Pavia_center_right_gt_traintest_6_9848f9.mat', 'Pavia_center_right_gt_traintest_7_2e4963.mat', 'Pavia_center_right_gt_traintest_8_12c92f.mat', 'Pavia_center_right_gt_traintest_9_7593be.mat', 'Pavia_center_right_gt_traintest_10_30cc68.mat' ];
    # tang_run_acc(data, labels, traintestfilenames=datasettrainingfiles[1:])

    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'KSC.mat'))
    data = pxnn.remove_intensity_gaps_in_chans(mat_contents['KSC'].astype(np.float32))
    data = pxnn.normalize_channels(data)
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'KSC_gt.mat'))
    labels = mat_contents['KSC_gt']

    datasettrainingfiles = [ 'KSC_gt_traintest_1_6061b3.mat', 'KSC_gt_traintest_2_c4043d.mat', 'KSC_gt_traintest_3_db432b.mat', 'KSC_gt_traintest_4_95e0ef.mat', 'KSC_gt_traintest_5_3d7a8e.mat', 'KSC_gt_traintest_6_2a60db.mat', 'KSC_gt_traintest_7_ae63a4.mat', 'KSC_gt_traintest_8_b128c8.mat', 'KSC_gt_traintest_9_9ed856.mat', 'KSC_gt_traintest_10_548b31.mat' ];
    tang_run_acc(data, labels, traintestfilenames=datasettrainingfiles[1:])

    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Smith.mat'))
    # data = mat_contents['Smith'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Smith_gt.mat'))
    # labels = mat_contents['Smith_gt']

    # datasettrainingfiles = [ 'Smith_gt_traintest_p05_1_dd77f9.mat', 'Smith_gt_traintest_p05_2_e75152.mat', 'Smith_gt_traintest_p05_3_c8e897.mat', 'Smith_gt_traintest_p05_4_e2bd4d.mat', 'Smith_gt_traintest_p05_5_59815b.mat', 'Smith_gt_traintest_p05_6_316c37.mat', 'Smith_gt_traintest_p05_7_6aef72.mat', 'Smith_gt_traintest_p05_8_c24907.mat', 'Smith_gt_traintest_p05_9_3c2737.mat', 'Smith_gt_traintest_p05_10_75deb4.mat' ];
    # tang_run_acc(data, labels, traintestfilenames=datasettrainingfiles[1:])


    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Salinas_corrected.mat'))
    # data = mat_contents['salinas_corrected'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Salinas_gt.mat'))
    # labels = mat_contents['salinas_gt']

    # datasettrainingfiles = [ 'Salinas_gt_traintest_p05_1_4228ee.mat', 'Salinas_gt_traintest_p05_2_eb1804.mat', 'Salinas_gt_traintest_p05_3_fad367.mat', 'Salinas_gt_traintest_p05_4_8cb8a3.mat', 'Salinas_gt_traintest_p05_5_d2384b.mat', 'Salinas_gt_traintest_p05_6_e34195.mat', 'Salinas_gt_traintest_p05_7_249774.mat', 'Salinas_gt_traintest_p05_8_f772c1.mat', 'Salinas_gt_traintest_p05_9_371ee5.mat', 'Salinas_gt_traintest_p05_10_22b46b.mat' ];
    # tang_run_acc(data, labels, traintestfilenames=datasettrainingfiles[1:])

    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Botswana.mat'))
    # data = mat_contents['Botswana'].astype(np.float32)
    # data /= np.max(np.abs(data))
    # mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'Botswana_gt.mat'))
    # labels = mat_contents['Botswana_gt']

    # traintestfilenames = [ 'Botswana_gt_traintest_1_e24fae.mat', 'Botswana_gt_traintest_2_518c23.mat', 'Botswana_gt_traintest_3_7b7b6a.mat', 'Botswana_gt_traintest_4_588b5a.mat', 'Botswana_gt_traintest_5_60813e.mat', 'Botswana_gt_traintest_6_05a6b3.mat', 'Botswana_gt_traintest_7_fbba81.mat', 'Botswana_gt_traintest_8_a083a4.mat', 'Botswana_gt_traintest_9_8591e0.mat', 'Botswana_gt_traintest_10_996e67.mat' ];
    # tang_run_acc(data, labels, traintestfilenames=traintestfilenames[:1])

if __name__ == '__main__':
    tang_run_accs()

    # pdb.set_trace()

    # [class_accs, oa, aa, kappa] = experiment_acc(struct('path', '/scratch0/ilya/locDoc/data/hyperspec'), 'Smith_gt_traintest_p05_1_dd77f9.mat_WST3D_expt.mat')

