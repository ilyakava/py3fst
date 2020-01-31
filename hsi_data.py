"""Data functions for HSI data
"""

import itertools
import operator
import os
import random

import h5py
import hdf5storage
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from tqdm import tqdm

import pdb

DATASET_PATH = '/scratch0/ilya/locDoc/data/hyperspec/datasets'
PAD_TYPE = 'symmetric'

nclass_dict = {
    'PaviaU': 9,
    'PaviaCR': 9,
    'Botswana': 14,
    'KSC': 13,
    'IP': 16
}
dset_dims = {
    'PaviaU.mat': (610, 340, 103),
    'Botswana.mat': (1476, 256, 145),
    'KSC_corrected.mat': (512, 614, 176),
    'Indian_pines_corrected.mat': (145, 145, 200)
}
# in train, label order
dset_filenames_dict = {
    'PaviaU': ('PaviaU.mat', 'PaviaU_gt.mat'),
    'PaviaCR': ('Pavia_center_right.mat', 'Pavia_center_right_gt.mat'),
    'Botswana': ('Botswana.mat', 'Botswana_gt.mat'),
    'KSC': ('KSC_corrected.mat', 'KSC_gt.mat'),
    'IP': ('Indian_pines_corrected.mat', 'Indian_pines_gt.mat'),
}
# in train label order
dset_fieldnames_dict = {
    'PaviaU': ('paviaU', 'paviaU_gt'),
    'PaviaCR': ('Pavia_center_right', 'Pavia_center_right_gt'),
    'Botswana': ('Botswana', 'Botswana_gt'),
    'KSC': ('KSC', 'KSC_gt'),
    'IP': ('indian_pines_corrected', 'indian_pines_gt')
}


def load_data(trainimgname, trainimgfield, trainlabelname, trainlabelfield):
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, trainimgname))
    data = mat_contents[trainimgfield].astype(np.float32)
    data /= np.max(np.abs(data))
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, trainlabelname))
    labels = mat_contents[trainlabelfield]
    return data, labels

def load_labels(trainlabelname, trainlabelfield):
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, trainlabelname))
    labels = mat_contents[trainlabelfield]
    return labels
    
def multiversion_matfile_get_field(fname, field, dtype=int):
    mat_contents = None
    try:
        mat_contents = sio.loadmat(fname)
    except:
        mat_contents = hdf5storage.loadmat(fname)
    return mat_contents[field].astype(dtype).squeeze()

def get_train_val_splits(data, labels, train_mask, val_mask, addl_padding=(4,4,0), n_eval=int(1e12)):
    """Apply train/val masks on data cube.
    """
    
    [height, width, nbands] = data.shape
    print('Loaded %i x %i x %i (h,w,b) image' % (height, width, nbands))

    all_pixels = np.array(list(itertools.product(range(width),range(height))))

    ap = np.array(addl_padding)
    assert np.all(ap % 2 == 0), 'Assymetric is not supported'
    
    padded_data = np.pad(data, ((ap[0]//2,ap[0]//2),(ap[1]//2,ap[1]//2),(ap[2]//2,ap[2]//2)), PAD_TYPE)
    

    train_pixels = np.array(filter(lambda (x,y): labels[y,x]*train_mask[x*height+y] != 0, all_pixels))
    val_pixels = np.array(filter(lambda (x,y): labels[y,x]*val_mask[x*height+y] != 0, all_pixels))
    

    train_pixels_list = train_pixels.tolist()
    random.shuffle(train_pixels_list)
    train_pixels = np.array(train_pixels_list)
    

    val_pixels_list = val_pixels.tolist()
    random.shuffle(val_pixels_list)
    val_pixels = np.array(val_pixels_list)

    print("Train / Validation split is %i / %i" % (train_pixels.shape[0], val_pixels.shape[0]))

    batch_item_shape = tuple(map(operator.add, addl_padding, (1,1,data.shape[2])))

    trainX = np.zeros((train_mask.sum(),) + batch_item_shape, dtype=np.float32)
    trainY = np.zeros((train_mask.sum(),))
    n_eval = min(n_eval, val_mask.sum())
    valX = np.zeros((n_eval,) + batch_item_shape, dtype=np.float32)
    valY = np.zeros((n_eval,))


    for pixel_i, pixel in enumerate(tqdm(train_pixels[:,:], desc='Loading train data: ')):
        # this iterates through columns first
        [pixel_x, pixel_y] = pixel
        trainX[pixel_i,:,:,:] = padded_data[pixel_y:(pixel_y+ap[0]+1), pixel_x:(pixel_x+ap[1]+1), :]
        trainY[pixel_i] = labels[pixel_y,pixel_x] - 1

    for pixel_i, pixel in enumerate(tqdm(val_pixels[:n_eval,:], desc='Loading val data: ', total=n_eval)):
        # this iterates through columns first
        [pixel_x, pixel_y] = pixel
        valX[pixel_i,:,:,:] = padded_data[pixel_y:(pixel_y+ap[0]+1), pixel_x:(pixel_x+ap[1]+1), :]
        valY[pixel_i] = labels[pixel_y,pixel_x] - 1

    return trainX, trainY, valX, valY

def tupsum(*args):
    """Adds any number of tuples together as if they were vectors.
    """
    def tupsum_(tuple1, tuple2):
        if len(tuple1) != len(tuple2):
            raise ValueError('Tuples must be of same length to be summed')
        return tuple(map(operator.add, tuple1, tuple2))
    return reduce(lambda x, y: tupsum_(x,y), args)

def pca_embedding(data, n_components=3):
    """Collapses dimension of data cube with PCA.
    """
    h, w, b = data.shape
    pca = PCA(n_components=n_components)
    pca.fit(data.reshape((-1,b)))
    print("First %i components of PCA explain %.2f Percent of the energy." % (n_components, pca.explained_variance_ratio_.sum()*100))
    pc = pca.transform(data.reshape((-1,b)))
    norm_pc = (pc - pc.min(axis=0)) / (pc.max(axis=0) - pc.min(axis=0))
    return norm_pc.reshape((h,w,n_components))
    