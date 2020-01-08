import os
import h5py
import hdf5storage

import numpy as np
import scipy.io as sio
from sklearn.neighbors import KNeighborsClassifier

import pdb

# DATA_PATH = '/scratch0/ilya/locDoc/data/hyperspec'
DATA_PATH = '/Users/artsyinc/Documents/MATH630/research/data/hyper'
# DATASET_PATH = '/scratch0/ilya/locDoc/data/hyperspec/datasets'
DATASET_PATH = '/Users/artsyinc/Documents/MATH630/research/data/hyper'


# mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'PaviaU.mat'))
# data = mat_contents['paviaU'].astype(np.float32)
# data /= np.max(np.abs(data))
mat_contents = sio.loadmat(os.path.join(DATASET_PATH, 'PaviaU_gt.mat'))
labels = mat_contents['paviaU_gt']
traintestfilename = 'PaviaU_gt_traintest_coarse_32px32p.mat'

mat_contents = None
try:
    mat_contents = sio.loadmat(os.path.join(DATA_PATH, traintestfilename))
except:
    mat_contents = hdf5storage.loadmat(os.path.join(DATA_PATH, traintestfilename))
train_mask = mat_contents['train_mask'].astype(int).squeeze()
test_mask = mat_contents['test_mask'].astype(int).squeeze()

flat_labels = labels.transpose().reshape(height*width)

trainX = np.array(filter(lambda (x,y): labels[y,x]*train_mask[x*height+y] != 0, all_pixels))
trainY = flat_labels[train_mask==1]
testX = np.array(filter(lambda (x,y): labels[y,x]*test_mask[x*height+y] != 0, all_pixels))
testY = flat_labels[test_mask==1]

pdb.set_trace()

print('train')
clasifier = KNeighborsClassifier(n_neighbors=1)
print('test')
classifier.fit(trainX, trainY)
y_pred = classifier.predict(testX)

acc = (y_pred == testY)

