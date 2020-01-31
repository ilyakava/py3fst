"""Does the compute intensive tsne embedding for an HSI image.
"""

import numpy as np
from sklearn.manifold import TSNE

import os
import scipy.io as sio

import pdb


DATASET_PATH = '/scratch0/ilya/locDoc/data/hyperspec/datasets'
DATASET_PATH = '/fs/vulcan-scratch/ilyak/locDoc/data/hyperspec'
dataset_name, data_struct_field_name = ['Salinas_corrected.mat', 'salinas_corrected']
dataset_name, data_struct_field_name = ['Indian_pines_corrected.mat', 'indian_pines_corrected']
dataset_name, data_struct_field_name = ['Botswana.mat', 'Botswana']
dataset_name, data_struct_field_name = ['KSC_corrected.mat', 'KSC']
# dataset_name, data_struct_field_name = ['HoustonU.mat', 'HoustonU']
# ^  would take at least 360 hours

mat_contents = sio.loadmat(os.path.join(DATASET_PATH, dataset_name))
data = mat_contents[data_struct_field_name].astype(np.float32)
X = data.reshape((-1,data.shape[2]))

tsne_fitter = TSNE(n_components=3,verbose=4,init='pca', n_jobs=16)
# ^ requires sklearn v >= 0.22
tsne_embed_full = tsne_fitter.fit_transform(X)
img = tsne_embed_full.reshape((data.shape[0], data.shape[1], 3))

np.savez('%s/%s.npz' % (DATASET_PATH, data_struct_field_name), tsne=img)

