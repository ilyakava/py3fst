"""Test spatial depence of feature extraction methods

Extracts quadrants of input images such that a certain percentage of
training examples are present in those quadrants. The hope is to
select training sets where training pixels and test pixels are not
spatially interspersed.
"""
from collections import namedtuple
import itertools
import os

import h5py
import hdf5storage
import numpy as np
import scipy.io as sio
import cvxpy as cvx

import matplotlib.pyplot as plt

import pdb


# DATASET_PATH = '/Users/artsyinc/Documents/MATH630/research/data/hyper'
DATASET_PATH = (os.environ['DATASET_PATH'])

quadO = namedtuple('quadO', ['x', 'y', 'x_end', 'y_end', 'hist'])

def get_quadrants(labels, n_x, n_y):
    pass

def get_train_mask(quads, selection):
    train = np.zeros((height, width))
    for i, q in enumerate(quads):
        if selection[i] == 1:
            train[q.y:q.y_end,q.x:q.x_end] = 1
    return train
    # plt.imshow(train)
    # plt.show()

def get_gt(name):
    mat_contents = sio.loadmat(os.path.join(DATASET_PATH, name))
    if name == 'Indian_pines_gt.mat':
        return mat_contents['indian_pines_gt']
    elif name == 'PaviaU_gt.mat':
        return mat_contents['paviaU_gt']
    elif name == 'Salinas_gt.mat':
        return mat_contents['salinas_gt']
    elif name == 'Botswana_gt.mat':
        return mat_contents['Botswana_gt']
    elif name == 'Smith_gt.mat':
        return mat_contents['Smith_gt']
    elif name == 'Pavia_center_right_gt.mat':
        return mat_contents['Pavia_center_right_gt']
    else:
        error('No such dataset %s' % name)

def quads_to_traintestfile(quads, selection, gt, name):
    train = np.zeros((height, width))
    for i, q in enumerate(quads):
        if selection[i] == 1:
            train[q.y:q.y_end,q.x:q.x_end] = 1
    train_mask = (gt != 0) * train
    train_mask = train_mask.transpose().reshape(height*width)

    test_mask = (gt != 0) * (1-train)
    test_mask = test_mask.transpose().reshape(height*width)

    mat_outdata = {}
    mat_outdata[u'test_mask'] = test_mask
    mat_outdata[u'train_mask'] = train_mask
    hdf5storage.write(mat_outdata, filename=os.path.join(DATASET_PATH, name), matlab_compatible=True)

def myhist(nlabels, lab_img):
    hist_arr = np.zeros(nlabels)
    for k in lab_img.flatten():
        if k != 0:
            hist_arr[k-1] += 1
    return hist_arr

def make_staggered_quads(gt, rare_classes, q_sz, q_stagger=None):
    [height, width] = gt.shape
    nlabels = len(np.unique(gt)) - 1
    if q_stagger is None:
        q_stagger = q_sz

    quads = []
    for x1 in range(0,width-q_sz,q_stagger):
        for y1 in range(0,height-q_sz,q_stagger):
            x2 = x1 + q_sz
            y2 = y1 + q_sz
            gt_subregion = gt[y1:y2,x1:x2]
            subregion_hist = myhist(nlabels, gt_subregion)
            # if we are on a staggered block, and we don't include
            # any small amount labels, then skip it
            if (x1 % q_sz !=0 or y1 % q_sz !=0) and (np.max(rare_classes * subregion_hist) < 2):
                next
            elif np.sum(subregion_hist) > 0:
                quads.append(quadO(x1, y1, x2, y2, subregion_hist))
    return quads


def quads_to_mats(nlabels, non_empty_quads):
    n_ne_quads = len(non_empty_quads)
    P = np.zeros((nlabels, n_ne_quads))
    C = np.zeros((nlabels, n_ne_quads))
    for i, q in enumerate(non_empty_quads):
        P[:, i] = q.hist / gt_hist
        C[:, i] = q.hist
    return [P, C]

if __name__ == '__main__':

    gt = get_gt('PaviaU_gt.mat')

    [height, width] = gt.shape
    nlabels = len(np.unique(gt)) - 1
    gt_hist = myhist(nlabels, gt)

    pdb.set_trace()

    non_empty_quads = make_staggered_quads(gt, (gt_hist < np.median(gt_hist)).astype(int), 16, 4)

    print('done making quads')

    n_ne_quads = len(non_empty_quads)

    [P, C] = quads_to_mats(nlabels, non_empty_quads)

    A = 100 * P
    # b = 10
    # w = gt_hist.max() / gt_hist
    x = cvx.Variable(n_ne_quads, boolean=True)

    obj = cvx.Minimize(cvx.norm((A * x - 12),2))
    prob = cvx.Problem(obj)


    prob.solve(solver=cvx.GLPK_MI) # this is the better than default solver for integer problems

    if x.value is None:
        print('no solution found')
        pdb.set_trace()
    else:

        selection = np.round(x.value).astype(int)
        myhist(nlabels ,(gt * get_train_mask(non_empty_quads, selection)).astype(int)) / gt_hist


        plt.imshow(get_train_mask(non_empty_quads, selection))
        plt.show()
        # print what this binary mask image looks like

        # next

        pdb.set_trace()

        quads_to_traintestfile(non_empty_quads, selection, gt, 'PaviaU_gt_traintest_coarse_32px32p.mat')
