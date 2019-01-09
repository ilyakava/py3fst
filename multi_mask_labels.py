"""Used for simmyride
"""

import os
from PIL import Image
import numpy as np
import numpngw

import pdb

def separate_labels(npim, label_list, outpath):
    """
    For converting the classifier results to unreal masks
    """
    for i,label in enumerate(label_list):
        out = np.zeros(npim.shape)
        out[npim == i+1] = 65535
        out = out.astype(np.uint16)
        numpngw.write_png('%s.png' % os.path.join(outpath, label), out)

train_size_per_class = 400

def combine_masks(img_list, outfile):
    """
    For making the training set
    """
    im=Image.open(img_list[0]).convert('L')
    acc = np.zeros(np.array(im, dtype=np.float32).shape)
    for i, filename in enumerate(img_list):
        im=Image.open(filename).convert('L')
        npim = np.array(im, dtype=np.float32) / 255.0
        lab_idxs = np.array(np.where(npim != 0))
        subset_idxs = np.random.choice(lab_idxs.shape[1], train_size_per_class, replace=False)
        for y_i,x_i in lab_idxs[:,subset_idxs].T:
            acc[y_i,x_i] = i+1
    np.save(outfile, acc)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

datasets = {
    'ballew': {
        'classes': ['buildings', 'dirt', 'grass', 'roads', 'trees', 'water'],
        'cmap': ListedColormap(['black', 'brown', 'green', 'gray', 'white', 'blue'])
    }
}

def plot_labels(dataset_name, data):

    #discrete color scheme
    cMap = datasets[dataset_name]['cmap']

    #data
    fig, ax = plt.subplots()
    heatmap = ax.imshow(data, cmap=cMap)

    #legend
    cbar = plt.colorbar(heatmap)

    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(datasets[dataset_name]['classes']):
        cbar.ax.text(.5, (2 * j + 1) / 8.0, lab, ha='center', va='center')
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('# of contacts', rotation=270)


    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()

    #labels
    column_labels = list('ABCD')
    row_labels = list('WXYZ')
    ax.set_xticklabels(column_labels, minor=False)
    ax.set_yticklabels(row_labels, minor=False)

    plt.show()

def save_materials():
    arr = np.load('/Users/artsyinc/Documents/simmyride/data/materials/second/resized/svm.npy')
    outpath = '/Users/artsyinc/Documents/simmyride/data/materials/second/resized/full_materials'
    separate_labels(arr.reshape(2017,2017), datasets['ballew']['classes'], outpath)

def view():
    arr = np.load('/Users/artsyinc/Documents/simmyride/data/materials/second/resized/svm.npy')
    # pdb.set_trace()
    plot_labels('ballew', arr.reshape(2017,2017))

def main():
    path = '/Users/artsyinc/Documents/simmyride/data/materials/second/resized'
    filenames = ['mask-buildings.png', 'mask-dirt.png', 'mask-grass.png', 'mask-roads.png', 'mask-trees.png', 'mask-water.png']
    fullfilenames = [os.path.join(path, fname) for fname in filenames]
    combine_masks(fullfilenames, os.path.join(path, 'labels'))


if __name__ == '__main__':
    main()
