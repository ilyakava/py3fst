from __future__ import print_function

import time
from collections import namedtuple
import itertools

from math import ceil
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn.functional as F

import h5py
from tqdm import tqdm

import pdb

# dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor  # Uncomment this to run on GPU

winO = namedtuple('winO', ['nfilt', 'filters', 'filter_params'])

def gabor_window_factory_3D(Ms):
    M1, M2, M3 = Ms[0], Ms[1], Ms[2]

    x1 = np.tile(np.reshape(np.array(range(1, M1 + 1)), [M1, 1, 1]), [1, M2, M3])
    x2 = np.tile(np.reshape(np.array(range(1, M2 + 1)), [1, M2, 1]), [M1, 1, M3])
    x3 = np.tile(np.reshape(np.array(range(1, M3 + 1)), [1, 1, M3]), [M1, M2, 1])

    nfilt = int(np.prod(np.ceil((Ms - 1) / 2.) + 1))

    # out_channels x in_channels x kT x kH x kW
    # a real part channel and an imaginary part channel
    filters = np.zeros((nfilt, 2, M1, M2, M3))
    filter_params = np.zeros((nfilt, 3))

    winOp = 0
    for m1 in range(0, 1 + int(ceil((Ms[0] - 1) / 2.))):
        for m2 in range(0, 1 + int(ceil((Ms[1] - 1) / 2.))):
            for m3 in range(0, 1 + int(ceil((Ms[2] - 1) / 2.))):
                complexfilt = np.exp(2 * np.pi * 1j * (m1 / float(M1) * x1 + \
                    m2 / float(M2) * x2 + m3 / float(M3) * x3)) / np.linalg.norm(Ms);
                filters[winOp, 0, :, :, :] = complexfilt.real
                filters[winOp, 1, :, :, :] = complexfilt.imag
                filter_params[winOp, :] = [m1, m2, m3];
                winOp = winOp + 1;
    return winO(nfilt, filters, filter_params)

def tang_phi_window_3D(J,x,y,b):
    phi = np.exp(-9*(x**2 + y**2 + b**2) / 2**(2*J+3))
    lamdaJ = 1 / np.linalg.norm(phi)
    return phi * lamdaJ

def tang_psi_window_3D_flat(scale, nu, kappa, kernel_size):
    """
    Args:
        kernel_size: a tuple of filter size (x,y,b)
    """
    x_pts = np.linspace(0, kernel_size[0]-1, kernel_size[0]) - (kernel_size[0]-1)/2.0
    y_pts = np.linspace(0, kernel_size[1]-1, kernel_size[1]) - (kernel_size[1]-1)/2.0
    b_pts = np.linspace(0, kernel_size[2]-1, kernel_size[2]) - (kernel_size[2]-1)/2.0

    coords = np.array(list(itertools.product(x_pts, y_pts, b_pts)))

    kernel_pts = np.zeros((coords.shape[0],), dtype=np.complex64)
    for coord_i, coord in enumerate(coords):
        x,y,b = coords[coord_i]
        kernel_pts[coord_i] = tang_psi_window_3D_coordinate(scale, nu, kappa,x,y,b)
    return [kernel_pts, coords]

def tang_window_factory_3D(J, angles, H, W, B):
    x = np.tile(np.arrange(H), (1, W, B))
    y = np.tile(np.arrange(W), (H, 1, B))
    b = np.tile(np.arrange(B), (H, W, 1))
    
    fourier_filters[0, :,:,:] = np.fft(tang_phi_window_3D(J,x,y,b))
    fp = 1
    for j in range(J):
        for nu in angles:
            for kappa in angles:
                fourier_filters[fp, :,:,:] = np.fft(tang_psi_window_3D(j, nu, kappa,x,y,b))
                fp += 1
    filter_params = np.zeros((fp, 3)) # TODO
    return winO(fp, fourier_filters, filter_params)

def main():
    hf = h5py.File('/scratch0/ilya/locDoc/data/hyperspec/features/np_data.h5', 'w')
    # see http://christopherlovell.co.uk/blog/2016/04/27/h5py-intro.html

    Mss = np.array([[9,9,4], [9,9,16], [9,9,4]])
    strides = ((1,1,4), (1,1,16), (1,1,4))
    paddings = ((4,4,2), (4,4,8), (4,4,2))
    H, W, D = 1096, 492, 102
    hyper = autograd.Variable(torch.randn(1, 1, H, W, D).type(dtype))
    Phi = np.empty((H*W, 0), dtype='single')

    # paddings = np.ceil((Mss - 1) / 2.).astype(int)

    # prepare filters
    winO1 = gabor_window_factory_3D(Mss[0,:])
    M1filt = autograd.Variable(torch.from_numpy(winO1.filters).type(dtype))
    winO2 = gabor_window_factory_3D(Mss[1,:])
    M2filt = autograd.Variable(torch.from_numpy(winO2.filters).type(dtype))
    winO3 = gabor_window_factory_3D(Mss[2,:])
    M3filt = autograd.Variable(torch.from_numpy(winO3.filters[0:1,:,:,:,:]).type(dtype))

    for i in range(0, winO1.nfilt):
        i1 = (i*2);
        i2 = i1 + 2;
        tmp1 = F.conv3d(hyper, M1filt[i1:i2,:,:,:,:], None, strides[0], paddings[0])
        pdb.set_trace()
        tmp1 = tmp1 * tmp1;
        out1 = torch.sum(tmp1, dim=1, keepdim=True)
        out1 = torch.sqrt(out1)
        del tmp1
        if i == 0:
            Phi = np.append(Phi, (out1.view(H*W,out1.size(4)).data).cpu().numpy(), axis=1)
            del out1
        else:
            for j in range(0, winO2.nfilt):
                j1 = (j*2);
                j2 = j1 + 2;
                tmp2 = F.conv3d(out1, M2filt[j1:j2,:,:,:,:], None, strides[1], paddings[1])
                tmp2 = tmp2 * tmp2;
                out2 = torch.sum(tmp2, dim=1, keepdim=True)
                out2 = torch.sqrt(out2)
                del tmp2
                if j == 0:
                    Phi = np.append(Phi, (out2.view(H*W,out2.size(4)).data).cpu().numpy(), axis=1)
                    del out2
                else:
                    tmp3 = F.conv3d(out2, M3filt, None, strides[2], paddings[2])
                    tmp3 = tmp3 * tmp3;
                    out3 = torch.sum(tmp3, dim=1, keepdim=True)
                    out3 = torch.sqrt(out3)
                    del tmp3
                    Phi = np.append(Phi, (out3.view(H*W,out3.size(4)).data).cpu().numpy(), axis=1)
                    del out3

    hf.create_dataset('feats', data=Phi)
    hf.close()

# import numpy as np
import matplotlib.pyplot as plt

class IndexTracker(object):
    """For scrolling through layers of 3d vis
    https://matplotlib.org/2.1.2/gallery/animation/image_slices_viewer.html

    Example usage:

    X = np.real(cube)
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, X)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
    """
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()



if __name__ == '__main__':
    print("Hello dropbox")
    # cube = tang_psi_window_3D(1, 1*np.pi/3, 1*np.pi/3, [7,7,7])
    # cube = np.imag(cube)
    # pyplot_slices(cube[:,:,3], cube[:,3,:], cube[3,:,:])



# if __name__ == '__main__':

#     catfilt = np.concatenate((winO.filters.real, winO.filters.imag))

#     filters = autograd.Variable(torch.from_numpy(catfilt[0:75, :,:,:,:]).type(dtype))

#     H, W, D = 1096, 492, 102

#     inputs = autograd.Variable(torch.randn(1, 1, H, W, D).type(dtype))
#     start = time.time()
#     out = F.conv3d(inputs, filters, None, (1, 1, 4), (4, 4, 2))
#     end = time.time()
#     print(end - start)
#     print('done')
