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

def tang_psi_window_3D(scale, nu, kappa, kernel_size):
    """
    Args:
        kernel_size: a tuple of filter size (x,y,b)
    """
    x_pts = np.linspace(0, kernel_size[0]-1, kernel_size[0]) - (kernel_size[0]-1)/2
    y_pts = np.linspace(0, kernel_size[1]-1, kernel_size[1]) - (kernel_size[1]-1)/2
    b_pts = np.linspace(0, kernel_size[2]-1, kernel_size[2]) - (kernel_size[2]-1)/2

    coords = np.array(list(itertools.product(x_pts, y_pts, b_pts)))

    x_idxs = np.linspace(0, kernel_size[0]-1, kernel_size[0], dtype=int)
    y_idxs = np.linspace(0, kernel_size[1]-1, kernel_size[1], dtype=int)
    b_idxs = np.linspace(0, kernel_size[2]-1, kernel_size[2], dtype=int)

    coords_idxs = np.array(list(itertools.product(x_idxs, y_idxs, b_idxs)))

    kernel = np.zeros(kernel_size, dtype=np.complex64)
    for coord_i, coord in enumerate(coords):
        x_i,y_i,b_i = coords_idxs[coord_i]
        x,y,b = coords[coord_i]
        kernel[x_i,y_i,b_i] = tang_psi_window_3D_coordinate(scale, nu, kappa,x,y,b)
    return kernel


def tang_psi_window_3D_coordinate(scale, nu, kappa,x,y,b):
    """Un-normalized
    
    """
    xi = 3 * np.pi / 4
    var = (4/3)**2 # see scatwave morlet_filter_bank_1d.m
    xprime = np.cos(nu)*np.cos(kappa)*x + \
            -np.cos(nu)*np.sin(kappa)*y + \
            np.sin(nu)*b
    psi_jgamma = 2**(-2*scale) * np.exp(1j * xi * xprime * 2**(-scale) \
         - (x**2 + y**2 + b**2)/(2*var))
    # S = np.linalg.norm(psi_jgamma)
    return psi_jgamma

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

import plotly.offline as py
from plotly.graph_objs import *
import scipy.misc

pl_BrBG=[[0.0, 'rgb(84, 48, 5)'],
         [0.1, 'rgb(138, 80, 9)'],
         [0.2, 'rgb(191, 129, 45)'],
         [0.3, 'rgb(222, 192, 123)'],
         [0.4, 'rgb(246, 232, 195)'],
         [0.5, 'rgb(244, 244, 244)'],
         [0.6, 'rgb(199, 234, 229)'],
         [0.7, 'rgb(126, 203, 192)'],
         [0.8, 'rgb(53, 151, 143)'],
         [0.9, 'rgb(0, 101, 93)'],
         [1.0, 'rgb(0, 60, 48)']]

def get_the_slice(x,y,z, surfacecolor,  colorscale='Hot', showscale=False):
    # Greys,YlGnBu,Greens,YlOrRd,Bluered,RdBu,Reds,Blues,Picnic,Rainbow,Portland,Jet,Hot,Blackbody,Earth,Electric,Viridis,CividisGreys,YlGnBu,Greens,YlOrRd,Bluered,RdBu,Reds,Blues,Picnic,Rainbow,Portland,Jet,Hot,Blackbody,Earth,Electric,Viridis,Cividis
    return Surface(x=x,# https://plot.ly/python/reference/#surface
                   y=y,
                   z=z,
                   surfacecolor=surfacecolor,
                   colorscale=colorscale,
                   showscale=showscale)   
def get_lims_colors(surfacecolor):# color limits for a slice
    return np.min(surfacecolor), np.max(surfacecolor)

def egplot():
    volume=lambda x,y,z: x*np.exp(-x**2-y**2-z**2)
    
    x=np.linspace(-2,2, 50)
    y=np.linspace(-2,2, 50)
    x,y=np.meshgrid(x,y)
    z=np.zeros(x.shape)
    surfcolor_z=volume(x,y,z)

    x=np.linspace(-2,2, 50)
    z=np.linspace(-2,2, 50)
    x,z=np.meshgrid(x,z)
    y=np.zeros(x.shape)
    surfcolor_y=volume(x,y,z)

    y=np.linspace(-2,2, 50)
    z=np.linspace(-2,2, 50)
    y,z=np.meshgrid(y,z)
    x=np.zeros(y.shape)
    surfcolor_x=volume(x,y,z)
    
    pyplot_slices(surfcolor_y, surfcolor_z.transpose(), surfcolor_x.transpose())

def pyplot_slices(surfcolor_z, surfcolor_y,surfcolor_x,resample_factor=50):
    """
    Args:
        surfcolor_z: img that has x on vertical, y on horiz
        surfcolor_y: img that has z on vertical, x on horiz
        surfcolor_x: img that has z on vertical, y on horiz
    """
    surfcolor_z = scipy.misc.imresize(surfcolor_z, resample_factor*np.array(surfcolor_z.shape), interp='nearest')
    surfcolor_y= scipy.misc.imresize(surfcolor_y, resample_factor*np.array(surfcolor_y.shape), interp='nearest')
    surfcolor_x = scipy.misc.imresize(surfcolor_x, resample_factor*np.array(surfcolor_x.shape), interp='nearest')

    x=np.linspace(-1,1, surfcolor_z.shape[0])
    y=np.linspace(-1,1, surfcolor_z.shape[1])
    y,x=np.meshgrid(y,x)
    z=np.zeros(x.shape)
    slice_z=get_the_slice(x,y,z, surfcolor_z)    
    
    x=np.linspace(-1,1, surfcolor_y.shape[1])
    z=np.linspace(-1,1, surfcolor_y.shape[0])
    x,z=np.meshgrid(x,z)
    y=np.zeros(x.shape)
    slice_y=get_the_slice(x,y,z, surfcolor_y)

    y=np.linspace(-1,1, surfcolor_x.shape[1])
    z=np.linspace(-1,1, surfcolor_x.shape[0])
    y,z=np.meshgrid(y,z)
    x=np.zeros(z.shape)
    slice_x=get_the_slice(x,y,z, surfcolor_x)

    sminz, smaxz=get_lims_colors(surfcolor_z)
    sminy, smaxy=get_lims_colors(surfcolor_y)
    sminx, smaxx=get_lims_colors(surfcolor_x)
    vmin=min([sminz, sminy, sminx])
    vmax=max([smaxz, smaxy, smaxx])

    # slice_z.update(cmin=vmin, cmax=vmax)
    # slice_y.update(cmin=vmin, cmax=vmax)
    # slice_x.update(cmin=vmin, cmax=vmax, showscale=True)
    slice_x.update(showscale=True)

    axis = dict(showbackground=True, 
            backgroundcolor="rgb(230, 230,230)",
            gridcolor="rgb(255, 255, 255)",      
            zerolinecolor="rgb(255, 255, 255)",  
            )


    layout = Layout(
             title='Slices in volumetric data', 
             width=700,
             height=700,
             scene=Scene(xaxis=XAxis(axis),
                         yaxis=YAxis(axis), 
                         zaxis=ZAxis(axis), 
                         aspectratio=dict(x=1,
                                          y=1, 
                                          z=1
                                         ),
                        )
            )

    fig=Figure(data=Data([slice_z,slice_y,slice_x]), layout=layout)
    # pdb.set_trace()
    py.plot(fig, filename='Slice-volumetric-2.html')

if __name__ == '__main__':
    cube = tang_psi_window_3D(1, np.pi/3, np.pi/3, [7,7,7])
    cube = np.real(cube)
    pyplot_slices(cube[:,:,3], cube[:,3,:].transpose(), cube[3,:,:].transpose())
    # egplot()
    # pdb.set_trace()



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
