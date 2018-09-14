
from collections import namedtuple
import itertools

import numpy as np

import pdb

winO = namedtuple('winO', ['nfilt', 'filters', 'filter_params', 'kernel_size'])


def tang_phi_window_3D(J, kernel_size):
    x_pts = np.linspace(0, kernel_size[0]-1, kernel_size[0]) - (kernel_size[0]-1)/2.0
    y_pts = np.linspace(0, kernel_size[1]-1, kernel_size[1]) - (kernel_size[1]-1)/2.0
    b_pts = np.linspace(0, kernel_size[2]-1, kernel_size[2]) - (kernel_size[2]-1)/2.0

    coords = np.array(list(itertools.product(x_pts, y_pts, b_pts)))

    x_idxs = np.linspace(0, kernel_size[0]-1, kernel_size[0], dtype=int)
    y_idxs = np.linspace(0, kernel_size[1]-1, kernel_size[1], dtype=int)
    b_idxs = np.linspace(0, kernel_size[2]-1, kernel_size[2], dtype=int)

    coords_idxs = np.array(list(itertools.product(x_idxs, y_idxs, b_idxs)))

    def phi(x,y,b):
        return np.exp(-9*(x**2 + y**2 + b**2) / 2**(2*J+3))

    kernel = np.zeros(kernel_size, dtype=np.complex64)
    for coord_i, coord in enumerate(coords):
        x_i,y_i,b_i = coords_idxs[coord_i]
        x,y,b = coords[coord_i]
        kernel[x_i,y_i,b_i] = phi(x,y,b)
    lamdaJ = 1 / np.linalg.norm(kernel)
    kernel = kernel * lamdaJ

    return winO(1, np.expand_dims(kernel,-1), [[J]], kernel_size)


def tang_psi_factory(J, K, kernel_size):
    filter_params = np.array(list(itertools.product(range(J),range(K),range(K))))
    nfilt = filter_params.shape[0]
    filters = np.zeros(kernel_size + [nfilt], dtype=np.complex64)

    for idx, filter_param in enumerate(filter_params):
        [scale, nu, kappa] = filter_param
        filters[:,:,:,idx] = tang_psi_window_3D(scale, nu*np.pi/3, kappa*np.pi/3, [7,7,7])

    return winO(nfilt, filters, filter_params, kernel_size)

def tang_psi_window_3D(scale, nu, kappa, kernel_size):
    """
    Args:
        kernel_size: a tuple of filter size (x,y,b)
    """
    x_pts = np.linspace(0, kernel_size[0]-1, kernel_size[0]) - (kernel_size[0]-1)/2.0
    y_pts = np.linspace(0, kernel_size[1]-1, kernel_size[1]) - (kernel_size[1]-1)/2.0
    b_pts = np.linspace(0, kernel_size[2]-1, kernel_size[2]) - (kernel_size[2]-1)/2.0

    coords = np.array(list(itertools.product(x_pts, y_pts, b_pts)))

    x_idxs = np.linspace(0, kernel_size[0]-1, kernel_size[0], dtype=int)
    y_idxs = np.linspace(0, kernel_size[1]-1, kernel_size[1], dtype=int)
    b_idxs = np.linspace(0, kernel_size[2]-1, kernel_size[2], dtype=int)

    coords_idxs = np.array(list(itertools.product(x_idxs, y_idxs, b_idxs)))

    kernel = np.zeros(kernel_size, dtype=np.complex64)
    for coord_i, coord in enumerate(coords):
        x_i,y_i,b_i = coords_idxs[coord_i]
        x,y,b = coords[coord_i]
        kernel[x_i,y_i,b_i] = tang_psi_window_3D_coordinate(float(scale), float(nu), float(kappa),float(x),float(y),float(b))
    S = np.linalg.norm(kernel)
    return kernel / S


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

