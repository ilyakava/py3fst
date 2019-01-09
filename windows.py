"""Filter creation.

3D especially for FST and WST
"""

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


def tang_psi_factory(J, K, kernel_size, min_scale=0):
    """
    Note how scale is the "most significant bit"
    """
    assert min_scale < J, 'min scale >= max scale'
    filter_params = np.array(list(itertools.product(range(min_scale, J),range(K),range(K))))
    nfilt = filter_params.shape[0]
    filters = np.zeros(kernel_size + [nfilt], dtype=np.complex64)

    for idx, filter_param in enumerate(filter_params):
        [scale, nu, kappa] = filter_param
        filters[:,:,:,idx] = tang_psi_window_3D(scale, nu*np.pi/3, kappa*np.pi/3, kernel_size)

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

def fst3d_phi_window_3D(kernel_size):
    """
    Args:
        kernel_size: a tuple of filter size (x,y,b)
    """
    x_pts = np.linspace(1, kernel_size[0], kernel_size[0])
    y_pts = np.linspace(1, kernel_size[1], kernel_size[1])
    b_pts = np.linspace(1, kernel_size[2], kernel_size[2])

    coords = np.array(list(itertools.product(x_pts, y_pts, b_pts)))

    x_idxs = np.linspace(0, kernel_size[0]-1, kernel_size[0], dtype=int)
    y_idxs = np.linspace(0, kernel_size[1]-1, kernel_size[1], dtype=int)
    b_idxs = np.linspace(0, kernel_size[2]-1, kernel_size[2], dtype=int)

    coords_idxs = np.array(list(itertools.product(x_idxs, y_idxs, b_idxs)))

    kernel = np.zeros(kernel_size, dtype=np.complex64)
    for coord_i, coord in enumerate(coords):
        x_i,y_i,b_i = coords_idxs[coord_i]
        x,y,b = coords[coord_i]
        kernel[x_i,y_i,b_i] = fst3d_psi_window_3D_coordinate(0, 0, 0, float(x),float(y),float(b))
    S = np.linalg.norm(kernel_size)
    kernel = kernel / S

    return winO(1, np.expand_dims(kernel,-1), [[0,0,0]], kernel_size)

def fst3d_psi_factory(kernel_size, min_freq=[0,0,0]):
    """
    Args:
        min_freq:
    """
    min_freq = np.array(min_freq)
    assert np.all(min_freq >= np.array([0,0,0])), 'some min freq < 0'
    assert np.all(min_freq < np.array([1,1,1])), 'some min freq >= 1'
    filter_params = np.array(list(itertools.product(
        np.linspace(0, 1, kernel_size[0], endpoint=False),
        np.linspace(0, 1, kernel_size[1], endpoint=False),
        np.linspace(0, 1, kernel_size[2], endpoint=False)
    )))
    # never do any averaging
    filter_params = np.array(filter(lambda fp: np.all(fp > np.array([0,0,0])), filter_params))
    # remove filters with too low freq
    filter_params = np.array(filter(lambda fp: np.all(fp >= min_freq), filter_params))
    filter_params = np.array(filter(lambda fp: np.any(fp > min_freq), filter_params))
    nfilt = filter_params.shape[0]
    if nfilt == 0:
        return winO(0, None, None, kernel_size)
    else:
        filters = np.zeros(kernel_size + [nfilt], dtype=np.complex64)

        for idx, filter_param in enumerate(filter_params):
            [mdM1, mdM2, mdM3] = filter_param
            filters[:,:,:,idx] = fst3d_psi_window_3D(mdM1, mdM2, mdM3, kernel_size)

        return winO(nfilt, filters, filter_params, kernel_size)

def fst3d_psi_window_3D(m1divM1, m2divM2, m3divM3, kernel_size):
    """
    Args:
        kernel_size: a tuple of filter size (x,y,b)
    """
    x_pts = np.linspace(1, kernel_size[0], kernel_size[0])
    y_pts = np.linspace(1, kernel_size[1], kernel_size[1])
    b_pts = np.linspace(1, kernel_size[2], kernel_size[2])

    coords = np.array(list(itertools.product(x_pts, y_pts, b_pts)))

    x_idxs = np.linspace(0, kernel_size[0]-1, kernel_size[0], dtype=int)
    y_idxs = np.linspace(0, kernel_size[1]-1, kernel_size[1], dtype=int)
    b_idxs = np.linspace(0, kernel_size[2]-1, kernel_size[2], dtype=int)

    coords_idxs = np.array(list(itertools.product(x_idxs, y_idxs, b_idxs)))

    kernel = np.zeros(kernel_size, dtype=np.complex64)
    for coord_i, coord in enumerate(coords):
        x_i,y_i,b_i = coords_idxs[coord_i]
        x,y,b = coords[coord_i]
        kernel[x_i,y_i,b_i] = fst3d_psi_window_3D_coordinate(m1divM1, m2divM2, m3divM3,float(x),float(y),float(b))
    S = np.linalg.norm(kernel_size)
    return kernel / S

def fst3d_psi_window_3D_coordinate(m1divM1,m2divM2,m3divM3,x,y,b):
    return np.exp( 2*np.pi*1j*(m1divM1*x + m2divM2*y + m3divM3*b) )

def fst2d_psi_factory(kernel_size, min_freq=[0,0], include_avg=False, filt_steps_ovr=None):
    """
    Args:
        min_freq:
    """
    min_freq = np.array(min_freq)
    assert np.all(min_freq >= np.array([0,0])), 'some min freq < 0'
    assert np.all(min_freq < np.array([1,1])), 'some min freq >= 1'
    if not filt_steps_ovr:
        filt_steps_ovr = kernel_size
    filter_params = np.array(list(itertools.product(
        np.linspace(0, 1, filt_steps_ovr[0], endpoint=False),
        np.linspace(0, 1, filt_steps_ovr[1], endpoint=False)
    )))
    # never do any averaging
    if not include_avg:
        filter_params = np.array(filter(lambda fp: np.all(fp > np.array([0,0])), filter_params))
    # remove filters with too low freq
    filter_params = np.array(filter(lambda fp: np.all(fp >= min_freq), filter_params))
    filter_params = np.array(filter(lambda fp: np.any(fp > min_freq), filter_params))
    nfilt = filter_params.shape[0]
    if nfilt == 0:
        return winO(0, None, None, kernel_size)
    else:
        filters = np.zeros(kernel_size + [nfilt], dtype=np.complex64)

        for idx, filter_param in enumerate(filter_params):
            [mdM1, mdM2] = filter_param
            filters[:,:,idx] = (fst3d_psi_window_3D(mdM1, mdM2, 0, kernel_size + [1]).squeeze() * \
                np.linalg.norm(kernel_size + [1]) / np.linalg.norm(kernel_size))

        return winO(nfilt, filters, filter_params, kernel_size)

def fst2d_phi_factory(kernel_size):
    """
    """
    kernel = (fst3d_psi_window_3D(0, 0, 0, kernel_size + [1]) * \
        np.linalg.norm(kernel_size + [1]) / np.linalg.norm(kernel_size))

    return winO(1, kernel, [[0,0]], kernel_size)

import matplotlib.pyplot as plt

def show_IP_fst_filters():
    psi = fst3d_psi_factory([3,9,9])
    reshaped = np.real(np.transpose(psi.filters, [1,2,0,3]))
    fig, axes = plt.subplots(8, 16)
    for col in range(8):
        for row in range(16):
            idx = col * 16 + row
            filt_img = reshaped[:,:,:,idx]
            filt_img -= filt_img.min()
            filt_img /= filt_img.max()
            axes[col, row].imshow(reshaped[:,:,:,idx])
            axes[col, row].axis('off')

    plt.show()

def show_wave_filters():
    psi = tang_psi_factory(5, 5, [3,7,7])
    reshaped = np.real(np.transpose(psi.filters, [1,2,0,3]))
    fig, axes = plt.subplots(8, 16)
    for col in range(8):
        for row in range(16):
            idx = col * 16 + row
            if idx < 125:
                filt_img = reshaped[:,:,:,idx]
                filt_img -= filt_img.min()
                filt_img /= filt_img.max()
                axes[col, row].imshow(reshaped[:,:,:,idx])
            axes[col, row].axis('off')

    plt.show()
if __name__ == '__main__':
    show_IP_fst_filters()
