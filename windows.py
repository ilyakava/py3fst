"""Filter creation.

3D especially for FST and WST
"""

from collections import namedtuple
from enum import Enum
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

def gabor_psi_factory(kernel_size):
    """
    """
    filter_params = np.array(list(itertools.product(
        np.linspace(0, 1, kernel_size[0], endpoint=False),
        np.linspace(0, 1, kernel_size[1], endpoint=False),
        np.linspace(0, 1, kernel_size[2], endpoint=False)
    )))
   
    nfilt = filter_params.shape[0]
    
    filters = np.zeros(kernel_size + [nfilt], dtype=np.complex64)

    for idx, filter_param in enumerate(filter_params):
        [mdM1, mdM2, mdM3] = filter_param
        filters[:,:,:,idx] = fst3d_psi_window_3D(mdM1, mdM2, mdM3, kernel_size)

    return winO(nfilt, filters, filter_params, kernel_size)

def fst3d_psi_factory(kernel_size, min_freq=[0,0,0]):
    """
    Args:
        min_freq:
    """
    min_freq = np.array(min_freq)
    assert np.all(min_freq >= np.array([0,0,0])), 'some min freq < 0'
    assert np.all(min_freq < np.array([1,1,1])), 'some min freq >= 1'
    filter_params = list(itertools.product(
        np.linspace(0, 1, kernel_size[0], endpoint=False),
        np.linspace(0, 1, kernel_size[1], endpoint=False),
        np.linspace(0, 1, kernel_size[2], endpoint=False)
    ))
    # never do any averaging
    filter_params = list(filter(lambda fp: np.all(fp > np.array([0,0,0])), filter_params))
    # remove filters with too low freq
    filter_params = list(filter(lambda fp: np.all(fp >= min_freq), filter_params))
    filter_params = list(filter(lambda fp: np.any(fp > min_freq), filter_params))
    nfilt = filter_params.shape[0]
    if nfilt == 0:
        return winO(0, None, None, kernel_size)
    else:
        filters = np.zeros(kernel_size + [nfilt], dtype=np.complex64)

        for idx, filter_param in enumerate(filter_params):
            [mdM1, mdM2, mdM3] = filter_param
            filters[:,:,:,idx] = fst3d_psi_window_3D(mdM1, mdM2, mdM3, kernel_size)

        return winO(nfilt, filters, np.array(filter_params), kernel_size)

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

def dlrgf_window_3D_coordinate(omegavec, sigmavec, x,y,b):
    """
    page 1384, bottom left column
    """
    sx, sy, sb = sigmavec
    omegax,omegay,omegab = omegavec
    return 1 / ((2*np.pi)**1.5 *sx*sy*sb) * np.exp( -( (x/sx)**2 + (y/sy)**2 + (b/sb)**2)/2.0 ) * np.cos(omegax*x) * np.cos(omegay*y) * np.sin(omegab*b)


def params2omegas(modomega, phi, theta):
    """
    Returns:
      omegax,omegay,omegab
    
    page 1383, bottom left column
    """
    return (modomega * np.sin(phi)*np.cos(theta),modomega * np.sin(phi)*np.sin(theta),  modomega * np.cos(phi))

def dlrgf_window_3D(omegavec, sigmavec, kernel_size):
    sx, sy, sb = sigmavec
    # we will sample within 2 std deviations bc 95% of the energy is there
    x_pts = np.linspace(-2*sx, 2*sx, kernel_size[0])
    y_pts = np.linspace(-2*sy, 2*sy, kernel_size[1])
    b_pts = np.linspace(-2*sb, 2*sb, kernel_size[2])

    coords = np.array(list(itertools.product(x_pts, y_pts, b_pts)))

    x_idxs = np.linspace(0, kernel_size[0]-1, kernel_size[0], dtype=int)
    y_idxs = np.linspace(0, kernel_size[1]-1, kernel_size[1], dtype=int)
    b_idxs = np.linspace(0, kernel_size[2]-1, kernel_size[2], dtype=int)

    coords_idxs = np.array(list(itertools.product(x_idxs, y_idxs, b_idxs)))

    kernel = np.zeros(kernel_size, dtype=np.float32)
    for coord_i, coord in enumerate(coords):
        x_i,y_i,b_i = coords_idxs[coord_i]
        x,y,b = coords[coord_i]
        kernel[x_i,y_i,b_i] = dlrgf_window_3D_coordinate(omegavec, sigmavec, float(x),float(y),float(b))
    # 
    return kernel / np.linalg.norm(kernel)

def dlrgf_factory(kernel_size, sigmavec):
    """    
    """    
    nfilt = 52
    # page 1388 middle left column
    params = list(itertools.product([np.pi/2, np.pi/4, np.pi/8, np.pi/16], [0, np.pi/4, np.pi/2, 3*np.pi/4], [0, np.pi/4, np.pi/2, 3*np.pi/4]))
    omegavecs = list(set([params2omegas(*param_) for param_ in params]))
    assert len(omegavecs) == nfilt, 'Inconsistent number of filters'
    
    filters = np.zeros(kernel_size + [nfilt], dtype=np.float)
    

    for idx, omegavec in enumerate(omegavecs):
        filters[:,:,:,idx] = dlrgf_window_3D(omegavec, sigmavec, kernel_size)

    return winO(nfilt, filters, omegavecs, kernel_size)


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
    filter_params = list(itertools.product(
        np.linspace(0, 1, filt_steps_ovr[0], endpoint=False),
        np.linspace(0, 1, filt_steps_ovr[1], endpoint=False)
    ))
    
    # never do any averaging
    if not include_avg:
        filter_params = list(filter(lambda fp: np.all(fp > np.array([0,0])), filter_params))
    # remove filters with too low freq
    filter_params = list(filter(lambda fp: np.all(fp >= min_freq), filter_params))
    filter_params = list(filter(lambda fp: np.any(fp > min_freq), filter_params))
    nfilt = len(filter_params)
    if nfilt == 0:
        return winO(0, None, None, kernel_size)
    else:
        filters = np.zeros(kernel_size + [nfilt], dtype=np.complex64)

        for idx, filter_param in enumerate(filter_params):
            [mdM1, mdM2] = filter_param
            filters[:,:,idx] = (fst3d_psi_window_3D(mdM1, mdM2, 0, kernel_size + [1]).squeeze() * \
                np.linalg.norm(kernel_size + [1]) / np.linalg.norm(kernel_size))

        return winO(nfilt, filters, np.array(filter_params), kernel_size)

def fst2d_phi_factory(kernel_size):
    """
    """
    kernel = (fst3d_psi_window_3D(0, 0, 0, kernel_size + [1]) * \
        np.linalg.norm(kernel_size + [1]) / np.linalg.norm(kernel_size))

    return winO(1, kernel, [[0,0]], kernel_size)
    
class FilterType(Enum):
    LOWPASS = 1
    BANDPASS = 2
    HIGHPASS = 3
    
def gen_cort(freq, l=32, sr=1000//8, filt_type=FilterType.BANDPASS):
    """
    sr: (SRT) frames per second, related to frame_length
    """
    # nsltools function, cortical filter in temporal dimension
    t = np.arange(l, dtype=float) / sr * freq
    h = np.sin(2*np.pi*t) * t**2 * np.exp(-3.5*t) * freq;
    h -= h.mean()
    H0 = np.fft.fft(h, 2*l)
    A = np.angle(H0[:l]);
    H = np.abs(H0[:l])
    maxHi = np.argmax(H)
    maxH = H[maxHi]
    H /= maxH
    
    if filt_type is FilterType.LOWPASS:
        H[:maxHi] = 1
    elif filt_type is FilterType.HIGHPASS:
        H[(maxHi+1):] = 1
              
    return H * np.exp(1j*A)
    
def gen_corf(freq, l=32, sr=24, filt_type=FilterType.BANDPASS):
    """
    sr: (SRF)
    """
    # nsltools function, cortical filter in freq dimension
    r1 = np.arange(l, dtype=float) / l * sr / 2 / freq
    r1 = r1**2
    H = r1 * np.exp(1-r1)
    maxHi = np.argmax(H)
    sumH = H.sum()
    
    if filt_type is FilterType.LOWPASS:
        H[:maxHi] = 1
        H = H / H.sum() * sumH
    elif filt_type is FilterType.HIGHPASS:
        H[(maxHi+1):] = 1
        H = H / H.sum() * sumH
            
    return H

def cortical_2x1d_FDomain_factory(rv, sv, l):
    """
    in schematc nsltools uses params:
    [4, 8, 16, 32], [.25, .5, 1, 2, 4, 8], 128
    
    Doesn't add additional filters by modifying for +/- phase.
    """
    filters = np.zeros((len(sv),len(rv),2,l), dtype=np.complex64)
    for rdx, rate in enumerate(rv):
        for sdx, scale in enumerate(sv):

            rate_filt_type = FilterType.BANDPASS
            if rate == min(rv):
                rate_filt_type = FilterType.LOWPASS
            elif rate == max(rv):
                rate_filt_type = FilterType.HIGHPASS
            scale_filt_type = FilterType.BANDPASS
            if scale == min(sv):
                scale_filt_type = FilterType.LOWPASS
            elif scale == max(sv):
                scale_filt_type = FilterType.HIGHPASS

            R1 = gen_cort(rate, l, filt_type=rate_filt_type)
            R2 = gen_corf(scale, l, filt_type=scale_filt_type)

            filters[sdx,rdx,0,:] = R1
            filters[sdx,rdx,1,:] = R2
    return filters
    
def cortical_window(rate, scale, sign=1, l=32, rate_filt_type=FilterType.BANDPASS, scale_filt_type=FilterType.BANDPASS):
    """
    rate = t_freq
    scale = f_freq
    """
    R1 = gen_cort(rate, l, filt_type=rate_filt_type);
    r1 = np.fft.ifft(R1, l*2);
    r1 = r1[:l];

    # % scale response
    R2 = gen_corf(scale, l/2, filt_type=scale_filt_type);
    r2 = np.fft.ifft(R2, l);
    r2 = np.roll(r2, l//2)

    r1 = np.expand_dims(r1,0)
    r2 = np.expand_dims(r2,-1)

    if sign == -1:
        r1 = r1.conj()

    r = np.dot(r2, r1);
    return r
    
def cortical_psi_factory(rv, sv, l):
    """
    in schematc nsltools uses params:
    [4, 8, 16, 32], [.25, .5, 1, 2, 4, 8], 128
    """
    params = np.array(list(itertools.product(rv, sv, [1,-1])))
    nfilt = len(params)
    filters = np.zeros((l,l,nfilt), dtype=np.complex64)
    for i, param in enumerate(params):
        rate, scale, sign = param
        rate_filt_type = FilterType.BANDPASS
        if rate == min(rv):
            rate_filt_type = FilterType.LOWPASS
        elif rate == max(rv):
            rate_filt_type = FilterType.HIGHPASS
        scale_filt_type = FilterType.BANDPASS
        if scale == min(sv):
            scale_filt_type = FilterType.LOWPASS
        elif scale == max(sv):
            scale_filt_type = FilterType.HIGHPASS
        filters[:,:,i] = cortical_window(rate, scale, sign, l, rate_filt_type, scale_filt_type)
    return winO(nfilt, filters, params, np.array([l,l]))

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
