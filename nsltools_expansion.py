import numpy as np
import scipy

import nsltools as nslpy
from windows import cortical_2x1d_FDomain_factory

def get_cfs(l=128):
    return cortical_2x1d_FDomain_factory([4, 8, 16, 32], [.25, .5, 1, 2, 4, 8], l)

def fft_forward_cortical(cfs, input_fft):
    """fft forward cortical
    
    python implementation of aud2cor in nsltools matlab
    """
    l = cfs.shape[3]
    cort = np.zeros((cfs.shape[0], 2*cfs.shape[1], l, l), dtype=np.complex)
    for sdx in range(cfs.shape[0]):
        for rdx in range(cfs.shape[1]):
            for sign in [1,-1]:
                R1 = cfs[sdx,rdx,0]
                R2 = cfs[sdx,rdx,1]

                if sign == 1:
                    R1 = np.pad(R1, (0,l))
                else:
                    R1 = np.pad(R1, (0,l))
                    R1 = [R1[0]] + list(np.conj(R1[1:])[::-1])
                    R1[l] = np.abs(R1[l+1])
                    R1 = np.array(R1)

                R1 = np.expand_dims(R1, -1)
                cort_0_0 = input_fft * R1

                R2 = np.expand_dims(R2, 0)
                cort_0_0 = cort_0_0 * R2

                cort_0_0 = scipy.fft.ifft2(cort_0_0, (2*l,2*l))
                
                cort_0_0 = cort_0_0[:l,:l]

                cort[sdx,rdx + int(sign==1)*cfs.shape[1]] = cort_0_0
    return cort

def forward_cortical(cfs, audspec):
    """forward cortical in time domain
    """
    l = cfs.shape[3]
    cort = np.zeros((cfs.shape[0], 2*cfs.shape[1], audspec.shape[0], audspec.shape[1]), dtype=np.complex)
    for sdx in range(cfs.shape[0]):
        for rdx in range(cfs.shape[1]):
            for sign in [1,-1]:
                R1 = cfs[sdx,rdx,0]
                R2 = cfs[sdx,rdx,1]

                if sign == 1:
                    R1 = np.pad(R1, (0,l))
                else:
                    R1 = np.pad(R1, (0,l))
                    R1 = [R1[0]] + list(np.conj(R1[1:])[::-1])
                    R1[l] = np.abs(R1[l+1])
                    R1 = np.array(R1)

                r1 = scipy.fft.ifft(R1, l*2, 0)
                r1 = np.expand_dims(r1,-1)
                
                z1t_py  = scipy.signal.convolve(audspec, np.roll(r1,l), mode='same')

                r2 = scipy.fft.ifft(R2, l*2)
                r2 = np.expand_dims(r2,0)
                
                cort_0_0  = scipy.signal.convolve(z1t_py, np.roll(r2,l), mode='same')
                
                cort[sdx,rdx + int(sign==1)*cfs.shape[1]] = cort_0_0
    return cort
    
def fft_backward_cortical(cfs, input_cort):
    """fft backward cortical
    
    python implementation of cor2aud in nsltools matlab
    """
    NORM = 0.9
    l = cfs.shape[3]
    audspec_fft = np.zeros((2*l, l), dtype=np.complex)
    H_cum = np.zeros((l*2, l), dtype=np.complex)
    for rdx in range(cfs.shape[1]):
        for sign in [1,-1]:
            for sdx in range(cfs.shape[0]):
                z = input_cort[sdx,rdx + int(sign==1)*cfs.shape[1]]
                z = np.pad(z,((0,0),(0,l)))
                Z = scipy.fft.fft2(z, (2*l,2*l))
                Z = Z[:,:l]
                
                
                R1 = cfs[sdx,rdx,0]
                R2 = cfs[sdx,rdx,1]
                if sign == 1:
                    R1 = np.conj(np.pad(R1, (0,l)))
                else:
                    R1 = np.conj(np.pad(R1, (0,l)))
                    R1 = [R1[0]] + list(np.conj(R1[1:])[::-1])
                    R1[l] = np.abs(R1[l+1])
                    R1 = np.array(R1)
                
                R1 = np.expand_dims(R1,-1)
                R2 = np.expand_dims(R2,0)
                H = np.dot(R1, R2);
                
                H_cum += H * np.conj(H)
                audspec_fft += H * Z
    
    H_cum[:,0] *= 2
    sumH = H_cum.sum()
    H_cum = NORM * H_cum + (1 - NORM) * H_cum.max();
    H_cum = H_cum / H_cum.sum() * sumH;
    audspec_fft = audspec_fft / H_cum
    
    audspec = scipy.fft.ifft2(audspec_fft, (2*l,2*l))
    audspec = 2*audspec[:l,:l]
                
    return audspec

def wav2aud_8kHz(y):
    return nslpy.wav2aud(y, [8,8,-2,-1])
    
def aud2wav_8kHz(audspec, yhat=np.array([])):
    return nslpy.aud2wav(audspec, yhat, [8,8,-2,-1] + [10,0,0])
