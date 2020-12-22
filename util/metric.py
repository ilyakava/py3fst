import numpy as np

def sisnr(y, yhat, axis=(0,)):
    """Scale invariant sdr
    """
    scaling = (np.abs(y) / np.abs(yhat)).mean(axis=axis, keepdims=True)
    diff = (y - scaling*yhat)
    return -10*np.log10(( diff*np.conj(diff) ).sum() / ( y*np.conj(y) ).sum())

def snr(y, yhat):
    """sdr
    """
    diff = (y - yhat)
    return -10*np.log10(( diff*np.conj(diff) ).sum() / ( y*np.conj(y) ).sum())
