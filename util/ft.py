import math

import numpy as np

def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))
def prev_power_of_2(x):
    return 1 if x == 0 else 2**(math.ceil(math.log2(x))-1)
def closest_power_of_2(x):
    n = next_power_of_2(x)
    p = prev_power_of_2(x)
    if abs(x-n) < abs(x-p):
        return n
    else:
        return p
        
def normalize_0_1(values, m=None, M=None):
    """deep-voice-conversion
    """
    m = m if m is not None else values.min()
    M = M if M is not None else values.max()
    normalized = np.clip((values - m) / (M - m), 0, 1)
    return normalized
    
def unnormalize_0_1(normalized, m, M):
    """deep-voice-conversion
    """
    return (np.clip(normalized, 0, 1) * (M - m)) + m
    
def dct_filters(n_filters, n_input):
    """http://man.hubwiz.com/docset/LibROSA.docset/Contents/Resources/Documents/_modules/librosa/filters.html
    """
    basis = np.empty((n_filters, n_input))
    basis[0, :] = 1.0 / np.sqrt(n_input)

    samples = np.arange(1, 2*n_input, 2) * np.pi / (2.0 * n_input)

    for i in range(1, n_filters):
        basis[i, :] = np.cos(i*samples) * np.sqrt(2.0/n_input)

    return basis

def power_law_compression(data, alpha=0.3):
    """Stevens's power law
    Done separately for positive/negative numbers
    """
    sgn = np.sign(data)
    return sgn * np.abs(data)**alpha
    