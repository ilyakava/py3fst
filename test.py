from math import ceil
import time
from collections import namedtuple
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn.functional as F

# dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor  # Uncomment this to run on GPU
inptype = torch.cuda.IntTensor

winO = namedtuple('winO', ['nfilt', 'filters', 'filter_params'])


def gabor_window_factory_3D(Ms):
    M1, M2, M3 = Ms[0], Ms[1], Ms[2]

    x1 = np.tile(np.reshape(np.array(range(1, M1 + 1)), [M1, 1, 1]), [1, M2, M3])
    x2 = np.tile(np.reshape(np.array(range(1, M2 + 1)), [1, M2, 1]), [M1, 1, M3])
    x3 = np.tile(np.reshape(np.array(range(1, M3 + 1)), [1, 1, M3]), [M1, M2, 1])

    nfilt = int(np.prod(np.ceil((Ms - 1) / 2.) + 1))

    # out_channels x in_channels x kT x kH x kW
    filters = np.zeros((nfilt, 1, M1, M2, M3), 'complex128')
    filter_params = np.zeros((nfilt, 3))

    winOp = 0
    for m1 in range(0, 1 + int(ceil((Ms[0] - 1) / 2.))):
        for m2 in range(0, 1 + int(ceil((Ms[1] - 1) / 2.))):
            for m3 in range(0, 1 + int(ceil((Ms[2] - 1) / 2.))):
                filters[winOp, 0, :, :, :] = np.exp(2 * np.pi * 1j * (m1 / float(M1) * x1 + \
                                                                      m2 / float(M2) * x2 + m3 / float(
                    M3) * x3)) / np.linalg.norm(Ms);
                filter_params[winOp, :] = [m1, m2, m3];
                winOp = winOp + 1;
    return winO(nfilt, filters, filter_params)


if __name__ == '__main__':
    winO = gabor_window_factory_3D(np.array([9, 9, 4]))
    catfilt = np.concatenate((winO.filters.real, winO.filters.imag))

    filters = autograd.Variable(torch.from_numpy(catfilt[0:75, :,:,:,:]).type(dtype))

    H, W, D = 1096, 492, 102

    inputs = autograd.Variable(torch.randn(1, 1, H, W, D).type(dtype))
    start = time.time()
    out = F.conv3d(inputs, filters, None, (1, 1, 4), (4, 4, 2))
    end = time.time()
    print(end - start)
    print('done')
