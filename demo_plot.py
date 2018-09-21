import numpy as np

import windows as win
from window_plot import pyplot_slices

import pdb

if __name__ == '__main__':
    cube = win.fst3d_psi_window_3D(0, 0, 1/7., [7,7,7])
    cube = np.imag(cube)
    # pdb.set_trace()
    pyplot_slices(cube[:,:,3], cube[:,3,:], cube[3,:,:])
    # pdb.set_trace()