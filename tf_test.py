
from collections import namedtuple
import itertools

import numpy as np
import tensorflow as tf
import scipy.io as sio

import windows as win

import pdb

layerO = namedtuple('layerO', ['strides', 'padding'])

def scat3d(x, win_params, layer_params):
    """
    Args:
        x is input with dim (batch, depth, height, width, channels)
        win_params.filters is complex with dim (depth, height, width, channels)
    """
    real1 = tf.layers.conv3d(
        x,
        win_params.nfilt,
        win_params.kernel_size,
        strides=layer_params.strides,
        padding=layer_params.padding,
        dilation_rate=(1, 1, 1),
        activation=None,
        use_bias=False,
        kernel_initializer=tf.constant_initializer(win_params.filters.real, dtype=tf.float32),
        trainable=False,
        name=None
    )

    imag1 = tf.layers.conv3d(
        x,
        win_params.nfilt,
        win_params.kernel_size,
        strides=layer_params.strides,
        padding=layer_params.padding,
        dilation_rate=(1, 1, 1),
        activation=None,
        use_bias=False,
        kernel_initializer=tf.constant_initializer(win_params.filters.imag, dtype=tf.float32),
        trainable=False,
        name=None
    )

    return tf.abs(tf.complex(real1, imag1))

def tang_feat(data):
    x = np.pad(data, ((9,9),(9,9),(9,9)), 'wrap')

    kernel_size = [7,7,7]
    max_scale = 2
    K = 1

    psiO = win.tang_psi_factory(max_scale, K, kernel_size)
    phiO = win.tang_phi_window_3D(max_scale, kernel_size)

    # x = tf.random_normal([19,19,218])

    def model_fn(x, psiO, phiO):
        """
        Args:
            x: padded input
        Output:
            center pixel feature vector
        """
        x = tf.transpose(x, [2, 0, 1])

        x = tf.expand_dims(x, 0)
        x = tf.expand_dims(x, -1)

        layer1_params = layerO((1,1,1), 'valid')
        U1 = scat3d(x, psiO, layer1_params)

        # swap channels with batch
        U1 = tf.transpose(U1, [4, 1, 2, 3, 0])

        layer2_params = layerO((1,1,1), 'valid')
        U2 = scat3d(U1, psiO, layer2_params)

        # merge channels with batch
        U2 = tf.transpose(U2, [0, 4, 1, 2, 3])
        U2 = tf.reshape(U2, [U2.shape[0]*U2.shape[1], U2.shape[2], U2.shape[3], U2.shape[4]])
        U2 = tf.expand_dims(U2, -1)

        # convolve with phis
        layer2_lo_params = layerO((1,1,1), 'valid')
        S2 = scat3d(U2, phiO, layer2_lo_params)

        S1 = scat3d(U1[:,3:-3, 3:-3, 3:-3, :], phiO, layer2_lo_params)
        S0 = scat3d(x[:,6:-6, 6:-6, 6:-6, :], phiO, layer2_lo_params)

        S2 = tf.reshape(S2, [S2.shape[0] * S2.shape[1]])
        S1 = tf.reshape(S1, [S1.shape[0] * S1.shape[1]])
        S0 = tf.reshape(S0, [S0.shape[1]])

        return tf.concat([S0,S1,S2],0)

    pixels = np.array(list(itertools.product(range(data.shape[0]),range(data.shape[1]))))
    for pixel in pixels:
        [pixel_x, pixel_y] = pixel
        subimg = x[pixel_x:(pixel_x+19), pixel_y:(pixel_y+19), :]
    
        # feat = model_fn(tf.Variable(subimg), psiO, phiO)

    # running
    pdb.set_trace()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    myres = sess.run(feat)


if __name__ == '__main__':
    mat_contents = sio.loadmat('/Users/artsyinc/Documents/MATH630/research/data/hyper/Indian_pines.mat')
    data = mat_contents['indian_pines'].astype(np.float32)
    data /= np.max(np.abs(data))
    tang_feat(data)
