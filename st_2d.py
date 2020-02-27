"""2D ST primitives.
"""

import tensorflow as tf

import windows as win

def scat2d(x, win_params, layer_params):
    """Single layer of 2d scattering
    Args:
        x is input with dim (batch, height, width, 1)
        win_params.filters is complex with dim ......... (depth, height, width, channels)
    """
    real1 = tf.layers.conv2d(
        x,
        win_params.nfilt,
        win_params.kernel_size[:2],
        strides=layer_params.strides,
        padding=layer_params.padding,
        dilation_rate=(1,1),
        activation=None,
        use_bias=False,
        kernel_initializer=tf.constant_initializer(win_params.filters.real, dtype=tf.float32),
        trainable=False,
        name=None
    )

    imag1 = tf.layers.conv2d(
        x,
        win_params.nfilt,
        win_params.kernel_size[:2],
        strides=layer_params.strides,
        padding=layer_params.padding,
        dilation_rate=(1,1),
        activation=None,
        use_bias=False,
        kernel_initializer=tf.constant_initializer(win_params.filters.imag, dtype=tf.float32),
        trainable=False,
        name=None
    )

    return tf.abs(tf.complex(real1, imag1))