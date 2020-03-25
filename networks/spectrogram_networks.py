from collections import namedtuple

import tensorflow as tf

from networks.highway import highway_block

layerO = namedtuple('layerO', ['strides', 'padding'])

def amazon_net(x_dict, dropout, reuse, is_training, n_classes, args):
    spec_h = args.feature_height # ~freq
    spec_w = args.network_example_length // args.hop_length # time

    hidden_units = 64
    bottleneck_size = hidden_units // 2

    context_chunks = 31
    context_chunk_size = spec_w // context_chunks
    
    with tf.variable_scope('amazon_net', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        out = tf.transpose(x, [0,2,1]) # batch, time, depth

        # prenet
        out = tf.layers.dense(out, units=hidden_units*2, activation=tf.nn.relu)
        out = tf.layers.dropout(out, rate=dropout, training=is_training)
        out = tf.layers.dense(out, units=hidden_units, activation=tf.nn.relu)
        out = tf.layers.dropout(out, rate=dropout, training=is_training)
        
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))

        # bottleneck
        out = tf.layers.dense(out, units=bottleneck_size, activation=None, name="bottleneck")

        # context window
        # [-1, t, C]
        out = tf.reshape(out, [-1, spec_w // 2, bottleneck_size*2])

        for i in range(6):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_classifier_{}'.format(i))

        # get classification
        out = tf.layers.dense(out, n_classes)
        out = tf.squeeze(out, axis=-1) # [-1, t//2]

    return out


def st_net_v1(x, dropout, reuse, is_training, n_classes, args):
    """Network to follow ST preprocessing.
    
    x should be (...)
    """
    
    sz = 13
    psi = win.fst2d_psi_factory([sz, sz], include_avg=False)
    
    layer_params = layerO((1,1), 'valid')
    nfeat = 32

    spec_h = args.feature_height
    spec_w = args.network_example_length // args.hop_length
    
    with tf.variable_scope('wst_net_v1', reuse=reuse):
        # x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        x = tf.expand_dims(x, -1)

        ### bs, h, w, channels
        U1 = scat2d(x, psi, layer_params)
        h = U1.shape[1]
        w = U1.shape[2]
        ### bs, h, w, time varying, frequency varying
        U1 = tf.reshape(U1, [-1, h, w, sz-1, sz-1])
        ### bs, time varying, frequency varying, h, w
        U1 = tf.transpose(U1, [0,3,4,1,2])

        ds = (sz-1)//2
        rategram = tf.layers.max_pooling3d(U1, (1,ds,1), (1,ds,1), padding='same')
        scalegram = tf.layers.max_pooling3d(U1, (ds,1,1), (ds,1,1), padding='same')

        nsz = (sz-1)**2 // ds
        rategram = tf.reshape(rategram, [-1, nsz, h, w])
        scalegram = tf.reshape(scalegram, [-1, nsz, h, w])

        cortical = tf.concat([rategram, scalegram], axis=1)
        cortical = tf.transpose(cortical, [0,2,3,1])
        
        conv = tf.layers.conv2d(cortical, nfeat, (7,1), 1, activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, nfeat, (1,7), 1, activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, nfeat, 5, 1, activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, nfeat, 5, 2, activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, nfeat, 5, 4, activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, nfeat, 5, 8, activation=tf.nn.relu)
        
        fc = tf.contrib.layers.flatten(conv)
        fc = tf.layers.dense(fc, 300)
        out = tf.layers.dense(fc, n_classes)
    return tf.squeeze(out, axis=1)
