from collections import namedtuple

import tensorflow as tf

from networks.deepvoice import cbhg, conv1d_banks, conv1d, normalize, gru
from networks.highway import highway_block
from st_2d import scat2d
import windows as win

import pdb

layerO = namedtuple('layerO', ['strides', 'padding'])

def conv_bank_1d(prenet_out, num_banks, hidden_units, norm_type, is_training):
    out = conv1d_banks(prenet_out,
                           K=num_banks,
                           num_units=hidden_units,
                           norm_type=norm_type,
                           is_training=is_training)  # (N, T, K * E / 2)

    out = tf.layers.max_pooling1d(out, 2, 1, padding="same")  # (N, T, K * E / 2)

    out = conv1d(out, hidden_units, 3, scope="conv1d_1")  # (N, T, E/2)
    out = normalize(out, type=norm_type, is_training=is_training, activation_fn=tf.nn.relu)
    out = conv1d(out, hidden_units, 3, scope="conv1d_2")  # (N, T, E/2)
    out += prenet_out  # (N, T, E/2) # residual connections
    return out
    
def prenet(out, hidden_units, is_training, dropout):
    # to go from nfft -> hidden units
    out = tf.layers.dense(out, units=hidden_units*2, activation=tf.nn.relu)
    out = tf.layers.dropout(out, rate=dropout, training=is_training)
    out = tf.layers.dense(out, units=hidden_units, activation=tf.nn.relu)
    prenet_out = tf.layers.dropout(out, rate=dropout, training=is_training)
    return prenet_out
    
def simple_prenet(out, hidden_units, is_training, dropout):
    # to go from nfft -> hidden units
    out = tf.layers.dense(out, units=hidden_units*2, activation=tf.nn.relu)
    out = tf.layers.dense(out, units=hidden_units, activation=tf.nn.relu)
    return out

def concat_rategram_scalegram(stft_mag, rv, sv, l, is_training, dropout):
    psi = win.cortical_psi_factory(rv, sv, l)
    nrate = len(rv)
    nscale = 2*len(sv)
    layer_params = layerO((1,1), 'same')

    x = tf.expand_dims(stft_mag, -1) # batch, freq, time, 1

    U1 = scat2d(x, psi, layer_params)
    h = U1.shape[1]
    w = U1.shape[2]
    ### bs, h, w, time varying, frequency varying
    U1 = tf.reshape(U1, [-1, h, w, nrate, nscale])
    ### bs, time varying, frequency varying, h, w
    U1 = tf.transpose(U1, [0,3,4,1,2])
    
    U1 = tf.layers.dropout(U1, rate=dropout, training=is_training)
    
    # rategram is max over scale varying
    rategram = tf.layers.max_pooling3d(U1, (1,nscale,1), (1,nscale,1), padding='same')
    rategram = tf.reshape(rategram, [-1, nrate, h, w])
    # scalegram is max over rate varying
    scalegram = tf.layers.max_pooling3d(U1, (nrate,1,1), (nrate,1,1), padding='same')
    scalegram = tf.reshape(scalegram, [-1, nscale, h, w])
    
    cortical = tf.concat([rategram, scalegram], axis=1)
    cortical = tf.transpose(cortical, [0,2,3,1])
    # batch, freq, time, channels
    return cortical

def CBHG_net(x_dict, dropout, reuse, is_training, n_classes, spec_h, spec_w):
    num_banks = 8
    hidden_units = 64
    num_highway_blocks = 4
    norm_type = 'ins'
    
    with tf.variable_scope('CBHG', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        out = tf.transpose(x, [0,2,1]) # batch, time, depth

        with tf.variable_scope('prenet', reuse=reuse):
            # out = prenet(out, hidden_units, is_training, dropout)
            out = simple_prenet(out, hidden_units, is_training, dropout)
        # with tf.variable_scope('conv_bank_1d', reuse=reuse):
        #     out = conv_bank_1d(out, num_banks, hidden_units, norm_type, is_training)
        
        # highway
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))
        
        # gru
        with tf.variable_scope('gru', reuse=reuse):
            out = gru(out, hidden_units, True)  # (N, T, E)
                   
        # get classification
        with tf.variable_scope('classification', reuse=reuse):
            out = tf.layers.dense(out, n_classes)
            if n_classes == 1:
                out = tf.squeeze(out, axis=2)

    return out
    
def HG_net(x_dict, dropout, reuse, is_training, n_classes, args):
    # for phoneme pretraining for Guo_Li_net
    
    spec_h = args.feature_height # ~freq
    spec_w = args.network_feature_width # time
    
    num_banks = 8
    hidden_units = 64
    
    with tf.variable_scope('HG', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        out = tf.transpose(x, [0,2,1]) # batch, time, depth

        with tf.variable_scope('prenet', reuse=reuse):
            out = simple_prenet(out, hidden_units, is_training, dropout)
        
        # highway
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))
        
        # gru
        with tf.variable_scope('gru', reuse=reuse):
            out = gru(out, hidden_units, True)  # (N, T, E)
                   
        # get classification
        with tf.variable_scope('classification', reuse=reuse):
            out = tf.layers.dense(out, n_classes)
            if n_classes == 1:
                out = tf.squeeze(out, axis=2) # [-1, t//2]

    return out
    
def CBHBG_net(x_dict, dropout, reuse, is_training, n_classes, args):
    spec_h = args.feature_height # ~freq
    spec_w = args.network_feature_width # time
    
    num_banks = 8
    hidden_units = 64
    num_highway_blocks = 4
    norm_type = 'ins'
    bottleneck_size = hidden_units // 2
    
    with tf.variable_scope('amazon_net', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        out = tf.transpose(x, [0,2,1]) # batch, time, depth

        out = prenet(out, hidden_units, is_training, dropout)
        out = conv_bank_1d(out, num_banks, hidden_units, norm_type, is_training)
        
        # highway
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))

        # bottleneck
        out = tf.layers.dense(out, units=bottleneck_size, activation=None, name="bottleneck")

        # context window
        # [-1, t, C]
        out = tf.reshape(out, [-1, spec_w // 2, bottleneck_size*2])
        
        # gru
        out = gru(out, hidden_units, True)  # (N, T, E)
                   
        # get classification
        out = tf.layers.dense(out, n_classes)
        if n_classes == 1:
            out = tf.squeeze(out, axis=2) # [-1, t//2]

    return out
    
def CBHBH_net(x_dict, dropout, reuse, is_training, n_classes, args):
    """
    1-D convolution bank + highway network, bottleneck, highway network
    """
    spec_h = args.feature_height # ~freq
    spec_w = args.network_feature_width # time
    
    num_banks = 8 # 16 in tacotron
    hidden_units = 64 # 128 in tacotron
    num_highway_blocks = 4
    norm_type = 'ins'
    
    bottleneck_size = hidden_units // 2
    
    with tf.variable_scope('CBHBH', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        out = tf.transpose(x, [0,2,1]) # batch, time, depth

        with tf.variable_scope('prenet', reuse=reuse):
            out = prenet(out, hidden_units, is_training, dropout)
        with tf.variable_scope('conv_bank_1d', reuse=reuse):
            out = conv_bank_1d(out, num_banks, hidden_units, norm_type, is_training)
        
        # highway
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))

        # bottleneck
        out = tf.layers.dense(out, units=bottleneck_size, activation=None, name="bottleneck")

        # context window
        # [-1, t, C]
        out = tf.reshape(out, [-1, spec_w // 2, bottleneck_size*2])

        # classifier highway
        for i in range(6):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_classifier_{}'.format(i))
                   
        # get classification
        out = tf.layers.dense(out, n_classes)
        if n_classes == 1:
            out = tf.squeeze(out, axis=2) # [-1, t//2]

    return out

def amazon_net(x_dict, dropout, reuse, is_training, n_classes, args):
    """
    1-D convolution bank + highway network, bottleneck, highway network
    """
    spec_h = args.feature_height # ~freq
    spec_w = args.network_feature_width # time
    
    num_banks = 8 # 16 in tacotron
    hidden_units = 64 # 128 in tacotron
    num_highway_blocks = 4
    norm_type = 'ins'
    
    bottleneck_size = hidden_units // 2
    
    with tf.variable_scope('amazon_net', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        out = tf.transpose(x, [0,2,1]) # batch, time, depth

        out = prenet(out, hidden_units, is_training, dropout)
        
        # highway
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))

        # bottleneck
        out = tf.layers.dense(out, units=bottleneck_size, activation=None, name="bottleneck")

        # context window
        # [-1, t, C]
        out = tf.reshape(out, [-1, spec_w // 2, bottleneck_size*2])

        # classifier highway
        for i in range(6):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_classifier_{}'.format(i))
                   
        # get classification
        out = tf.layers.dense(out, n_classes)
        if n_classes == 1:
            out = tf.squeeze(out, axis=2) # [-1, t//2]
    return out
    
def Guo_Li_net(x_dict, dropout, reuse, is_training, n_classes, spec_h, spec_w):
    """
    prenet + highway network, bottleneck, highway network
    
    Args:
        int spec_h: ~freq
        int spec_h: time
    """
        
    num_banks = 8 # 16 in tacotron
    hidden_units = 64 # 128 in tacotron
    norm_type = 'ins'
    n_right = 10
    n_left = 20
    
    bottleneck_size = 28
    
    with tf.variable_scope('CBHBH', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        out = tf.transpose(x, [0,2,1]) # batch, time, depth

        with tf.variable_scope('prenet', reuse=reuse):
            out = simple_prenet(out, hidden_units, is_training, dropout)
        
        # highway
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))

        # bottleneck
        out = tf.layers.dense(out, units=bottleneck_size, activation=None, name="bottleneck")

        # context window
        # input is [-1, spec_w, bottleneck_size]
        # output is [-1, spec_w-30, bottleneck_size*31]
        contexts = []
        for i in range(1,n_left+1):
            contexts.append( out[:,n_left-i:-(n_right+i),:] )
        contexts.append( out[:,n_left:-n_right,:] )
        for i in range(1,n_right):
            contexts.append( out[:,n_left+i:-n_right+i,:] )
        contexts.append( out[:,n_left+n_right:,:] )
        
        out = tf.concat(contexts, axis=2)
        new_num_units = (n_right+n_left+1) * bottleneck_size

        # classifier highway
        for i in range(6):
            out = highway_block(out, num_units=new_num_units, scope='highwaynet_classifier_{}'.format(i))
                   
        # get classification
        out = tf.layers.dense(out, n_classes)
        if n_classes == 1:
            out = tf.squeeze(out, axis=1) # [-1]

    return out
    
def Guo_Li_complex_projection_net(x_dict, dropout, reuse, is_training, n_classes, spec_h, spec_w):
    """
    Complex projection
    prenet + highway network, bottleneck, highway network
    
    Args:
        int spec_h: ~freq
        int spec_h: time
        
    c.f. Complex Linear Projection (CLP): A Discriminative Approach to Joint Feature Extraction and Acoustic Modeling
        Ehsan Variani, Tara N. Sainath, Izhak Shafran, Michiel Bacchiani
    """
        
    num_banks = 8 # 16 in tacotron
    hidden_units = 64 # 128 in tacotron
    norm_type = 'ins'
    n_right = 10
    n_left = 20
    
    bottleneck_size = 28
    
    with tf.variable_scope('CBHBH', reuse=reuse):
        z = x_dict['spectrograms']
        z = tf.reshape(z, (-1,spec_h,spec_w))
        x_r = z[:,:(spec_h//2), :]
        x_i = z[:, (spec_h//2):, :]
        
        x_r = tf.transpose(x_r, [0,2,1]) # batch, time, depth
        x_i = tf.transpose(x_i, [0,2,1]) # batch, time, depth
        
        with tf.variable_scope('complex_projection', reuse=reuse):
            wrxr = tf.layers.dense(x_r, units=128, activation=None, name='wr')
            wixi = tf.layers.dense(x_i, units=128, activation=None, name='wi')
            
            wrxi = tf.layers.dense(x_i, units=128, activation=None, name='wr', reuse=True)
            wixr = tf.layers.dense(x_r, units=128, activation=None, name='wi', reuse=True)
            
            ry = wrxr - wixi
            iy = wrxi + wixr
        
            out = tf.pow(tf.pow(ry, 2) + tf.pow(iy, 2), 0.5)

        with tf.variable_scope('prenet', reuse=reuse):
            out = simple_prenet(out, hidden_units, is_training, dropout)
        
        # highway
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))

        # bottleneck
        out = tf.layers.dense(out, units=bottleneck_size, activation=None, name="bottleneck")

        # context window
        # input is [-1, spec_w, bottleneck_size]
        # output is [-1, spec_w-30, bottleneck_size*31]
        contexts = []
        for i in range(1,n_left+1):
            contexts.append( out[:,n_left-i:-(n_right+i),:] )
        contexts.append( out[:,n_left:-n_right,:] )
        for i in range(1,n_right):
            contexts.append( out[:,n_left+i:-n_right+i,:] )
        contexts.append( out[:,n_left+n_right:,:] )
        
        out = tf.concat(contexts, axis=2)
        new_num_units = (n_right+n_left+1) * bottleneck_size

        # classifier highway
        for i in range(6):
            out = highway_block(out, num_units=new_num_units, scope='highwaynet_classifier_{}'.format(i))
                   
        # get classification
        out = tf.layers.dense(out, n_classes)
        if n_classes == 1:
            out = tf.squeeze(out, axis=1) # [-1]

    return out

def cortical_net_v0(x_dict, dropout, reuse, is_training, n_classes, spec_h, spec_w):
    """
    1-D convolution bank + highway network, bottleneck, highway network
    
    Args:
        int spec_h: ~freq
        int spec_h: time
    """
    rv = [4, 8, 16, 32]
    sv = [.25, .5, 1, 2, 4, 8]
    ncort = len(rv) + 2*len(sv)
    filter_size = 32

    num_banks = 8 # 16 in tacotron
    hidden_units = 64 # 128 in tacotron
    norm_type = 'ins'
    n_right = 10
    n_left = 20
    
    bottleneck_size = 28
    
    with tf.variable_scope('CBHBH', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        
        cortical = concat_rategram_scalegram(x, rv, sv, filter_size, is_training, dropout) # batch, freq, time, chan
        
        out = tf.transpose(cortical, [0,2,1,3])
        out = tf.reshape(out, [-1, spec_w, spec_h * ncort])

        # batch, time, depth
         
        with tf.variable_scope('prenet', reuse=reuse):
            out = simple_prenet(out, hidden_units, is_training, dropout)
        
        
        # highway
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))

        # bottleneck
        out = tf.layers.dense(out, units=bottleneck_size, activation=None, name="bottleneck")

        # context window
        # input is [-1, spec_w, bottleneck_size]
        # output is [-1, spec_w-30, bottleneck_size*31]
        contexts = []
        for i in range(1,n_left+1):
            contexts.append( out[:,n_left-i:-(n_right+i),:] )
        contexts.append( out[:,n_left:-n_right,:] )
        for i in range(1,n_right):
            contexts.append( out[:,n_left+i:-n_right+i,:] )
        contexts.append( out[:,n_left+n_right:,:] )
        
        out = tf.concat(contexts, axis=2)
        new_num_units = (n_right+n_left+1) * bottleneck_size

        # classifier highway
        for i in range(6):
            out = highway_block(out, num_units=new_num_units, scope='highwaynet_classifier_{}'.format(i))
                   
        # get classification
        out = tf.layers.dense(out, n_classes)
        if n_classes == 1:
            out = tf.squeeze(out, axis=1) # [-1]

    return out
    
def cortical_net_v0res1(x_dict, dropout, reuse, is_training, n_classes, spec_h, spec_w):
    """
    1-D convolution bank + highway network, bottleneck, highway network
    
    Args:
        int spec_h: ~freq
        int spec_h: time
    """
    rv = [4, 8, 16, 32]
    sv = [.25, .5, 1, 2, 4, 8]
    ncort = len(rv) + 2*len(sv)
    filter_size = 32

    num_banks = 8 # 16 in tacotron
    hidden_units = 64 # 128 in tacotron
    norm_type = 'ins'
    n_right = 10
    n_left = 20
    
    bottleneck_size = 28
    
    with tf.variable_scope('CBHBH', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        
        cortical = concat_rategram_scalegram(x, rv, sv, filter_size, is_training, dropout) # batch, freq, time, chan
        
        res = tf.expand_dims(x, -1) # batch, freq, time, 1
        res = tf.layers.conv2d(res, ncort, 1, 1, activation=None, name="conv1")
        cortical = cortical + res
        
        out = tf.transpose(cortical, [0,2,1,3])
        out = tf.reshape(out, [-1, spec_w, spec_h * ncort])

        # batch, time, depth
         
        with tf.variable_scope('prenet', reuse=reuse):
            out = simple_prenet(out, hidden_units, is_training, dropout)
        
        
        # highway
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))

        # bottleneck
        out = tf.layers.dense(out, units=bottleneck_size, activation=None, name="bottleneck")

        # context window
        # input is [-1, spec_w, bottleneck_size]
        # output is [-1, spec_w-30, bottleneck_size*31]
        contexts = []
        for i in range(1,n_left+1):
            contexts.append( out[:,n_left-i:-(n_right+i),:] )
        contexts.append( out[:,n_left:-n_right,:] )
        for i in range(1,n_right):
            contexts.append( out[:,n_left+i:-n_right+i,:] )
        contexts.append( out[:,n_left+n_right:,:] )
        
        out = tf.concat(contexts, axis=2)
        new_num_units = (n_right+n_left+1) * bottleneck_size

        # classifier highway
        for i in range(6):
            out = highway_block(out, num_units=new_num_units, scope='highwaynet_classifier_{}'.format(i))
                   
        # get classification
        out = tf.layers.dense(out, n_classes)
        if n_classes == 1:
            out = tf.squeeze(out, axis=1) # [-1]

    return out
    
def cortical_net_v0res1b(x_dict, dropout, reuse, is_training, n_classes, spec_h, spec_w):
    """
    1-D convolution bank + highway network, bottleneck, highway network
    
    Args:
        int spec_h: ~freq
        int spec_h: time
    """
    rv = [4, 8, 16, 32]
    sv = [.25, .5, 1, 2, 4, 8]
    ncort = len(rv) + 2*len(sv)
    filter_size = 32

    num_banks = 8 # 16 in tacotron
    hidden_units = 64 # 128 in tacotron
    norm_type = 'ins'
    n_right = 10
    n_left = 20
    
    bottleneck_size = 28
    
    with tf.variable_scope('CBHBH', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        
        cortical = concat_rategram_scalegram(x, rv, sv, filter_size, is_training, dropout) # batch, freq, time, chan
        
        res = tf.expand_dims(x, -1) # batch, freq, time, 1
        res = tf.layers.conv2d(res, ncort, 1, 1, activation=None, name="conv1")
        res = tf.layers.dropout(res, rate=dropout, training=is_training)
        cortical = cortical + res
        
        out = tf.transpose(cortical, [0,2,1,3])
        out = tf.reshape(out, [-1, spec_w, spec_h * ncort])

        # batch, time, depth
         
        with tf.variable_scope('prenet', reuse=reuse):
            out = simple_prenet(out, hidden_units, is_training, dropout)
        
        
        # highway
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))

        # bottleneck
        out = tf.layers.dense(out, units=bottleneck_size, activation=None, name="bottleneck")

        # context window
        # input is [-1, spec_w, bottleneck_size]
        # output is [-1, spec_w-30, bottleneck_size*31]
        contexts = []
        for i in range(1,n_left+1):
            contexts.append( out[:,n_left-i:-(n_right+i),:] )
        contexts.append( out[:,n_left:-n_right,:] )
        for i in range(1,n_right):
            contexts.append( out[:,n_left+i:-n_right+i,:] )
        contexts.append( out[:,n_left+n_right:,:] )
        
        out = tf.concat(contexts, axis=2)
        new_num_units = (n_right+n_left+1) * bottleneck_size

        # classifier highway
        for i in range(6):
            out = highway_block(out, num_units=new_num_units, scope='highwaynet_classifier_{}'.format(i))
                   
        # get classification
        out = tf.layers.dense(out, n_classes)
        if n_classes == 1:
            out = tf.squeeze(out, axis=1) # [-1]

    return out
    
def cortical_net_v0res1c(x_dict, dropout, reuse, is_training, n_classes, spec_h, spec_w):
    """
    1-D convolution bank + highway network, bottleneck, highway network
    
    Args:
        int spec_h: ~freq
        int spec_h: time
    """
    rv = [4, 8, 16, 32]
    sv = [.25, .5, 1, 2, 4, 8]
    ncort = len(rv) + 2*len(sv)
    filter_size = 32

    num_banks = 8 # 16 in tacotron
    hidden_units = 64 # 128 in tacotron
    norm_type = 'ins'
    n_right = 10
    n_left = 20
    
    bottleneck_size = 28
    
    with tf.variable_scope('CBHBH', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        
        cortical = concat_rategram_scalegram(x, rv, sv, filter_size, is_training, 0.2) # batch, freq, time, chan
        
        res = tf.expand_dims(x, -1) # batch, freq, time, 1
        res = tf.layers.conv2d(res, ncort, 1, 1, activation=None, name="conv1")
        res = tf.layers.dropout(res, rate=dropout, training=is_training)
        cortical = cortical + res
        
        out = tf.transpose(cortical, [0,2,1,3])
        out = tf.reshape(out, [-1, spec_w, spec_h * ncort])

        # batch, time, depth
         
        with tf.variable_scope('prenet', reuse=reuse):
            out = simple_prenet(out, hidden_units, is_training, 0.2)
        
        
        # highway
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))

        # bottleneck
        out = tf.layers.dense(out, units=bottleneck_size, activation=None, name="bottleneck")

        # context window
        # input is [-1, spec_w, bottleneck_size]
        # output is [-1, spec_w-30, bottleneck_size*31]
        contexts = []
        for i in range(1,n_left+1):
            contexts.append( out[:,n_left-i:-(n_right+i),:] )
        contexts.append( out[:,n_left:-n_right,:] )
        for i in range(1,n_right):
            contexts.append( out[:,n_left+i:-n_right+i,:] )
        contexts.append( out[:,n_left+n_right:,:] )
        
        out = tf.concat(contexts, axis=2)
        new_num_units = (n_right+n_left+1) * bottleneck_size

        # classifier highway
        for i in range(6):
            out = highway_block(out, num_units=new_num_units, scope='highwaynet_classifier_{}'.format(i))
                   
        # get classification
        out = tf.layers.dense(out, n_classes)
        if n_classes == 1:
            out = tf.squeeze(out, axis=1) # [-1]

    return out
    
def cortical_net_v0res1cmini(x_dict, dropout, reuse, is_training, n_classes, spec_h, spec_w):
    """
    1-D convolution bank + highway network, bottleneck, highway network
    
    Args:
        int spec_h: ~freq
        int spec_h: time
    """
    rv = [4, 8, 16, 32]
    sv = [.25, .5, 1, 2, 4, 8]
    ncort = len(rv) + 2*len(sv)
    filter_size = 32

    num_banks = 8 # 16 in tacotron
    hidden_units = 64 # 128 in tacotron
    norm_type = 'ins'
    n_right = 10
    n_left = 20
    
    bottleneck_size = 28
    
    with tf.variable_scope('CBHBH', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        
        cortical = concat_rategram_scalegram(x, rv, sv, filter_size, is_training, 0.2) # batch, freq, time, chan
        
        res = tf.expand_dims(x, -1) # batch, freq, time, 1
        res = tf.layers.conv2d(res, ncort, 1, 1, activation=None, name="conv1")
        res = tf.layers.dropout(res, rate=dropout, training=is_training)
        cortical = cortical + res
        
        out = tf.transpose(cortical, [0,2,1,3])
        out = tf.reshape(out, [-1, spec_w, spec_h * ncort])

        # batch, time, depth
         
        with tf.variable_scope('prenet', reuse=reuse):
            out = tf.layers.dense(out, units=hidden_units, activation=tf.nn.relu)
            # out = simple_prenet(out, hidden_units, is_training, 0.2)
        
        
        # highway
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))

        # bottleneck
        out = tf.layers.dense(out, units=bottleneck_size, activation=None, name="bottleneck")

        # context window
        # input is [-1, spec_w, bottleneck_size]
        # output is [-1, spec_w-30, bottleneck_size*31]
        contexts = []
        for i in range(1,n_left+1):
            contexts.append( out[:,n_left-i:-(n_right+i),:] )
        contexts.append( out[:,n_left:-n_right,:] )
        for i in range(1,n_right):
            contexts.append( out[:,n_left+i:-n_right+i,:] )
        contexts.append( out[:,n_left+n_right:,:] )
        
        out = tf.concat(contexts, axis=2)
        new_num_units = (n_right+n_left+1) * bottleneck_size

        # classifier highway
        for i in range(6):
            out = highway_block(out, num_units=new_num_units, scope='highwaynet_classifier_{}'.format(i))
                   
        # get classification
        out = tf.layers.dense(out, n_classes)
        if n_classes == 1:
            out = tf.squeeze(out, axis=1) # [-1]

    return out
    
def cortical_net_approx0(x_dict, dropout, reuse, is_training, n_classes, spec_h, spec_w):
    """
    1-D convolution bank + highway network, bottleneck, highway network
    
    Args:
        int spec_h: ~freq
        int spec_h: time
    """
    rv = [4, 8, 16, 32]
    sv = [.25, .5, 1, 2, 4, 8]
    ncort = len(rv) + 2*len(sv)
    filter_size = 32

    num_banks = 8 # 16 in tacotron
    hidden_units = 64 # 128 in tacotron
    norm_type = 'ins'
    n_right = 10
    n_left = 20
    
    bottleneck_size = 28
    
    with tf.variable_scope('CBHBH', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        x = tf.expand_dims(x, -1) # batch, freq, time, 1
        
        # cortical = concat_rategram_scalegram(x, rv, sv, filter_size, is_training, dropout) # batch, freq, time, chan
        with tf.variable_scope('cortical_approx', reuse=reuse):
            ry = tf.layers.conv2d(x, ncort, filter_size, 1, activation=tf.math.square, name="real", padding='SAME')
            iy = tf.layers.conv2d(x, ncort, filter_size, 1, activation=tf.math.square, name="imag", padding='SAME')
            
            cortical = tf.pow(ry + iy, 0.5)
        
        
        
        res = tf.layers.conv2d(x, ncort, 1, 1, activation=None, name="conv1")
        res = tf.layers.dropout(res, rate=dropout, training=is_training)
        cortical = cortical + res
        
        out = tf.transpose(cortical, [0,2,1,3])
        out = tf.reshape(out, [-1, spec_w, spec_h * ncort])

        # batch, time, depth
         
        with tf.variable_scope('prenet', reuse=reuse):
            out = simple_prenet(out, hidden_units, is_training, dropout)
        
        
        # highway
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))

        # bottleneck
        out = tf.layers.dense(out, units=bottleneck_size, activation=None, name="bottleneck")

        # context window
        # input is [-1, spec_w, bottleneck_size]
        # output is [-1, spec_w-30, bottleneck_size*31]
        contexts = []
        for i in range(1,n_left+1):
            contexts.append( out[:,n_left-i:-(n_right+i),:] )
        contexts.append( out[:,n_left:-n_right,:] )
        for i in range(1,n_right):
            contexts.append( out[:,n_left+i:-n_right+i,:] )
        contexts.append( out[:,n_left+n_right:,:] )
        
        out = tf.concat(contexts, axis=2)
        new_num_units = (n_right+n_left+1) * bottleneck_size

        # classifier highway
        for i in range(6):
            out = highway_block(out, num_units=new_num_units, scope='highwaynet_classifier_{}'.format(i))
                   
        # get classification
        out = tf.layers.dense(out, n_classes)
        if n_classes == 1:
            out = tf.squeeze(out, axis=1) # [-1]

    return out
    

def cortical_net_approx1(x_dict, dropout, reuse, is_training, n_classes, spec_h, spec_w):
    """
    1-D convolution bank + highway network, bottleneck, highway network
    
    Args:
        int spec_h: ~freq
        int spec_h: time
    """
    rv = [4, 8, 16, 32]
    sv = [.25, .5, 1, 2, 4, 8]
    ncort = len(rv) + 2*len(sv)
    filter_size = 32

    num_banks = 8 # 16 in tacotron
    hidden_units = 64 # 128 in tacotron
    norm_type = 'ins'
    n_right = 10
    n_left = 20
    
    bottleneck_size = 28
    
    with tf.variable_scope('CBHBH', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        x = tf.transpose(x, [0,2,1])
        
        with tf.variable_scope('cortical_approx', reuse=reuse):
            
            wrx = tf.layers.dense(x, units=hidden_units*2, activation=None, name='wr')
            wix = tf.layers.dense(x, units=hidden_units*2, activation=None, name='wi')
            
            ry = wrx - wix
            iy = wrx + wix
        
            cortical = tf.pow(tf.pow(ry, 2) + tf.pow(iy, 2), 0.5)
            
        res = tf.layers.dense(x, units=hidden_units*2, activation=None)
        out = cortical + res
        out = tf.nn.relu(out)
        
        out = tf.layers.dense(out, units=hidden_units, activation=tf.nn.relu)
        
        # highway
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))

        # bottleneck
        out = tf.layers.dense(out, units=bottleneck_size, activation=None, name="bottleneck")

        # context window
        # input is [-1, spec_w, bottleneck_size]
        # output is [-1, spec_w-30, bottleneck_size*31]
        contexts = []
        for i in range(1,n_left+1):
            contexts.append( out[:,n_left-i:-(n_right+i),:] )
        contexts.append( out[:,n_left:-n_right,:] )
        for i in range(1,n_right):
            contexts.append( out[:,n_left+i:-n_right+i,:] )
        contexts.append( out[:,n_left+n_right:,:] )
        
        out = tf.concat(contexts, axis=2)
        new_num_units = (n_right+n_left+1) * bottleneck_size

        # classifier highway
        for i in range(6):
            out = highway_block(out, num_units=new_num_units, scope='highwaynet_classifier_{}'.format(i))
                   
        # get classification
        out = tf.layers.dense(out, n_classes)
        if n_classes == 1:
            out = tf.squeeze(out, axis=1) # [-1]

    return out
    
def cortical_net_v0res2(x_dict, dropout, reuse, is_training, n_classes, spec_h, spec_w):
    """
    1-D convolution bank + highway network, bottleneck, highway network
    
    Args:
        int spec_h: ~freq
        int spec_h: time
    """
    rv = [4, 8, 16, 32]
    sv = [.25, .5, 1, 2, 4, 8]
    ncort = len(rv) + 2*len(sv)
    filter_size = 32

    num_banks = 8 # 16 in tacotron
    hidden_units = 64 # 128 in tacotron
    norm_type = 'ins'
    n_right = 10
    n_left = 20
    
    bottleneck_size = 28
    
    with tf.variable_scope('CBHBH', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        
        cortical = concat_rategram_scalegram(x, rv, sv, filter_size, is_training, dropout) # batch, freq, time, chan
        
        x = tf.expand_dims(x, -1) # batch, freq, time, 1
        res = tf.layers.conv2d(cortical, 1, 1, 1, activation=None, name="conv1")
        res = x + res
        
        out = tf.transpose(res, [0,2,1,3])
        out = tf.reshape(out, [-1, spec_w, spec_h * 1])

        # batch, time, depth
         
        with tf.variable_scope('prenet', reuse=reuse):
            out = simple_prenet(out, hidden_units, is_training, dropout)
        
        
        # highway
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))

        # bottleneck
        out = tf.layers.dense(out, units=bottleneck_size, activation=None, name="bottleneck")

        # context window
        # input is [-1, spec_w, bottleneck_size]
        # output is [-1, spec_w-30, bottleneck_size*31]
        contexts = []
        for i in range(1,n_left+1):
            contexts.append( out[:,n_left-i:-(n_right+i),:] )
        contexts.append( out[:,n_left:-n_right,:] )
        for i in range(1,n_right):
            contexts.append( out[:,n_left+i:-n_right+i,:] )
        contexts.append( out[:,n_left+n_right:,:] )
        
        out = tf.concat(contexts, axis=2)
        new_num_units = (n_right+n_left+1) * bottleneck_size

        # classifier highway
        for i in range(6):
            out = highway_block(out, num_units=new_num_units, scope='highwaynet_classifier_{}'.format(i))
                   
        # get classification
        out = tf.layers.dense(out, n_classes)
        if n_classes == 1:
            out = tf.squeeze(out, axis=1) # [-1]

    return out
    
def cortical_net_v0res3(x_dict, dropout, reuse, is_training, n_classes, spec_h, spec_w):
    """
    1-D convolution bank + highway network, bottleneck, highway network
    
    Args:
        int spec_h: ~freq
        int spec_h: time
    """
    rv = [4, 8, 16, 32]
    sv = [.25, .5, 1, 2, 4, 8]
    ncort = len(rv) + 2*len(sv)
    filter_size = 32

    num_banks = 8 # 16 in tacotron
    hidden_units = 64 # 128 in tacotron
    norm_type = 'ins'
    n_right = 10
    n_left = 20
    
    bottleneck_size = 28
    
    with tf.variable_scope('CBHBH', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        
        cortical = concat_rategram_scalegram(x, rv, sv, filter_size, is_training, dropout) # batch, freq, time, chan
        
        x = tf.expand_dims(x, -1) # batch, freq, time, 1
        res1 = tf.layers.conv2d(x, 4, 1, 1, activation=None, name="conv1a")
        res2 = tf.layers.conv2d(cortical, 4, 1, 1, activation=None, name="conv1b")
        res1 = res1 + res2
        
        out = tf.transpose(res1, [0,2,1,3])
        out = tf.reshape(out, [-1, spec_w, spec_h * 4])

        # batch, time, depth
         
        with tf.variable_scope('prenet', reuse=reuse):
            out = simple_prenet(out, hidden_units, is_training, dropout)
        
        
        # highway
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))

        # bottleneck
        out = tf.layers.dense(out, units=bottleneck_size, activation=None, name="bottleneck")

        # context window
        # input is [-1, spec_w, bottleneck_size]
        # output is [-1, spec_w-30, bottleneck_size*31]
        contexts = []
        for i in range(1,n_left+1):
            contexts.append( out[:,n_left-i:-(n_right+i),:] )
        contexts.append( out[:,n_left:-n_right,:] )
        for i in range(1,n_right):
            contexts.append( out[:,n_left+i:-n_right+i,:] )
        contexts.append( out[:,n_left+n_right:,:] )
        
        out = tf.concat(contexts, axis=2)
        new_num_units = (n_right+n_left+1) * bottleneck_size

        # classifier highway
        for i in range(6):
            out = highway_block(out, num_units=new_num_units, scope='highwaynet_classifier_{}'.format(i))
                   
        # get classification
        out = tf.layers.dense(out, n_classes)
        if n_classes == 1:
            out = tf.squeeze(out, axis=1) # [-1]

    return out
    
def cortical_net_v0res4(x_dict, dropout, reuse, is_training, n_classes, spec_h, spec_w):
    """
    1-D convolution bank + highway network, bottleneck, highway network
    
    Args:
        int spec_h: ~freq
        int spec_h: time
    """
    rv = [4, 8, 16, 32]
    sv = [.25, .5, 1, 2, 4, 8]
    ncort = len(rv) + 2*len(sv)
    filter_size = 32

    num_banks = 8 # 16 in tacotron
    hidden_units = 64 # 128 in tacotron
    norm_type = 'ins'
    n_right = 10
    n_left = 20
    
    bottleneck_size = 28
    
    with tf.variable_scope('CBHBH', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        
        cortical = concat_rategram_scalegram(x, rv, sv, filter_size, is_training, dropout) # batch, freq, time, chan
        out = tf.transpose(cortical, [0,2,1,3])
        out = tf.reshape(out, [-1, spec_w, spec_h * ncort])
        out = tf.layers.dense(out, units=hidden_units, activation=None)
        
        res = tf.transpose(x, [0,2,1])
        res = tf.layers.dense(res, units=hidden_units, activation=None)
        
        out = out + res
        out = tf.nn.relu(out)
        
        
        # highway
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))

        # bottleneck
        out = tf.layers.dense(out, units=bottleneck_size, activation=None, name="bottleneck")

        # context window
        # input is [-1, spec_w, bottleneck_size]
        # output is [-1, spec_w-30, bottleneck_size*31]
        contexts = []
        for i in range(1,n_left+1):
            contexts.append( out[:,n_left-i:-(n_right+i),:] )
        contexts.append( out[:,n_left:-n_right,:] )
        for i in range(1,n_right):
            contexts.append( out[:,n_left+i:-n_right+i,:] )
        contexts.append( out[:,n_left+n_right:,:] )
        
        out = tf.concat(contexts, axis=2)
        new_num_units = (n_right+n_left+1) * bottleneck_size

        # classifier highway
        for i in range(6):
            out = highway_block(out, num_units=new_num_units, scope='highwaynet_classifier_{}'.format(i))
                   
        # get classification
        out = tf.layers.dense(out, n_classes)
        if n_classes == 1:
            out = tf.squeeze(out, axis=1) # [-1]

    return out
    
def cortical_net_v0res5(x_dict, dropout, reuse, is_training, n_classes, spec_h, spec_w):
    """
    1-D convolution bank + highway network, bottleneck, highway network
    
    Args:
        int spec_h: ~freq
        int spec_h: time
    """
    rv = [4, 8, 16, 32]
    sv = [.25, .5, 1, 2, 4, 8]
    ncort = len(rv) + 2*len(sv)
    filter_size = 32

    num_banks = 8 # 16 in tacotron
    hidden_units = 64 # 128 in tacotron
    norm_type = 'ins'
    n_right = 10
    n_left = 20
    
    bottleneck_size = 28
    
    with tf.variable_scope('CBHBH', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        
        cortical = concat_rategram_scalegram(x, rv, sv, filter_size, is_training, dropout) # batch, freq, time, chan
        out = tf.transpose(cortical, [0,2,1,3])
        out = tf.reshape(out, [-1, spec_w, spec_h * ncort])
        out = tf.layers.dense(out, units=hidden_units*2, activation=tf.nn.relu)
        out = tf.layers.dense(out, units=hidden_units, activation=None)
        
        res = tf.transpose(x, [0,2,1])
        res = tf.layers.dense(res, units=hidden_units, activation=None)
        
        out = out + res
        out = tf.nn.relu(out)
        
        # highway
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))

        # bottleneck
        out = tf.layers.dense(out, units=bottleneck_size, activation=None, name="bottleneck")

        # context window
        # input is [-1, spec_w, bottleneck_size]
        # output is [-1, spec_w-30, bottleneck_size*31]
        contexts = []
        for i in range(1,n_left+1):
            contexts.append( out[:,n_left-i:-(n_right+i),:] )
        contexts.append( out[:,n_left:-n_right,:] )
        for i in range(1,n_right):
            contexts.append( out[:,n_left+i:-n_right+i,:] )
        contexts.append( out[:,n_left+n_right:,:] )
        
        out = tf.concat(contexts, axis=2)
        new_num_units = (n_right+n_left+1) * bottleneck_size

        # classifier highway
        for i in range(6):
            out = highway_block(out, num_units=new_num_units, scope='highwaynet_classifier_{}'.format(i))
                   
        # get classification
        out = tf.layers.dense(out, n_classes)
        if n_classes == 1:
            out = tf.squeeze(out, axis=1) # [-1]

    return out

def cortical_net_v0b(x_dict, dropout, reuse, is_training, n_classes, spec_h, spec_w):
    """
    1-D convolution bank + highway network, bottleneck, highway network
    
    Args:
        int spec_h: ~freq
        int spec_h: time
    """
    rv = [4, 8, 16, 32]
    sv = [.25, .5, 1, 2, 4, 8]
    ncort = len(rv) + 2*len(sv)
    filter_size = 16

    num_banks = 8 # 16 in tacotron
    hidden_units = 64 # 128 in tacotron
    norm_type = 'ins'
    n_right = 10
    n_left = 20
    
    bottleneck_size = 28
    
    with tf.variable_scope('CBHBH', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        
        cortical = concat_rategram_scalegram(x, rv, sv, filter_size, is_training, dropout) # batch, freq, time, chan
        
        out = tf.transpose(cortical, [0,2,1,3])
        out = tf.reshape(out, [-1, spec_w, spec_h * ncort])

        # batch, time, depth
         
        with tf.variable_scope('prenet', reuse=reuse):
            out = simple_prenet(out, hidden_units, is_training, dropout)
        
        
        # highway
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))

        # bottleneck
        out = tf.layers.dense(out, units=bottleneck_size, activation=None, name="bottleneck")

        # context window
        # input is [-1, spec_w, bottleneck_size]
        # output is [-1, spec_w-30, bottleneck_size*31]
        contexts = []
        for i in range(1,n_left+1):
            contexts.append( out[:,n_left-i:-(n_right+i),:] )
        contexts.append( out[:,n_left:-n_right,:] )
        for i in range(1,n_right):
            contexts.append( out[:,n_left+i:-n_right+i,:] )
        contexts.append( out[:,n_left+n_right:,:] )
        
        out = tf.concat(contexts, axis=2)
        new_num_units = (n_right+n_left+1) * bottleneck_size

        # classifier highway
        for i in range(6):
            out = highway_block(out, num_units=new_num_units, scope='highwaynet_classifier_{}'.format(i))
                   
        # get classification
        out = tf.layers.dense(out, n_classes)
        if n_classes == 1:
            out = tf.squeeze(out, axis=1) # [-1]

    return out
    
def cortical_net_v0c(x_dict, dropout, reuse, is_training, n_classes, spec_h, spec_w):
    """
    1-D convolution bank + highway network, bottleneck, highway network
    
    Args:
        int spec_h: ~freq
        int spec_h: time
    """
    rv = [4, 8, 16, 32]
    sv = [.25, .5, 1, 2, 4, 8]
    ncort = len(rv) + 2*len(sv)
    filter_size = 8

    num_banks = 8 # 16 in tacotron
    hidden_units = 64 # 128 in tacotron
    norm_type = 'ins'
    n_right = 10
    n_left = 20
    
    bottleneck_size = 28
    
    with tf.variable_scope('CBHBH', reuse=reuse):
        x = x_dict['spectrograms']
        x = tf.reshape(x, (-1,spec_h,spec_w))
        
        cortical = concat_rategram_scalegram(x, rv, sv, filter_size, is_training, dropout) # batch, freq, time, chan
        
        out = tf.transpose(cortical, [0,2,1,3])
        out = tf.reshape(out, [-1, spec_w, spec_h * ncort])

        # batch, time, depth
         
        with tf.variable_scope('prenet', reuse=reuse):
            out = simple_prenet(out, hidden_units, is_training, dropout)
        
        
        # highway
        for i in range(4):
            out = highway_block(out, num_units=hidden_units, scope='highwaynet_featextractor_{}'.format(i))

        # bottleneck
        out = tf.layers.dense(out, units=bottleneck_size, activation=None, name="bottleneck")

        # context window
        # input is [-1, spec_w, bottleneck_size]
        # output is [-1, spec_w-30, bottleneck_size*31]
        contexts = []
        for i in range(1,n_left+1):
            contexts.append( out[:,n_left-i:-(n_right+i),:] )
        contexts.append( out[:,n_left:-n_right,:] )
        for i in range(1,n_right):
            contexts.append( out[:,n_left+i:-n_right+i,:] )
        contexts.append( out[:,n_left+n_right:,:] )
        
        out = tf.concat(contexts, axis=2)
        new_num_units = (n_right+n_left+1) * bottleneck_size

        # classifier highway
        for i in range(6):
            out = highway_block(out, num_units=new_num_units, scope='highwaynet_classifier_{}'.format(i))
                   
        # get classification
        out = tf.layers.dense(out, n_classes)
        if n_classes == 1:
            out = tf.squeeze(out, axis=1) # [-1]

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
