"""DFFN networks for HSI classification.
    
Based on:

Hyperspectral Image Classification With Deep Feature Fusion Network
Song, Li, Fang, Lu

https://github.com/weiweisong415/Demo_DFFN
    
Spectral size:
IP=3, PaviaU=5, Salinas=10

Spatial size:
IP=25, PaviaU=23, Salinas=27

Batch size: 100
"""

import tensorflow as tf

import pdb

        
def block1(inp, nchannels=16):
    """
    out = in->Conv->BN->Relu->Conv->BN
    out += in
    """
    conv1 = tf.layers.conv2d(inp, nchannels, 3, activation=None, padding='same')
    conv1 = tf.layers.batch_normalization(conv1)
    conv1 = tf.nn.relu(conv1)
    
    conv2 = tf.layers.conv2d(conv1, nchannels, 3, activation=None, padding='same')
    conv2 = tf.layers.batch_normalization(conv2)
    return tf.math.add(inp, conv2)

def block2(inp, nchannels=32):
    """Shrinks the input by 2
    
    path1 = in -> strided conv1 -> bn
    path2 = in -> strided conv3 -> bn->relu->conv3->bn
    
    out = path1 + path2
    """
    
    path1 = tf.layers.conv2d(inp, nchannels, 1, 2, activation=None, padding='same')
    path1 = tf.layers.batch_normalization(path1)
    
    path2 = tf.layers.conv2d(inp, nchannels, 3, 2, activation=None, padding='same')
    path2 = tf.layers.batch_normalization(path2)
    path2 = tf.nn.relu(path2)
    path2 = tf.layers.conv2d(path2, nchannels, 3, activation=None, padding='same')
    path2 = tf.layers.batch_normalization(path2)
    
    return tf.math.add(path1, path2)

def DFFN_3tower_5depth(x_dict, dropout, reuse, is_training, n_classes):
    """Three towers. Each depth 5.
    
    This is the train_paviaU network when input is 23.
    
    20 steps/second.
    
    94.5% on PaviaU 2%, lr 5e-5, within 10k
    """
    with tf.variable_scope('DFFN', reuse=reuse):
        x = x_dict['subimages']
        
        conv1 = tf.layers.conv2d(x, 16, 3, activation=None, padding='same')
        conv1 = tf.layers.batch_normalization(conv1)
        conv1 = tf.nn.relu(conv1)
        # beginning
        
        b1 = tf.nn.relu(block1(conv1))
        b1 = tf.nn.relu(block1(b1))
        b1 = tf.nn.relu(block1(b1))
        b1 = tf.nn.relu(block1(b1))
        b1 = tf.nn.relu(block1(b1))
        
        t2 = tf.nn.relu(block2(b1,32))
        t2 = tf.nn.relu(block1(t2,32))
        t2 = tf.nn.relu(block1(t2,32))
        t2 = tf.nn.relu(block1(t2,32))
        t2 = tf.nn.relu(block1(t2,32))
        
        t3 = block2(t2,64)
        t3 = block1(tf.nn.relu(t3),64)
        t3 = block1(tf.nn.relu(t3),64)
        t3 = block1(tf.nn.relu(t3),64)
        t3 = block1(tf.nn.relu(t3),64)
        
        # combine
        t1out = tf.layers.conv2d(b1, 64, 3, 4, activation=None, padding='same')
        t1out = tf.layers.batch_normalization(t1out)
        
        t2out = tf.layers.conv2d(t2, 64, 3, 2, activation=None, padding='same')
        t2out = tf.layers.batch_normalization(t2out)
        
        fuse = tf.math.add(t1out, t2out)
        fuse = tf.math.add(fuse, t3)
        fuse = tf.nn.relu(fuse)
        
        # end
        fuse = tf.reduce_mean(fuse, axis=(1,2), keepdims=True) # avg pooling
        fuse = tf.layers.dense(fuse, n_classes)
    return tf.squeeze(fuse)


def DFFN_3tower_4depth(x_dict, dropout, reuse, is_training, n_classes):
    """Three towers. Each depth 4.
    
    This is the train_indian_pines network when input is 25.
    This is the train_salinas network when input is 27.
    
    23 steps/second.
    
    97.0% IP p10, lr 5e-5
    """
    with tf.variable_scope('DFFN', reuse=reuse):
        x = x_dict['subimages']
        
        conv1 = tf.layers.conv2d(x, 16, 3, activation=None, padding='same')
        conv1 = tf.layers.batch_normalization(conv1)
        conv1 = tf.nn.relu(conv1)
        # beginning
        
        b1 = tf.nn.relu(block1(conv1))
        b1 = tf.nn.relu(block1(b1))
        b1 = tf.nn.relu(block1(b1))
        b1 = tf.nn.relu(block1(b1))
        
        t2 = tf.nn.relu(block2(b1,32))
        t2 = tf.nn.relu(block1(t2,32))
        t2 = tf.nn.relu(block1(t2,32))
        t2 = tf.nn.relu(block1(t2,32))
        
        t3 = block2(t2,64)
        t3 = block1(tf.nn.relu(t3),64)
        t3 = block1(tf.nn.relu(t3),64)
        t3 = block1(tf.nn.relu(t3),64)
        
        # combine
        t1out = tf.layers.conv2d(b1, 64, 3, 4, activation=None, padding='same')
        t1out = tf.layers.batch_normalization(t1out)
        
        t2out = tf.layers.conv2d(t2, 64, 3, 2, activation=None, padding='same')
        t2out = tf.layers.batch_normalization(t2out)
        
        fuse = tf.math.add(t1out, t2out)
        fuse = tf.math.add(fuse, t3)
        fuse = tf.nn.relu(fuse)
        
        # end
        fuse = tf.reduce_mean(fuse, axis=(1,2), keepdims=True) # avg pooling
        fuse = tf.layers.dense(fuse, n_classes)
    return tf.squeeze(fuse)

def DFFN_3tower_3depth(x_dict, dropout, reuse, is_training, n_classes):
    """Three towers. Each depth 3.
    
    30 steps/second.
    
    Input as 25.
    97.0% IP p10, within 10k itr, lr 5e-5
    """
    with tf.variable_scope('DFFN', reuse=reuse):
        x = x_dict['subimages']
        
        conv1 = tf.layers.conv2d(x, 16, 3, activation=None, padding='same')
        conv1 = tf.layers.batch_normalization(conv1)
        conv1 = tf.nn.relu(conv1)
        # beginning
        
        b1 = tf.nn.relu(block1(conv1))
        b1 = tf.nn.relu(block1(b1))
        b1 = tf.nn.relu(block1(b1))
        
        t2 = tf.nn.relu(block2(b1,32))
        t2 = tf.nn.relu(block1(t2,32))
        t2 = tf.nn.relu(block1(t2,32))
        
        t3 = block2(t2,64)
        t3 = block1(tf.nn.relu(t3),64)
        t3 = block1(tf.nn.relu(t3),64)
        
        # combine
        t1out = tf.layers.conv2d(b1, 64, 3, 4, activation=None, padding='same')
        t1out = tf.layers.batch_normalization(t1out)
        
        t2out = tf.layers.conv2d(t2, 64, 3, 2, activation=None, padding='same')
        t2out = tf.layers.batch_normalization(t2out)
        
        fuse = tf.math.add(t1out, t2out)
        fuse = tf.math.add(fuse, t3)
        fuse = tf.nn.relu(fuse)
        
        # end
        fuse = tf.reduce_mean(fuse, axis=(1,2), keepdims=True) # avg pooling
        fuse = tf.layers.dense(fuse, n_classes)
    return tf.squeeze(fuse)

def DFFN_3tower_2depth(x_dict, dropout, reuse, is_training, n_classes):
    """Three towers. Each depth 2.
    
    40 steps/second.
    
    Input as 25.
    97.7% IP p10, within 10k itr, lr 1e-4 (should try lower lr)
    """
    with tf.variable_scope('DFFN', reuse=reuse):
        x = x_dict['subimages']
        
        conv1 = tf.layers.conv2d(x, 16, 3, activation=None, padding='same')
        conv1 = tf.layers.batch_normalization(conv1)
        conv1 = tf.nn.relu(conv1)
        # beginning
        
        b1 = tf.nn.relu(block1(conv1))
        b1 = tf.nn.relu(block1(b1))
        
        t2 = tf.nn.relu(block2(b1,32))
        t2 = tf.nn.relu(block1(t2,32))
        
        t3 = block2(t2,64)
        t3 = block1(tf.nn.relu(t3),64)
        
        # combine
        t1out = tf.layers.conv2d(b1, 64, 3, 4, activation=None, padding='same')
        t1out = tf.layers.batch_normalization(t1out)
        
        t2out = tf.layers.conv2d(t2, 64, 3, 2, activation=None, padding='same')
        t2out = tf.layers.batch_normalization(t2out)
        
        fuse = tf.math.add(t1out, t2out)
        fuse = tf.math.add(fuse, t3)
        fuse = tf.nn.relu(fuse)
        
        # end
        fuse = tf.reduce_mean(fuse, axis=(1,2), keepdims=True) # avg pooling
        fuse = tf.layers.dense(fuse, n_classes)
    return tf.squeeze(fuse)

def DFFN_3tower_1depth(x_dict, dropout, reuse, is_training, n_classes):
    """Three towers. Each depth 1.
    
    60 steps/second.
    
    Input as 25.
    ~90% IP p05, within 4k itr, lr 1e-4
    96.8% IP p10, within 10k itr, lr 1e-4
    """
    with tf.variable_scope('DFFN', reuse=reuse):
        x = x_dict['subimages']
        
        conv1 = tf.layers.conv2d(x, 16, 3, activation=None, padding='same')
        conv1 = tf.layers.batch_normalization(conv1)
        conv1 = tf.nn.relu(conv1)
        # beginning
        
        b1 = tf.nn.relu(block1(conv1))
        
        t2 = tf.nn.relu(block2(b1,32))
        
        t3 = block2(t2,64)
        
        # combine
        t1out = tf.layers.conv2d(b1, 64, 3, 4, activation=None, padding='same')
        t1out = tf.layers.batch_normalization(t1out)
        
        t2out = tf.layers.conv2d(t2, 64, 3, 2, activation=None, padding='same')
        t2out = tf.layers.batch_normalization(t2out)
        
        fuse = tf.math.add(t1out, t2out)
        fuse = tf.math.add(fuse, t3)
        fuse = tf.nn.relu(fuse)
        
        # end
        fuse = tf.reduce_mean(fuse, axis=(1,2), keepdims=True) # avg pooling
        fuse = tf.layers.dense(fuse, n_classes)
    return tf.squeeze(fuse)
    
def DFFN_2tower_1depth(x_dict, dropout, reuse, is_training, n_classes):
    """Two towers. Each depth 1.
    
    Input as 13.
    84% IP p05, within 4k itr, lr 1e-2
    """
    with tf.variable_scope('DFFN', reuse=reuse):
        x = x_dict['subimages']
        
        conv1 = tf.layers.conv2d(x, 16, 3, activation=None, padding='same')
        conv1 = tf.layers.batch_normalization(conv1)
        conv1 = tf.nn.relu(conv1)
        # beginning
        
        b1 = tf.nn.relu(block1(conv1))
        
        t2 = block2(b1,32)
        
        # combine
        t1out = tf.layers.conv2d(b1, 32, 3, 2, activation=None, padding='same')
        t1out = tf.layers.batch_normalization(t1out)
        
        fuse = tf.math.add(t2, t1out)
        fuse = tf.nn.relu(fuse)
        
        # end
        
        fuse = tf.reduce_mean(fuse, axis=(1,2), keepdims=True) # avg pooling
        fuse = tf.layers.dense(fuse, n_classes)
    return tf.squeeze(fuse)

