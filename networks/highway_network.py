from tf.keras.layers import Conv1D, GRU, Dropout, Activation, Dropout, BatchNormalization, Dense
import tensorflow as tf

def weight_bias(W_shape, b_shape, bias_init=0.1):
    W = tf.Variable(tf.truncated_normal(shape=W_shape, stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(bias_init, shape=b_shape), name='bias')
    return W, b

def weight(W_shape):
    W = tf.Variable(tf.truncated_normal(shape=W_shape, stddev=0.1), name='weight')
    return W

def dense_layer(x, W_shape, b_shape):
    W, b = weight_bias([W_shape, b_shape], [b_shape])
    return tf.nn.relu(tf.matmul(x, W) + b)

def highway_block(x, size, carry_bias=-1.0):
    W = weight([size, size])

    with tf.name_scope('transform_gate'):
        W_T = weight([size, size])

    H = tf.nn.relu(tf.linalg.matmul(X, W), name='activation')
    T = tf.sigmoid(tf.linalg.matmul(x, W_T), name='transform_gate')
    C = tf.sub(1.0, T, name="carry_gate")

    y = tf.add(tf.linalg.matmul(H, T), tf.linalg.matmul(x, C), name='y') # y = (H * T) + (x * C)
    return y


def five_layer_highway(x, size,dropout_rate):
    y = highway_block(x, size)
    y = tf.nn.dropout(y, dropout_rate)
    y = highway_block(y, size)
    y = tf.nn.dropout(y, dropout_rate)
    y = highway_block(y, size)
    y = tf.nn.dropout(y, dropout_rate)
    y = highway_block(y, size)
    y = tf.nn.dropout(y, dropout_rate)
    y = highway_block(y, size)
    return y

def bottle_net_layer(x, size, bottleneck_size=28):
    W, b = weight_bias([size, bottleneck_size],[bottleneck_size])
    y = tf.layers.BatchNormalization(tf.matmul(x, W) + b)
    y = tf.nn.relu(y, name='activation')
    return y

def output_networks(x, bottle_neck_size, size, dropout_rate):
    W, b = weight_bias([bottleneck_size, size],[size])
    y = tf.nn.relu(tf.matmul(x, W) + b, name='activation')
    y = tf.nn.dropout(y, dropout_rate)
    y = highway_block(x, size)
    y = tf.nn.dropout(y, dropout_rate)
    y = highway_block(y, size)
    y = tf.nn.dropout(y, dropout_rate)
    y = highway_block(y, size)
    y = tf.nn.dropout(y, dropout_rate)
    y = highway_block(y, size)
    y = tf.nn.dropout(y, dropout_rate)
    y = highway_block(y, size)
    y = tf.nn.dropout(y, dropout_rate)
    y = dense_layer(y, size, 1):
    return y

def gru_model(x):
    X = tf.reshape(x, (batch_size, Tx, n_feq*n_frames))
    X = Conv1D(filters=256,kernel_size=15,strides=3)(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Dropout(0.2)(X)

    X = GRU(32, return_sequences=True)(X)
    X = Dropout(0.2)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.2)(X)

    X = GRU(32, return_sequences=True)(X)
    X = Dropout(0.2)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.2)(X)

    X = GRU(32, return_sequences=True)(X)
    X = Dropout(0.2)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.2)(X)

    X = GRU(64, return_sequences=True)(X)
    X = Dropout(0.2)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.2)(X)

    X = GRU(64, return_sequences=False)(X)
    X = Dropout(0.2)(X)
    X = BatchNormalization()(X)

    X = Dense(1)(X)
    return X