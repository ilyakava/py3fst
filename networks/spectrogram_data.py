import os
import random

import tensorflow as tf

def parser(serialized_example, h, w):
    """Parses a single tf.Example into x,y tensors.
    """
    features = tf.parse_single_example(
      serialized_example,
      features={
          'spectrogram': tf.FixedLenFeature([h * w], tf.float32),
          'spectrogram_label': tf.FixedLenFeature([w], tf.int64),
      })
    spec = features['spectrogram']
    label = tf.cast(features['spectrogram_label'], tf.int32)
    spec = tf.reshape(spec, (h, w))
    
    return spec, label

def time_cut_parser(serialized_example, h, in_w, out_w):
    """Parses a single tf.Example into x,y tensors.
    
    Does data augmentation by selecting a random
    sub-sequence at parse time.
    """
    features = tf.parse_single_example(
      serialized_example,
      features={
          'spectrogram': tf.FixedLenFeature([h * in_w], tf.float32),
          'spectrogram_label': tf.FixedLenFeature([in_w], tf.int64),
      })
    spec = features['spectrogram']
    label = tf.cast(features['spectrogram_label'], tf.int32)
    spec = tf.reshape(spec, (h, in_w))
    
    si = random.randint(0, in_w - out_w - 1)
    ei = si + out_w
    
    return spec[:,si:ei], label[si:ei]
    
def input_fn(tfrecord_dir, bs=32, parser=parser, infinite=True):
    """
    This function is called once for every example for every epoch, so
    data augmentation that happens randomly will be different every
    time.

    More info:
        https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset
    """
    tfrecord_files = tf.io.gfile.glob(os.path.join(tfrecord_dir, '*.tfrecord'))
    random.shuffle(tfrecord_files)
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    
    
    # Map the parser over dataset, and batch results by up to batch_size
    dataset = dataset.map(parser)
    dataset = dataset.batch(bs)
    dataset = dataset.prefetch(buffer_size=8*bs)
    dataset = dataset.shuffle(buffer_size=4*bs)
    if infinite:
        dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    
    features, labels = iterator.get_next()
    
    return { 'spectrograms': features }, labels
    
def identity_serving_input_receiver_fn(h, w):
    """
    This function is supposed to translate what the user gives the model
    to what should actually be given to the model.
    In our case this is the identity function.
    
    A useful way to use a 'serving_input_receiver_fn' would be to provide a
    string of image bytes and read/convert it into a tensor of numbers for
    the model.
    """
    serialized_tf_example = tf.placeholder(dtype=tf.float32, shape=[None, h, w])
    user_input = {'spectrograms': serialized_tf_example }
    model_input = user_input
    return tf.estimator.export.ServingInputReceiver(model_input, user_input)