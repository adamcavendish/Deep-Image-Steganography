'''
Steganography Model
'''

# pylint: disable=C0326, E1129, E0611

import tensorflow as tf
import tensorflow.contrib.slim as ts

import params
import ops
import tf_utils

# yapf: disable
fc            = ts.add_arg_scope(tf.layers.dense)
conv1d        = ts.add_arg_scope(tf.layers.conv1d)
conv2d        = ts.add_arg_scope(tf.layers.conv2d)
sep_conv2d    = ts.add_arg_scope(tf.layers.separable_conv2d)
max_pooling2d = ts.add_arg_scope(tf.layers.max_pooling2d)
batch_norm    = ts.add_arg_scope(tf.layers.batch_normalization)
# yapf: enable

conv2d_activation = tf.nn.elu

conv2d_params = {
    'kernel_size': 3,
    'strides': (1, 1),
    'padding': 'SAME',
    'kernel_initializer': ts.xavier_initializer(),
    'use_bias': True,
    'bias_initializer': tf.zeros_initializer(),
}

sep_conv2d_params = {
    'kernel_size': 3,
    'strides': (1, 1),
    'dilation_rate': (1, 1),
    'depth_multiplier': 1,
    'padding': 'SAME',
    'depthwise_initializer': ts.xavier_initializer(),
    'pointwise_initializer': ts.xavier_initializer(),
    'use_bias': True,
    'bias_initializer': tf.zeros_initializer(),
}

batch_norm_params = {
    'momentum': 0.9,
    'epsilon': 1e-5,
    'center': True,
    'scale': True,
    'fused': False,
}


def skip_align(inputs, filters, strides, data_format):
  '''
  Pads the input along the channel dimension
  Args:
    inputs: A tensor NHWC or NCHW
    filters: num of filters for padded output
    strides: the stride of the convolution operation
    data_format: 'channels_last' or 'channels_first'
  '''
  cnl_idx = 1 if data_format == 'channels_first' else 3
  paddings = [[0, 0], [0, 0], [0, 0], [0, 0]]

  inputs = tf.layers.average_pooling2d(inputs, 1, strides, padding='SAME', data_format=data_format)

  pad_total = filters - inputs.shape.as_list()[cnl_idx]
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  paddings[cnl_idx] = [pad_beg, pad_end]
  inputs = tf.pad(inputs, paddings)
  return inputs


def standard_block_s2c(inputs, filters, is_training, strides, data_format):
  '''
  Standard Network Building Block, from spatial info to channel info
  '''
  with tf.variable_scope(None, default_name='standard_block_s2c'):
    skip_connection = skip_align(inputs, filters, strides, data_format)
    inputs = batch_norm(inputs, training=is_training)
    inputs = conv2d_activation(inputs)
    inputs = sep_conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=strides,
        padding='SAME',
        data_format=data_format)
    inputs = inputs + skip_connection
    return tf.identity(inputs, name='value')


def standard_block_c2s(inputs, filters, is_training, strides, data_format):
  '''
  Standard Network Building Block, from channel info to spatial info
  '''
  with tf.variable_scope(None, default_name='standard_block_c2s'):
    inputs = batch_norm(inputs, training=is_training)
    inputs = conv2d_activation(inputs)
    inputs = conv2d(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides, data_format=data_format)
    return tf.identity(inputs, name='value')


def encrypter(orig_image, hide_image, is_training):
  '''
  Steganography Encrypter
  '''
  _, _, mncnls = params.MNROWS.value, params.MNCOLS.value, params.MNCNLS.value

  orig_image_shape = tf_utils.shape(orig_image)[1:4]
  hide_image_shape = tf_utils.shape(hide_image)[1:4]
  expc_shape = [params.MNROWS.value, params.MNCOLS.value, params.MNCNLS.value]
  assert orig_image_shape == expc_shape, \
      'Cover Image Dimension Error, Actual({}) != Expected({})'.format(
          orig_image_shape, expc_shape)
  assert hide_image_shape == expc_shape, \
      'Hidden Image Dimension Error, Actual({}) != Expected({})'.format(
          hide_image_shape, expc_shape)
  data_format = 'channels_first'

  with tf.variable_scope('encrypter'):
    with ts.arg_scope([conv2d], **conv2d_params), \
         ts.arg_scope([sep_conv2d], **sep_conv2d_params), \
         ts.arg_scope([batch_norm], **batch_norm_params):
      orig_image = tf.transpose(orig_image, [0, 3, 1, 2])
      hide_image = tf.transpose(hide_image, [0, 3, 1, 2])

      m = tf.concat([orig_image, hide_image], axis=1)

      m = standard_block_s2c(m, 32, is_training, 1, data_format)
      m = standard_block_s2c(m, 32, is_training, 1, data_format)
      m = standard_block_s2c(m, 64, is_training, 1, data_format)
      m = standard_block_s2c(m, 64, is_training, 1, data_format)
      m = standard_block_s2c(m, 128, is_training, 1, data_format)
      m = standard_block_s2c(m, 128, is_training, 1, data_format)

      m = standard_block_c2s(m, 32, is_training, 1, data_format)
      m = standard_block_c2s(m, mncnls, is_training, 1, data_format)

      m = tf.transpose(m, [0, 2, 3, 1], name='steg_image')

      return m


def decrypter(image, is_training):
  '''
  Steganography Decrypter
  '''
  _, _, mncnls = params.MNROWS.value, params.MNCOLS.value, params.MNCNLS.value

  steg_image_shape = tf_utils.shape(image)[1:4]
  expc_shape = [params.MNROWS.value, params.MNCOLS.value, params.MNCNLS.value]
  assert steg_image_shape == expc_shape, \
      'Stegged Image Dimension Error, Actual({}) != Expected({})'.format(
          steg_image_shape, expc_shape)
  data_format = 'channels_first'

  with tf.variable_scope('decrypter'):
    with ts.arg_scope([conv2d], **conv2d_params), \
         ts.arg_scope([sep_conv2d], **sep_conv2d_params), \
         ts.arg_scope([batch_norm], **batch_norm_params):
      m = image
      m = tf.transpose(m, [0, 3, 1, 2])

      m = standard_block_s2c(m, 32, is_training, 1, data_format)
      m = standard_block_s2c(m, 32, is_training, 1, data_format)
      m = standard_block_s2c(m, 64, is_training, 1, data_format)
      m = standard_block_s2c(m, 64, is_training, 1, data_format)
      m = standard_block_s2c(m, 128, is_training, 1, data_format)
      m = standard_block_s2c(m, 128, is_training, 1, data_format)

      m = standard_block_c2s(m, 32, is_training, 1, data_format)
      m = standard_block_c2s(m, mncnls, is_training, 1, data_format)

      m = tf.transpose(m, [0, 2, 3, 1], name='dcpt_image')

    return m