'''
Neural Network Operations
'''

# pylint: disable=invalid-name, no-member
import tensorflow as tf

import numpy as np


def rgb2gray(rgb):
  '''
  RGB to Gray Op
  '''
  return np.dot(rgb[:, :, :3], [0.299, 0.587, 0.114])


def up_sampling2d(inputs, size=None):
  '''
  Up Sampling 2D
  '''
  if not size:
    size = [2, 2]
  shape = inputs.shape.as_list()
  width, height = shape[1:3] * np.array(size, dtype=np.uint32)
  ret = tf.image.resize_nearest_neighbor(inputs, [width, height])
  ret.set_shape([shape[0], width, height, shape[3]])
  return ret
