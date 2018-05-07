'''
Tensorflow Utilities
'''
import tensorflow as tf


def shape(tensor):
  '''
  get shape as list
  '''
  return tensor.get_shape().as_list()
