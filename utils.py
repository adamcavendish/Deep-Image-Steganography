'''
Collection of utility functions
'''

# pylint: disable=C0111
import functools
import math
import multiprocessing as mp
import os
import pathlib
import re
import sys

import PIL.Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sh

eprint = functools.partial(print, file=sys.stderr, flush=True)

_process_pool = {}

_FORMAT_TYPE_F = (
    ((int, np.integer), lambda d: '%05d' % d),
    (
        (float, np.floating),  # pylint: disable=E1101
        lambda d: '%03.4f' % d),
    ((str, ), lambda d: d),
)


def format_one(value):
  for tf_tpl in _FORMAT_TYPE_F:
    types, func = tf_tpl
    for t in types:
      if isinstance(value, t):
        return func(value)
  raise RuntimeError('Unexpected type: {}, value={}'.format(type(value), value))


def format_kv(tpl):
  return ','.join('{}={}'.format(key, format_one(value)) for key, value in tpl)


def path_format_kv(tpl):
  return ','.join('{}={}'.format(key, format_one(value)) for key, value in tpl)


def print_kv(tpl):
  print("\033[K", end='\r')
  print(format_kv(tpl), end='')
  print('\r', end='')


def fpath(f):
  return pathlib.Path(os.path.dirname(f))


def start_process(name, f, args=(), daemon=True):
  p = mp.Process(target=f, args=args, daemon=daemon)
  if not name in _process_pool:
    _process_pool[name] = p
    p.start()
    eprint('Process %s[%d] has started' % (name, p.pid))
  else:
    raise RuntimeError('name: "%s" has been occupied' % name)


def join_all_processes():
  for name, p in _process_pool.items():
    eprint('Process join started: "%s"' % name)
    p.join()
    eprint('Process has finished: "%s"' % name)


def prepare_dirs(script_Path, gmode, info):
  name = path_format_kv(sorted(info.items()))

  if gmode == 'inference':
    ckpt_i_Path = script_Path / 'logging' / 'inference'
    ckpt_i_Path.mkdir(parents=True, exist_ok=True)
    ckpt_i_Path = ckpt_i_Path / 'checkpoint'
    if ckpt_i_Path.exists() and not ckpt_i_Path.is_symlink():
      raise RuntimeError("Inference mode's checkpoint directory should be symlink to training's")
    if ckpt_i_Path.is_symlink():
      ckpt_i_Path.unlink()
    ckpt_i_Path.symlink_to('../train/checkpoint/')

  ckpt_Path = script_Path / 'logging' / gmode / 'checkpoint' / name
  ckpt_Path.mkdir(parents=True, exist_ok=True)

  visual_Path = script_Path / 'logging' / gmode / 'visual' / name
  visual_Path.mkdir(parents=True, exist_ok=True)

  summary_Path = script_Path / 'logging' / gmode / 'summary' / name
  summary_Path.mkdir(parents=True, exist_ok=True)

  logging_Path = script_Path / 'logging' / gmode / 'logging' / name
  logging_Path.mkdir(parents=True, exist_ok=True)

  return ckpt_Path, visual_Path, summary_Path, logging_Path


def msg_st(msg, key, value):
  msg_path = msg
  key_path = key.split('|')
  last_key = key_path.pop()
  for k in key_path:
    if not k in msg_path:
      msg_path[k] = {}
    msg_path = msg_path.get(k)
  msg_path[last_key] = value


def msg_ud(msg, key, value):
  msg_path = msg
  key_path = key.split('|')
  last_key = key_path.pop()
  for k in key_path:
    if not k in msg_path:
      raise RuntimeError('%s(%s) is not in message' % (key, k))
    else:
      msg_path = msg_path.get(k)
  if not last_key in msg_path:
    raise RuntimeError('%s(%s) is not in message' % (key, last_key))
  msg_path[last_key] = value


def msg_gt(msg, key):
  key_path = key.split('|')
  for k in key_path:
    if not k in msg:
      raise RuntimeError('%s(%s) is not in message' % (key, k))
    msg = msg.get(k)
  return msg


class ImageSlicer(object):
  '''
  Image Slicer
  '''

  def __init__(self, inrows, incols, fnrows, fncols):
    '''
    inrows: Image rows (in pixel)
    incols: Image columns (in pixel)
    fnrows: Sliced fragment rows (in pixel)
    fncols: Sliced fragment columns (in pixel)
    '''
    self.inrows, self.incols = inrows, incols
    self.fnrows, self.fncols = fnrows, fncols
    self.rows = inrows // fnrows
    self.cols = incols // fncols
    self._slice_f = [None, None, self._slice_2, self._slice_3, self._slice_4]
    self._slice_f_assign = [
        None,
        None,  # Placeholder for dimension 0, 1
        self._slice_2_assign,
        self._slice_3_assign,
        self._slice_4_assign
    ]

  def slice(self, image, row_index, col_index):
    '''
    Get a slice of fragment at (row_index, col_index) from image

    image:
        numpy compatible data
    image shape:
        2: (row, column)                 -- grayscale image
        3: (row, column, channel)        -- one image
        4: (batch, row, column, channel) -- a batch of images
    '''
    shape = len(np.shape(image))
    if not shape in [2, 3, 4]:
      raise RuntimeError('Invalid image shape for slicing')
    if row_index >= self.rows:
      raise RuntimeError('Invalid row index ({}), max rows ')
    return self._slice_f[shape](image, row_index, col_index)

  def slice_assign(self, image, row_index, col_index, fragment):
    '''
    Assign a fragment at (row_index, col_index) to image
    '''
    shape = len(np.shape(image))
    if not shape in [2, 3, 4]:
      raise RuntimeError('Invalid image shape for slicing')
    if row_index >= self.rows:
      raise RuntimeError('Invalid row index ({}), max rows ')
    self._slice_f_assign[shape](image, row_index, col_index, fragment)

  def _slice_2(self, image, row_index, col_index):
    return image[row_index * self.fnrows:(row_index + 1) * self.fnrows, col_index * self.fncols:(
        col_index + 1) * self.fncols, ]

  def _slice_3(self, image, row_index, col_index):
    return image[row_index * self.fnrows:(row_index + 1) * self.fnrows, col_index * self.fncols:(
        col_index + 1) * self.fncols, :]

  def _slice_4(self, image, row_index, col_index):
    return image[:, row_index * self.fnrows:(row_index + 1) * self.fnrows, col_index * self.fncols:(
        col_index + 1) * self.fncols, :]

  def _slice_2_assign(self, image, row_index, col_index, fragment):
    image[row_index * self.fnrows:(row_index + 1) * self.fnrows, col_index * self.fncols:(
        col_index + 1) * self.fncols, ] = fragment

  def _slice_3_assign(self, image, row_index, col_index, fragment):
    image[row_index * self.fnrows:(row_index + 1) * self.fnrows, col_index * self.fncols:(
        col_index + 1) * self.fncols, :] = fragment

  def _slice_4_assign(self, image, row_index, col_index, fragment):
    image[:, row_index * self.fnrows:(row_index + 1) * self.fnrows, col_index * self.fncols:(
        col_index + 1) * self.fncols, :] = fragment


def image_comp(images_lst, padding=3, pad_value=0):
  '''
  Images:
      List[numpy.ndarray], NHWC type, normalized to [-1, 1]

      The list of images are placed side by side from left to right
  '''
  improw, nrows, ncols, ncnls = images_lst[0].shape
  impcol = len(images_lst)

  # Check shapes
  for images in images_lst:
    if (improw, nrows, ncols, ncnls) != images.shape:
      raise AssertionError(
          'Image shapes are not the same across the images list. Expect(%s) != Actual(%s)' %
          ((improw, nrows, ncols, ncnls), images.shape))

  ret_rows = improw * nrows + padding * (improw + 1)
  ret_cols = impcol * ncols + padding * (impcol + 1)

  ret = np.ones((ret_rows, ret_cols, ncnls)) * pad_value

  for ridx in range(improw):
    for cidx in range(impcol):
      img = images_lst[cidx][ridx]
      rlb = (padding + nrows) * ridx + padding
      rub = (padding + nrows) * (ridx + 1)
      clb = (padding + ncols) * cidx + padding
      cub = (padding + ncols) * (cidx + 1)
      ret[rlb:rub, clb:cub, :] = img

  return ret


def normalize(mat):
  '''
  Normalize a matrix into range [-1, 1]
  '''
  return 2 * (mat - np.min(mat)) / np.ptp(mat) - 1


def nchw2nhwc(image):
  '''
  Convert from NCHW to NHWC
  '''
  return np.transpose(image, [0, 2, 3, 1])


def nhwc2nchw(image):
  '''
  Convert from NHWC to NCHW
  '''
  return np.transpose(image, [0, 3, 1, 2])


def norm2mpl(image):
  '''
  Norm format: NHWC, normalized to [-1, 1]
  PIL  format: PIL.Image, [0, 255]
          rgb: HWC
         gray: HW
  matplotlib : numpy, normalized to [0, 1]
          rgb: HWC
         gray: HW
  '''
  image = image.squeeze()
  return (image + 1) / 2


def norm2pil(image):
  '''
  Norm format: NHWC, normalized to [-1, 1]
  PIL  format: PIL.Image, [0, 255]
          rgb: HWC
         gray: HW
  matplotlib : numpy, normalized to [0, 1]
          rgb: HWC
         gray: HW
  '''
  image = image.squeeze()
  image = (image + 1) * 255 / 2
  return PIL.Image.fromarray(image.astype('uint8'))


def mpl2norm(image):
  '''
  Norm format: NHWC, normalized to [-1, 1]
  PIL  format: PIL.Image, [0, 255]
          rgb: HWC
         gray: HW
  matplotlib : numpy, normalized to [0, 1]
          rgb: HWC
         gray: HW
  '''
  image = image * 2 - 1
  if len(image.shape) == 2:
    image.reshape((1, *image.shape, 1))
  if len(image.shape) == 3:
    image.reshape((1, *image.shape))
  return image


def pil2norm(image):
  '''
  Norm format: NHWC, normalized to [-1, 1]
  PIL  format: PIL.Image, [0, 255]
          rgb: HWC
         gray: HW
  matplotlib : numpy, normalized to [0, 1]
          rgb: HWC
         gray: HW
  '''
  image = np.array(image) / 255. * 2 - 1
  if len(image.shape) == 2:
    image.reshape((1, *image.shape, 1))
  if len(image.shape) == 3:
    image.reshape((1, *image.shape))
  return image


def mpl2pil(image):
  '''
  Norm format: NHWC, normalized to [-1, 1]
  PIL  format: PIL.Image, [0, 255]
          rgb: HWC
         gray: HW
  matplotlib : numpy, normalized to [0, 1]
          rgb: HWC
         gray: HW
  '''
  return PIL.Image.fromarray((image * 255).astype('uint8'))


def pil2mpl(image):
  '''
  Norm format: NHWC, normalized to [-1, 1]
  PIL  format: PIL.Image, [0, 255]
          rgb: HWC
         gray: HW
  matplotlib : numpy, normalized to [0, 1]
          rgb: HWC
         gray: HW
  '''
  return np.array(image) / 255.
