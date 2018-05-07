'''
Logger Module
'''

# pylint: disable=no-self-use
import json
import queue as Q

import numpy as np

import params
import utils

from ..common import Task
from ..common import BaseConsoleLogger, BaseFileLogger


class ConsoleLogger(BaseConsoleLogger):
  '''
  Console Logger
  '''

  def __init__(self):
    super(ConsoleLogger, self).__init__()

    self.line_pattern = [[
        ('gidx', 'message_info|gidx'),
        ('tidx', 'message_info|tidx'),
        ('vidx', 'message_info|vidx'),
    ], [
        ('epoch', 'message_info|epoch'),
        ('batch', 'message_info|batch'),
        ('time', 'running|timing'),
    ], [
        ('rcst_loss', 'post_info|rcst_loss'),
        ('rcst_vars', 'post_info|rcst_vars'),
        ('dcpt_loss', 'post_info|dcpt_loss'),
        ('dcpt_vars', 'post_info|dcpt_vars'),
    ], [
        ('loss', 'post_info|loss'),
    ]]

    self.train_last_msg = None
    self.valid_last_msg = None

  def logging_hv(self, msg):
    self.logging_lt(msg)

  def logging_lt(self, msg):
    mode = utils.msg_gt(msg, 'message_info|mode')
    if mode == 'train':
      self.train_last_msg = msg
    else:
      self.valid_last_msg = msg

    self.stdscr.erase()
    self.stdscr.addstr('Last Validation:\n')
    self.stdscr.addstr('  Validation:\n')
    if self.valid_last_msg:
      self.stdscr.addstr(self.log_one_msg(self.valid_last_msg, indent=2 * 2))
    self.stdscr.addstr('\n')

    self.stdscr.addstr('Last Train:\n')
    self.stdscr.addstr('  Train:\n')
    if self.train_last_msg:
      self.stdscr.addstr(self.log_one_msg(self.train_last_msg, indent=2 * 2))
    self.stdscr.addstr('\n')
    self.stdscr.refresh()


class FileLogger(BaseFileLogger):
  '''
  File Logger
  '''

  def logging_hv(self, msg):
    orig_covr_img = utils.msg_gt(msg, 'image|orig_covr')
    orig_hide_img = utils.msg_gt(msg, 'image|orig_hide')
    steg_img = utils.msg_gt(msg, 'image|steg')
    dcpt_covr_img = utils.msg_gt(msg, 'image|dcpt_covr')
    dcpt_hide_img = utils.msg_gt(msg, 'image|dcpt_hide')

    # Images saved to disk
    saved_ilabel = ['orig_covr', 'steg', 'dcpt_covr', 'orig_hide', 'dcpt_hide']
    saved_images = [orig_covr_img, steg_img, dcpt_covr_img, orig_hide_img, dcpt_hide_img]
    # Check not None
    saved_ilabel = [
        saved_ilabel[idx] for idx, images in enumerate(saved_images) if not images is None
    ]
    saved_images = [images for images in saved_images if not images is None]

    gmode, mode = params.GMODE, utils.msg_gt(msg, 'message_info|mode')

    if gmode == 'train':
      epoch = utils.msg_gt(msg, 'message_info|epoch')
      batch = utils.msg_gt(msg, 'message_info|batch')

      suffix_map = {
          'train': 't',
          'valid': 'v',
      }
      suffix = suffix_map[mode]

      im = utils.image_comp(saved_images)
      im = utils.norm2pil(im)
      im.save('{}/{:010d}_{:010d}_{}.jpg'.format(params.VISUAL_PATH, epoch, batch, suffix))
    elif gmode == 'inference':
      lidx = utils.msg_gt(msg, 'message_info|lidx')

      suffix = 'i'

      im = utils.image_comp(saved_images)
      im = utils.norm2pil(im)
      im.save('{}/{:010d}_{}.jpg'.format(params.VISUAL_PATH, lidx, suffix))

      detail_Path = params.VISUAL_PATH / '{:010d}'.format(lidx)
      detail_Path.mkdir(parents=True, exist_ok=True)
      for images, ilabel in zip(saved_images, saved_ilabel):
        batches = images.shape[0]
        for idx in range(batches):
          im = utils.norm2pil(images[idx])
          im.save('{}/{:010d}_{:03d}_{}.png'.format(detail_Path, lidx, idx, ilabel))
    else:
      raise RuntimeError('Unexpected global mode: %s' % gmode)
    self.logging_lt(msg)

  def logging_lt(self, msg):
    RT_LOG_PATH = params.LOGGING_PATH / params.RT_LOG_FILE
    json_msg = {k: v for k, v in msg.items() if k != 'image'}

    with RT_LOG_PATH.open(mode='at') as f:
      f.write(json.dumps(json_msg, sort_keys=True, cls=FileLogger.NPJsonEncoder))
      f.write('\n')


class Logger(Task):
  '''
  Logger
  '''

  def __init__(self, msgfactory):
    self.msgfactory = msgfactory
    self.console_logger = ConsoleLogger()
    self.file_logger = FileLogger()

  @property
  def name(self):
    return 'logger'

  def update_runtime_meta(self, msg):
    '''
    Write the runtime meta file for restore running
    '''
    RT_META_PATH = params.LOGGING_PATH / params.RT_META_FILE

    meta = self.msgfactory.create_runtime_meta()
    utils.msg_ud(meta, 'gidx', utils.msg_gt(msg, 'message_info|gidx'))
    utils.msg_ud(meta, 'tidx', utils.msg_gt(msg, 'message_info|tidx'))
    utils.msg_ud(meta, 'vidx', utils.msg_gt(msg, 'message_info|vidx'))

    with RT_META_PATH.open('w') as f:
      json.dump(meta, f)

  def logging_hv(self, msg):
    self.file_logger.logging_hv(msg)
    self.console_logger.logging_hv(msg)
    self.update_runtime_meta(msg)

  def logging_lt(self, msg):
    self.file_logger.logging_lt(msg)
    self.console_logger.logging_lt(msg)

  def apply(self, queue):
    '''
    Main Logging Entry
    '''
    iqueue = queue['post']

    msg = None
    while not (params.SHOULD_FINISH.value == b'post' and iqueue.empty()):
      try:
        msg = iqueue.get(timeout=params.QUEUE_TIMEOUT)
      except Q.Empty:
        continue

      heavy_logging = utils.msg_gt(msg, 'message_info|heavy_logging')

      if heavy_logging:
        self.logging_hv(msg)
      else:
        self.logging_lt(msg)

    if msg:
      self.logging_hv(msg)
    params.SHOULD_FINISH.value = self.bname
    utils.eprint('logger: exit')

