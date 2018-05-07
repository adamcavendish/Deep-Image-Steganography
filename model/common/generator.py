'''
Common Generator Module
'''

import json
import queue as Q
import sys
import time

import params
import utils

from .task import Task


class BaseGenerator(Task):
  '''
  Base Generator
  '''

  def __init__(self, message):
    self.message = message

    RT_META_PATH = params.LOGGING_PATH / params.RT_META_FILE
    if params.RESTART or not RT_META_PATH.exists():
      self.gidx = 0
      self.lidx = 0
      self.tidx = 0
      self.vidx = 0
    else:
      with RT_META_PATH.open('r') as meta_f:
        meta = json.load(meta_f)
      self.gidx = utils.msg_gt(meta, 'gidx') + 1
      self.lidx = 0
      TI, VI = params.TRAIN_INTERVAL, params.VALID_INTERVAL
      cycle = self.gidx // (TI + VI)
      offst = self.gidx % (TI + VI)
      self.tidx = cycle * TI + (TI if offst > TI else offst)
      self.vidx = cycle * params.VALID_INTERVAL + (offst if offst > TI else 0)

  @property
  def name(self):
    return 'generate'

  def generator_train(self, queue):
    '''
    Default train generator for one step
    '''
    covr_train, hide_train = queue['covr/train'].get_nowait(), queue['hide/train'].get_nowait()
    epoch = self.tidx // (params.DATASET_TRAIN_SIZE // params.BATCH_SIZE)
    batch = self.tidx % (params.DATASET_TRAIN_SIZE // params.BATCH_SIZE)
    heavy_logging = self.tidx % params.HEAVY_LOGGING_INTERVAL == 0

    msg = self.message.create_message()
    utils.msg_ud(msg, 'queue_info|generator', queue[self.name].qsize())
    utils.msg_ud(msg, 'message_info|gidx', self.gidx)
    utils.msg_ud(msg, 'message_info|lidx', self.lidx)
    utils.msg_ud(msg, 'message_info|tidx', self.tidx)
    utils.msg_ud(msg, 'message_info|vidx', self.vidx)
    utils.msg_ud(msg, 'message_info|epoch', epoch)
    utils.msg_ud(msg, 'message_info|batch', batch)
    utils.msg_ud(msg, 'message_info|mode', 'train')
    utils.msg_ud(msg, 'message_info|heavy_logging', heavy_logging)
    utils.msg_ud(msg, 'image|covr/train', covr_train)
    utils.msg_ud(msg, 'image|hide/train', hide_train)
    self.gidx += 1
    self.lidx += 1
    self.tidx += 1
    return msg

  def generator_valid(self, queue):
    '''
    Default valid generator for one step
    '''
    covr_valid, hide_valid = queue['covr/valid'].get_nowait(), queue['hide/valid'].get_nowait()
    epoch = self.tidx // params.DATASET_TRAIN_SIZE
    batch = self.tidx % params.DATASET_TRAIN_SIZE

    msg = self.message.create_message()
    utils.msg_ud(msg, 'queue_info|generator', queue[self.name].qsize())
    utils.msg_ud(msg, 'message_info|gidx', self.gidx)
    utils.msg_ud(msg, 'message_info|lidx', self.lidx)
    utils.msg_ud(msg, 'message_info|tidx', self.tidx)
    utils.msg_ud(msg, 'message_info|vidx', self.vidx)
    utils.msg_ud(msg, 'message_info|epoch', epoch)
    utils.msg_ud(msg, 'message_info|batch', batch)
    utils.msg_ud(msg, 'message_info|mode', 'valid')
    utils.msg_ud(msg, 'message_info|heavy_logging', True)
    utils.msg_ud(msg, 'image|covr/valid', covr_valid)
    utils.msg_ud(msg, 'image|hide/valid', hide_valid)
    self.gidx += 1
    self.lidx += 1
    self.vidx += 1
    return msg

  def apply(self, queue):
    oqueue = queue[self.name]

    while True:
      epoch = self.tidx // (params.DATASET_TRAIN_SIZE // params.BATCH_SIZE)

      if params.SHOULD_FINISH.value == b'STOP' or epoch > params.TRAIN_MAX_EPOCH:
        oqueue.close()
        oqueue.join_thread()
        params.SHOULD_FINISH.value = self.bname
        utils.eprint('generator: exit')
        break

      cidx = self.gidx % (params.TRAIN_INTERVAL + params.VALID_INTERVAL)
      try:
        if cidx < params.TRAIN_INTERVAL:
          msg = self.generator_train(queue)
          oqueue.put(msg)
        else:
          msg = self.generator_valid(queue)
          oqueue.put(msg)
      except Q.Empty:
        time.sleep(params.QUEUE_TIMEOUT)
        continue
