'''
Common Preprocessor Module
'''

import queue as Q

import params
import utils

from .task import Task


class BasePreprocessor(Task):
  '''
  Base Preprocessor
  '''

  @property
  def name(self):
    return 'prep'

  def apply(self, queue):
    # The image shape maybe updated before preprocess
    params.INROWS.value = params.INROWS.value
    params.INCOLS.value = params.INCOLS.value
    params.INCNLS.value = params.INCNLS.value

    iqueue = queue['generate']
    oqueue = queue[self.name]

    while not (params.SHOULD_FINISH.value == b'generate' and iqueue.empty()):
      try:
        msg = iqueue.get(timeout=params.QUEUE_TIMEOUT)
      except Q.Empty:
        continue

      utils.msg_ud(msg, 'queue_info|prep', oqueue.qsize())

      mode = utils.msg_gt(msg, 'message_info|mode')
      if mode == 'train':
        covr, hide = utils.msg_gt(msg, 'image|covr/train'), utils.msg_gt(msg, 'image|hide/train')
      elif mode == 'valid':
        covr, hide = utils.msg_gt(msg, 'image|covr/valid'), utils.msg_gt(msg, 'image|hide/valid')
      else:
        raise RuntimeError('Invalid mode: %s' % mode)

      utils.msg_ud(msg, 'image|orig_covr', covr)
      utils.msg_ud(msg, 'image|orig_hide', hide)
      oqueue.put(msg)

    oqueue.close()
    oqueue.join_thread()
    params.SHOULD_FINISH.value = self.bname
    utils.eprint('preprocessor: exit')
