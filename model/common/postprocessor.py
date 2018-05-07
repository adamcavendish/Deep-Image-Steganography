'''
Common Postprocessor Module
'''

import queue as Q

import params
import utils

from .task import Task


class BasePostprocessor(Task):
  '''
  Base Postprocessor
  '''

  @property
  def name(self):
    return 'post'

  def apply(self, queue):
    iqueue = queue['run']
    oqueue = queue[self.name]

    while not (params.SHOULD_FINISH.value == b'run' and iqueue.empty()):
      try:
        msg = iqueue.get(timeout=params.QUEUE_TIMEOUT)
      except Q.Empty:
        continue

      utils.msg_ud(msg, 'queue_info|post', oqueue.qsize())
      oqueue.put(msg)

    oqueue.close()
    oqueue.join_thread()
    params.SHOULD_FINISH.value = self.bname
    utils.eprint('postprocessor: exit')
