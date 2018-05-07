'''
Input Image Data Generators
'''

import contextlib
import queue as Q
import time

import params
import utils


def dataset_generator(queue, dataset, mode, role, batch_size):
  '''
  Generate image data from dataset
  '''
  if mode == 'train':
    dgen = dataset.fetch_train_data
  else:
    dgen = dataset.fetch_valid_data

  # queue-name: one of ['covr/train', 'hide/train', 'covr/valid', 'hide/valid']
  qname = '{}/{}'.format(role, mode)

  for image in dgen(batch_size):
    if params.SHOULD_FINISH.value:
      break
    with contextlib.suppress(Q.Full):
      queue[qname].put(image, timeout=params.QUEUE_TIMEOUT)
  # Setup queue to allow exit without flushing all the data to the pipe
  queue[qname].cancel_join_thread()
  utils.eprint('dataset_generator(%s/%s): exit' % (role, mode))
