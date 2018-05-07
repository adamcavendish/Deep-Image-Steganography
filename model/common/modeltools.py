import queue as Q
import sys
import time

import params
import utils


def terminate_pipeline(model, expect, queue):
  params.SHOULD_FINISH.value = b'STOP'
  utils.eprint('SHOULD_FINISH set to STOP')

  # Setup cleanup queue to allow exit without flushing
  if queue:
      queue.cancel_join_thread()

  while True:
    # Cleanup the Queue
    if queue:
      while True:
        try:
          queue.get(block=False)
        except Q.Empty:
          break
    # Check exit condition
    if expect is None:
      break
    if params.SHOULD_FINISH.value == expect:
      break
    # Wait for some time
    time.sleep(params.QUEUE_TIMEOUT)

  params.SHOULD_FINISH.value = model.bname
  utils.eprint('%s: exit' % model.name)
