'''
Message Factory Module
'''

class BaseMessageFactory(object):
  '''
  Common Base Message
  '''

  def create_message(self):
    '''
    The message interface for interprocess communication

    Use '|' as seperator is fine

    usage:
      utils.msg_ud(msg, 'message_info|gidx', 10)
      utils.msg_ud(msg, 'image|covr/train', image)
    '''
    return {
        'message_info': {
            'gidx': None,           # Global message index number, restored after restart
            'lidx': None,           # Local message index number, re-init to zero after restart
            'tidx': None,           # Training message index number
            'vidx': None,           # Validation message index Number
            'epoch': None,          # Corresponding index number transformed to epoch number
            'batch': None,          # Corresponding index number transformed to batch number
            'mode': None,           # Local Mode (for current message)
            'heavy_logging': None,  # Whether to write images, or any time-consuming stuffs
        },
        'queue_info': {
            'generator': None,
            'prep': None,
            'run': None,
            'post': None,
        },
        'running': {
            'timing': None,
            'train_cycle_timing': None,
            'task': None,
        },
        'image': {
            'covr/train': None,
            'hide/train': None,
            'covr/valid': None,
            'hide/valid': None,
            'orig_covr': None,
            'orig_hide': None,
            'steg': None,
            'dcpt_covr': None,
            'dcpt_hide': None,
        },
        'post_info': {
            'output_path': None,
        }
    } # yapf: disable

  def create_runtime_meta(self):
    '''
    Create Runtime Meta
    '''
    return {
        'gidx': 0,
        'tidx': 0,
        'vidx': 0,
    }
