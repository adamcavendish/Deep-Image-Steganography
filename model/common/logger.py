'''
Common Logger Module
'''
import curses
import json
import sys

import numpy as np

import utils


class BaseConsoleLogger(object):
  '''
  Base Console Logger

  The Console View Example:
  Last Validation:
      Validation:
          gidx: 00031882, tidx: _____N/A, vidx: 00000318
          rcst_loss: 2.411e-3, rcst_vars: 3.811e-4, dcpt_loss: 3.117e-3, dcpt_vars: 8.191e-4
  Last Train:
      Train:
          gidx: 00031881, tidx: 00031799, vidx:      N/A
          rcst_loss: 2.411e-3, rcst_vars: 3.811e-4, dcpt_loss: 3.117e-3, dcpt_vars: 8.191e-4
  '''

  def __init__(self):
    self.stdscr = curses.initscr()
    self._line_pattern = None

  def __del__(self):
    # @FIXME: curses.endwin magically disappears
    try:
      curses.endwin()
      print('curses endwin is correctly called', file=sys.stdout, flush=True)
      print('curses endwin is correctly called', file=sys.stderr, flush=True)
    except AttributeError:
      pass

  @property
  def line_pattern(self):
    '''
    Line Pattern: List[List[Tuple(str, str)]]

    A line pattern of: [
        [('gidx', 'message_info|gidx'), ('tidx', 'message_info|tidx')],
        [('vidx', 'message_info|vidx'), ('lidx', 'message_info|lidx')],
        [
            ('loss1', 'post_info|loss1'),
            ('loss2', 'post_info|loss2')
            ('loss3', 'post_info|loss3')
            ('loss4', 'post_info|loss4')
        ]
    ]

    May result in printed version of:
       gidx: 0000001,  tidx: 0000001
       vidx:     N/A,  lidx:     N/A
      loss1: 1.33e-8, loss2: 1.82e-3, loss3: 7.81e-2, loss4: 9.56e-7
    '''
    return self._line_pattern

  @line_pattern.setter
  def line_pattern(self, value):
    self._line_pattern = value

  @line_pattern.deleter
  def line_pattern(self):
    del self._line_pattern

  @staticmethod
  def _FORMAT_NAME_F(name):
    return '%10s' % name

  _FORMAT_TYPE_F = (
      ((int, np.integer), lambda d: '%010d' % d),
      ((float, np.floating), lambda d: '%10.3e' % d),  # pylint: disable=E1101
      ((str, ), lambda d: '%10s' % d),
      ((type(None), ), lambda _: '%10s' % 'N/A'))

  @staticmethod
  def _format_value(value):
    '''
    Format one value using BaseConsoleLogger._FORMAT_TYPE_F
    '''
    for tf_tpl in BaseConsoleLogger._FORMAT_TYPE_F:
      types, func = tf_tpl
      for t in types:
        if isinstance(value, t):
          return func(value)
    raise RuntimeError('Unexpected type: {}, value={}'.format(type(value), value))

  @staticmethod
  def _format_access(msg, key):
    return BaseConsoleLogger._format_value(utils.msg_gt(msg, key))

  @staticmethod
  def _format_one(msg, name, key):
    '''
    _format_one(msg, 'gidx',  'message_info|gidx') => 'gidx': 0000031882
    _format_one(msg, 'loss', 'post_info|the_loss') => 'loss':  7.181e-03
    '''
    return '%s: %s' % (BaseConsoleLogger._FORMAT_NAME_F(name),
                       BaseConsoleLogger._format_access(msg, key))

  def log_one_msg(self, msg, indent=0):
    '''
    logging one message example:
             gidx: 0000031882,      tidx:        N/A,      vidx: 0000000318
        rcst_loss:  2.412e-03, rcst_vars:  3.812e-04, dcpt_loss:  3.116e-03
    '''
    indent_space = indent * ' '
    if not self.line_pattern:
      raise RuntimeError("Line Pattern is not setup yet!")

    all_lines = []
    for pattern_curr_line in self.line_pattern:
      line_kv = [BaseConsoleLogger._format_one(msg, name, keys) for name, keys in pattern_curr_line]
      line = indent_space + ', '.join(line_kv)
      all_lines.append(line)
    ret = '\n'.join(all_lines)

    return ret

  def logging_hv(self, msg):
    '''
    Heavyweight logging
    '''
    raise NotImplementedError

  def logging_lt(self, msg):
    '''
    Lightweight logging
    '''
    raise NotImplementedError


class BaseFileLogger(object):
  '''
  Base File Logger
  '''

  class NPJsonEncoder(json.JSONEncoder):
    '''
    Derived JSON Encoder for numpy dtypes
    '''

    def default(self, o):  # pylint: disable=E0202
      if isinstance(o, np.integer):
        return int(o)
      elif isinstance(o, np.floating):  # pylint: disable=E1101
        return float(o)
      return json.JSONEncoder.default(self, o)

  def logging_hv(self, msg):
    '''
    Heavyweight logging
    '''
    raise NotImplementedError

  def logging_lt(self, msg):
    '''
    Lightweight logging
    '''
    raise NotImplementedError
