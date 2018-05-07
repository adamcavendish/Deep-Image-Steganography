'''
steg_net Model
'''
from ..common import (BaseMessageFactory, BaseGenerator, BasePreprocessor, BasePostprocessor)

from .runner import Runner
from .logger import Logger


def pipeline():
  '''
  Model Pipeline
  '''
  msgfactory = BaseMessageFactory()
  return [
      BaseGenerator(msgfactory),
      BasePreprocessor(),
      Runner(),
      BasePostprocessor(),
      Logger(msgfactory)
  ]


def name():
  '''
  Model Name
  '''
  return 'steg_net'
