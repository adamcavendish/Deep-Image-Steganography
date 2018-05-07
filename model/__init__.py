'''
Model Module
'''

from . import steg_net

_models = [
    steg_net,
]

_dispatcher = {model.name(): model for model in _models}


def get_model_by_name(name):
  '''
  Helper function to obtain the model by its name
  '''
  if name == 'common':
    raise ImportError('common is reserved for utility use')
  return _dispatcher[name]
