'''
Base Task Module
'''


class Task(object):
  '''
  Base Task
  '''

  @property
  def name(self):
    '''
    Task name
    '''
    raise NotImplementedError

  @property
  def bname(self):
    '''
    Task name in bytes
    '''
    return self.name.encode()

  def apply(self, queue):
    '''
    Task apply
    '''
    raise NotImplementedError
