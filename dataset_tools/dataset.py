class Dataset(object):
  def __init__(self, train_ratio=0.8, seed=42):
    raise NotImplementedError

  def get_name(self):
    raise NotImplementedError

  def get_shape(self):
    raise NotImplementedError

  def get_whole_size(self):
    raise NotImplementedError

  def get_train_size(self):
    raise NotImplementedError

  def get_valid_size(self):
    raise NotImplementedError

  def fetch_train_data(self, batch_size):
    raise NotImplementedError

  def fetch_valid_data(self, batch_size):
    raise NotImplementedError
