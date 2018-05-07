import io
import os

import numpy as np
from PIL import Image

import lmdb
import msgpack

from .dataset import Dataset

ILSVRC2012_MDB_PATH = os.environ['ILSVRC2012_MDB_PATH']

class DatasetILSVRC2012(Dataset):
  def __init__(self, train_ratio=0.8, seed=42):
    self.seed = seed
    self.mdb_path = ILSVRC2012_MDB_PATH
    self.env = lmdb.open(self.mdb_path, readonly=True)
    self.whole_size = self.env.stat()['entries']
    self.train_size = int(self.whole_size * train_ratio)
    self.valid_size = self.whole_size - self.train_size
    self.inrows, self.incols, self.incnls = 64, 64, 3

  def get_name(self):
    return 'ILSVRC2012'

  def get_shape(self):
    return self.inrows, self.incols, self.incnls

  def get_whole_size(self):
    return self.whole_size

  def get_train_size(self):
    return self.train_size

  def get_valid_size(self):
    return self.valid_size

  def _fetch_data_in_range(self, batch_size, lower_bound, upper_bound):
    # Image is normalized to [-1, 1]
    np.random.seed(self.seed)
    rand_range = np.arange(lower_bound, upper_bound)
    with self.env.begin() as txn:
      while True:
        image_v = np.zeros(shape=(batch_size, self.inrows, self.incols, self.incnls))
        image_idx = np.random.choice(rand_range, size=batch_size)
        for index in range(batch_size):
          image_rawd = txn.get('{:08d}'.format(image_idx[index]).encode())
          image_info = msgpack.unpackb(image_rawd, encoding='utf-8')
          with Image.open(io.BytesIO(image_info['image'])) as im:
            im = im.resize((self.inrows, self.incols), Image.ANTIALIAS)
            image_data = np.array(im)
          image_v[index, :, :, :] = image_data
        image_v = image_v / 255. * 2 - 1
        yield image_v

  def fetch_train_data(self, batch_size):
    return self._fetch_data_in_range(batch_size, 0, self.train_size)

  def fetch_valid_data(self, batch_size):
    return self._fetch_data_in_range(batch_size, self.train_size, self.whole_size)
