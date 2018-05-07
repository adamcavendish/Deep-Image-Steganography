import pathlib
import io
import math
import multiprocessing as mp

import numpy as np
from PIL import Image

from . import ilsvrc2012

_dispatcher = {
    'ILSVRC2012': ilsvrc2012.DatasetILSVRC2012
}


def get_dataset_by_name(name):
  return _dispatcher[name]
