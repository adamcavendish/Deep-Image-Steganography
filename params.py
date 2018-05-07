'''
Parameters
'''
import ctypes
import multiprocessing as mp

# Paths
SCRIPT_PATH = None
CKPT_PATH = None
VISUAL_PATH = None
SUMMARY_PATH = None
LOGGING_PATH = None
CKPT_FILE = 'checkpoint'
RT_LOG_FILE = 'runtime.log'
RT_META_FILE = 'runtime.meta'

# Randomization
SEED = 42

# Message
GMODE = None
MAJOR_QUEUE_SIZE = 32
MINOR_QUEUE_SIZE = 1
TRAIN_INTERVAL = 50000
VALID_INTERVAL = 1
HEAVY_LOGGING_INTERVAL = 2000
QUEUE_TIMEOUT = 0.5

# Training
RESTART = None
BATCH_SIZE = None
LEARNING_RATE = 1e-5
TRAIN_MAX_EPOCH = None
'''
SHOULD_FINISH:
  Start to terminate the pipeline.
  1. If the whole process is clean and good, it should stay equalling to empty string.
  2. If anyone sets it to b'STOP', the generator starts to terminate the whole pipeline.
  3. If current task has correctly finished, it should be set to the current task name.
     And therefore, the next task, can base on this parameter to infer whether to
     start to terminate itself or not.
'''
SHOULD_FINISH = mp.Array(ctypes.c_char, 128)

# Dataset
DATASET_TRAIN_SIZE = None
DATASET_VALID_SIZE = None
DATASET_TRAIN_SEED = 4201
DATASET_VALID_SEED = 4202

# Image and Model
INROWS, INCOLS, INCNLS = mp.Value('I', 0), mp.Value('I', 0), mp.Value('I', 0)
MNROWS, MNCOLS, MNCNLS = mp.Value('I', 0), mp.Value('I', 0), mp.Value('I', 0)
