'''
Main Logics
'''

import matplotlib as mpl
mpl.use('Agg')

import argparse
import multiprocessing as mp

import numpy as np
import tensorflow as tf

import dataset_tools
import generators as gns
import model
import params
import utils


if __name__ == '__main__':
  queue = {
      'covr/train': mp.Queue(maxsize=params.MAJOR_QUEUE_SIZE),
      'covr/valid': mp.Queue(maxsize=params.MINOR_QUEUE_SIZE),
      'hide/train': mp.Queue(maxsize=params.MAJOR_QUEUE_SIZE),
      'hide/valid': mp.Queue(maxsize=params.MINOR_QUEUE_SIZE),
  }

  # Arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', default='steg_net')
  parser.add_argument('--covr_dataset', default='ILSVRC2012')
  parser.add_argument('--hide_dataset', default='ILSVRC2012')
  parser.add_argument('--train_max_epoch', type=int, default=5)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--restart', action='store_true', default=False)
  parser.add_argument('--global_mode', default='train', choices=['train', 'inference'])
  args = parser.parse_args()

  M = model.get_model_by_name(args.model)

  # Training
  params.GMODE = args.global_mode
  params.TRAIN_MAX_EPOCH = args.train_max_epoch
  params.RESTART = args.restart

  # Datasets
  covr_dataset_name = args.covr_dataset
  hide_dataset_name = args.hide_dataset

  covr_ds = dataset_tools.get_dataset_by_name(covr_dataset_name)(
      train_ratio=0.8, seed=params.DATASET_TRAIN_SEED)
  hide_ds = dataset_tools.get_dataset_by_name(hide_dataset_name)(
      train_ratio=0.8, seed=params.DATASET_VALID_SEED)

  covr_rows, covr_cols, covr_cnls = covr_ds.get_shape()
  hide_rows, hide_cols, hide_cnls = hide_ds.get_shape()
  params.INROWS.value = max(covr_rows, hide_rows)
  params.INCOLS.value = max(covr_cols, hide_cols)
  params.INCNLS.value = max(covr_cnls, hide_cnls)

  params.DATASET_TRAIN_SIZE = max(covr_ds.get_train_size(), hide_ds.get_train_size())
  params.DATASET_VALID_SIZE = max(covr_ds.get_valid_size(), hide_ds.get_valid_size())

  # Seeds
  np.random.seed(params.SEED)
  tf.set_random_seed(params.SEED)

  params.BATCH_SIZE = args.batch_size
  batches = max(covr_ds.get_train_size(), hide_ds.get_train_size()) // params.BATCH_SIZE

  # Paths
  params.SCRIPT_PATH = utils.fpath(__file__)
  params.CKPT_PATH, params.VISUAL_PATH, params.SUMMARY_PATH, params.LOGGING_PATH = \
      utils.prepare_dirs(
          params.SCRIPT_PATH, params.GMODE,
          {
              'model': M.name(),
              'batch_size': params.BATCH_SIZE,
              'covr_ds': covr_ds.get_name(),
              'hide_ds': hide_ds.get_name(),
          })
  if not (params.CKPT_PATH / params.CKPT_FILE).exists() or not (
      params.LOGGING_PATH / params.RT_META_FILE).exists():
    params.RESTART = True

  print('        model: %s' % M.name())
  print('cover dataset: %s' % covr_ds.get_name())
  print(' hide dataset: %s' % hide_ds.get_name())
  print('   batch size: %s' % params.BATCH_SIZE)
  print('batches/epoch: %s' % batches)
  print('    max epoch: %s' % params.TRAIN_MAX_EPOCH)
  print('      restart: %s' % str(params.RESTART))

  # Dataset
  utils.start_process(
      'train/covr',
      gns.dataset_generator,
      args=(queue, covr_ds, 'train', 'covr', params.BATCH_SIZE))
  utils.start_process(
      'train/hide',
      gns.dataset_generator,
      args=(queue, hide_ds, 'train', 'hide', params.BATCH_SIZE))
  utils.start_process(
      'valid/covr',
      gns.dataset_generator,
      args=(queue, covr_ds, 'valid', 'covr', params.BATCH_SIZE))
  utils.start_process(
      'valid/hide',
      gns.dataset_generator,
      args=(queue, hide_ds, 'valid', 'hide', params.BATCH_SIZE))

  # Main Process Pipeline
  for task in M.pipeline():
    queue[task.name] = mp.Queue(maxsize=params.MAJOR_QUEUE_SIZE)
    utils.start_process(task.name, task.apply, args=(queue, ))
  utils.join_all_processes()
