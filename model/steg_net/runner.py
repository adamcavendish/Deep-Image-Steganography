'''
Main Model Runner
'''

import queue as Q
import timeit

import numpy as np
import tensorflow as tf
from recordclass import recordclass

import params
import utils

from ..common import Task

from . import steganography as stg

ModelRuntime = recordclass('ModelRuntime', 'sess summary_writer saver')


class Model(object):
  '''
  Running Model Wrapper
  '''

  def __init__(self):
    model_shape = [params.MNROWS.value, params.MNCOLS.value, params.MNCNLS.value]

    self.g_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    self.g_next_step = tf.assign_add(self.g_step, 1)

    self.covr_img = tf.placeholder(tf.float32, [None, *model_shape], name='covr_image')
    self.hide_img = tf.placeholder(tf.float32, [None, *model_shape], name='hide_image')

    self.steg_img = stg.encrypter(self.covr_img, self.hide_img, is_training=True)
    self.dcpt_img = stg.decrypter(self.steg_img, is_training=True)

    # Used as a dummy placeholder for validation process
    self.dummy = tf.constant(1, name='dummy_constant')

    # Reconstruct Loss
    with tf.variable_scope('rcst_loss'):
      self.rcst_loss = tf.reduce_mean(tf.abs(self.covr_img - self.steg_img), name='value')
    # Decryption Loss
    with tf.variable_scope('dcpt_loss'):
      self.dcpt_loss = tf.reduce_mean(tf.abs(self.dcpt_img - self.hide_img), name='value')
    # Variance Loss
    with tf.variable_scope('rcst_vars'):
      _, self.rcst_vars = tf.nn.moments(self.covr_img - self.steg_img, axes=[1, 2, 3])
      self.rcst_vars = tf.reduce_mean(self.rcst_vars, name='value')
    with tf.variable_scope('dcpt_vars'):
      _, self.dcpt_vars = tf.nn.moments(self.dcpt_img - self.hide_img, axes=[1, 2, 3])
      self.dcpt_vars = tf.reduce_mean(self.dcpt_vars, name='value')

    with tf.variable_scope('loss'):
      self.loss = tf.identity(
          1 / 4 * self.rcst_loss + 1 / 4 * self.rcst_vars + 1 / 4 * self.dcpt_loss +
          1 / 4 * self.dcpt_vars,
          name='loss')

    # Variables
    self.vars = tf.trainable_variables()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      optm = tf.train.AdamOptimizer(params.LEARNING_RATE, beta1=0.5, beta2=0.9)

      grads = optm.compute_gradients(self.loss)
      self.grad_smry = [
          tf.summary.histogram(var.op.name + '/gradient', grad) for grad, var in grads
      ]
      self.optm = optm.apply_gradients(grads)

    self.weight_smry = [tf.summary.histogram(var.op.name + '/weight', var) for var in self.vars]

    smry_lt_lst = [
        tf.summary.scalar('decrypt loss', self.dcpt_loss),
        tf.summary.scalar('reconst loss', self.rcst_loss),
        tf.summary.scalar('decrypt vars', self.dcpt_vars),
        tf.summary.scalar('reconst vars', self.rcst_vars),
        tf.summary.scalar('loss', self.loss),
    ]
    smry_hv_lst = smry_lt_lst + self.grad_smry + self.weight_smry

    self.smry_lt = tf.summary.merge(smry_lt_lst)
    self.smry_hv = tf.summary.merge(smry_hv_lst)

  def apply(self, msg, runtime):
    '''
    Run one step in the queue
    '''
    gmode = params.GMODE
    mode = utils.msg_gt(msg, 'message_info|mode')
    heavy_logging = utils.msg_gt(msg, 'message_info|heavy_logging')

    if gmode == 'train' and mode == 'train':
      msg = self.train_once(msg, runtime)

      if heavy_logging:
        runtime.saver.save(
            runtime.sess,
            str(params.CKPT_PATH / 'model'),
            global_step=self.g_step,
            latest_filename=params.CKPT_FILE,
            meta_graph_suffix='meta',
            write_meta_graph=True,
            write_state=True)
    else:
      msg = self.inference(msg, runtime)
    return msg

  def train_once(self, msg, runtime):
    '''
    Train once
    '''
    utils.msg_ud(msg, 'running|task', 'train_once')

    covr_img_v = utils.msg_gt(msg, 'image|orig_covr')
    hide_img_v = utils.msg_gt(msg, 'image|orig_hide')

    gmode = params.GMODE
    mode = utils.msg_gt(msg, 'message_info|mode')
    heavy_logging = utils.msg_gt(msg, 'message_info|heavy_logging')
    batch_size = params.BATCH_SIZE

    image_shape = [params.INROWS.value, params.INCOLS.value, params.INCNLS.value]
    model_shape = [params.MNROWS.value, params.MNCOLS.value, params.MNCNLS.value]
    mnrows, mncols, _ = image_shape
    inrows, incols, _ = model_shape

    slicer = utils.ImageSlicer(inrows, incols, mnrows, mncols)

    t_beg = timeit.default_timer()
    steg_img_v = np.zeros(shape=(batch_size, *image_shape))
    dcpt_img_v = np.zeros(shape=(batch_size, *image_shape))
    loss_va = []
    rcst_loss_va, rcst_vars_va, dcpt_loss_va, dcpt_vars_va = [], [], [], []
    for row_idx in range(inrows // mnrows):
      for col_idx in range(incols // mncols):
        # slice an image fragment at (row_idx, col_idx)
        covr_img_vs = slicer.slice(covr_img_v, row_idx, col_idx)
        hide_img_vs = slicer.slice(hide_img_v, row_idx, col_idx)

        if gmode == 'train' and mode == 'train':
          optm = self.optm
        else:
          optm = self.dummy

        if heavy_logging:
          smry = self.smry_hv
        else:
          smry = self.smry_lt

        _, smry_v, \
        loss_v, \
        rcst_loss_v, rcst_vars_v, dcpt_loss_v, dcpt_vars_v, \
        steg_img_vs, dcpt_img_vs = runtime.sess.run([
            optm, smry,
            self.loss,
            self.rcst_loss, self.rcst_vars, self.dcpt_loss, self.dcpt_vars,
            self.steg_img, self.dcpt_img
        ], feed_dict={self.covr_img: covr_img_vs, self.hide_img: hide_img_vs})

        slicer.slice_assign(steg_img_v, row_idx, col_idx, steg_img_vs)
        slicer.slice_assign(dcpt_img_v, row_idx, col_idx, dcpt_img_vs)

        loss_va.append(loss_v)
        rcst_loss_va.append(rcst_loss_v)
        rcst_vars_va.append(rcst_vars_v)
        dcpt_loss_va.append(dcpt_loss_v)
        dcpt_vars_va.append(dcpt_vars_v)
    t_end = timeit.default_timer()
    t_diff = t_end - t_beg
    run_time = t_diff

    utils.msg_ud(msg, 'running|timing', run_time)
    utils.msg_ud(msg, 'running|train_cycle_timing', run_time)
    utils.msg_st(msg, 'image|steg', steg_img_v)
    utils.msg_st(msg, 'image|dcpt_covr', None)
    utils.msg_st(msg, 'image|dcpt_hide', dcpt_img_v)
    utils.msg_st(msg, 'post_info|loss', np.average(loss_va))
    utils.msg_st(msg, 'post_info|rcst_loss', np.average(rcst_loss_va))
    utils.msg_st(msg, 'post_info|rcst_vars', np.average(rcst_vars_va))
    utils.msg_st(msg, 'post_info|dcpt_loss', np.average(dcpt_loss_va))
    utils.msg_st(msg, 'post_info|dcpt_vars', np.average(dcpt_vars_va))

    if gmode == 'train' and mode == 'train':
      step_v = runtime.sess.run(self.g_step)
      runtime.summary_writer.add_summary(smry_v, step_v)
      runtime.sess.run(self.g_next_step)

    return msg

  def inference(self, msg, runtime):
    '''
    Inference model
    '''
    covr_img_v = utils.msg_gt(msg, 'image|orig_covr')
    hide_img_v = utils.msg_gt(msg, 'image|orig_hide')

    batch_size = params.BATCH_SIZE

    image_shape = [params.INROWS.value, params.INCOLS.value, params.INCNLS.value]
    model_shape = [params.MNROWS.value, params.MNCOLS.value, params.MNCNLS.value]
    mnrows, mncols, _ = image_shape
    inrows, incols, _ = model_shape

    slicer = utils.ImageSlicer(inrows, incols, mnrows, mncols)

    t_beg = timeit.default_timer()
    steg_img_v = np.zeros(shape=(batch_size, *image_shape))
    dcpt_img_v = np.zeros(shape=(batch_size, *image_shape))
    loss_va = []
    rcst_loss_va, rcst_vars_va, dcpt_loss_va, dcpt_vars_va = [], [], [], []
    for row_idx in range(inrows // mnrows):
      for col_idx in range(incols // mncols):
        # slice an image fragment at (row_idx, col_idx)
        covr_img_vs = slicer.slice(covr_img_v, row_idx, col_idx)
        hide_img_vs = slicer.slice(hide_img_v, row_idx, col_idx)

        loss_v, \
        rcst_loss_v, rcst_vars_v, dcpt_loss_v, dcpt_vars_v, \
        steg_img_vs, dcpt_img_vs = runtime.sess.run([
            self.loss,
            self.rcst_loss, self.rcst_vars, self.dcpt_loss, self.dcpt_vars,
            self.steg_img, self.dcpt_img
        ], feed_dict={self.covr_img: covr_img_vs, self.hide_img: hide_img_vs})

        slicer.slice_assign(steg_img_v, row_idx, col_idx, steg_img_vs)
        slicer.slice_assign(dcpt_img_v, row_idx, col_idx, dcpt_img_vs)

        loss_va.append(loss_v)
        rcst_loss_va.append(rcst_loss_v)
        rcst_vars_va.append(rcst_vars_v)
        dcpt_loss_va.append(dcpt_loss_v)
        dcpt_vars_va.append(dcpt_vars_v)
    t_end = timeit.default_timer()
    t_diff = t_end - t_beg
    run_time = t_diff

    utils.msg_ud(msg, 'running|timing', run_time)
    utils.msg_ud(msg, 'running|train_cycle_timing', None)
    utils.msg_st(msg, 'image|steg', steg_img_v)
    utils.msg_st(msg, 'image|dcpt_covr', None)
    utils.msg_st(msg, 'image|dcpt_hide', dcpt_img_v)
    utils.msg_st(msg, 'post_info|loss', np.average(loss_va))
    utils.msg_st(msg, 'post_info|rcst_loss', np.average(rcst_loss_va))
    utils.msg_st(msg, 'post_info|rcst_vars', np.average(rcst_vars_va))
    utils.msg_st(msg, 'post_info|dcpt_loss', np.average(dcpt_loss_va))
    utils.msg_st(msg, 'post_info|dcpt_vars', np.average(dcpt_vars_va))

    return msg


class Runner(Task):
  '''
  Run Model
  '''

  @property
  def name(self):
    return 'run'

  def apply(self, queue):
    '''
    Run Entry
    '''
    # The model shape is updated here
    params.MNROWS.value = 64
    params.MNCOLS.value = 64
    params.MNCNLS.value = 3

    model = Model()
    if params.RESTART:
      init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # gpu_options = tf.GPUOptions(allow_growth=True)
    # sess_config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session() as sess:
      summary_writer = tf.summary.FileWriter(
          str(params.SUMMARY_PATH), graph=sess.graph, max_queue=32, flush_secs=300)
      saver = tf.train.Saver(
          max_to_keep=5,
          keep_checkpoint_every_n_hours=12,
          pad_step_number=True,
          save_relative_paths=True)

      if params.RESTART:
        sess.run(init)
      else:
        model_path = tf.train.latest_checkpoint(
            str(params.CKPT_PATH), latest_filename=params.CKPT_FILE)
        saver.restore(sess, model_path)

      iqueue = queue['prep']
      oqueue = queue[self.name]

      rt = ModelRuntime(sess, summary_writer, saver)

      while not (params.SHOULD_FINISH.value == b'prep' and iqueue.empty()):
        try:
          msg = iqueue.get(timeout=params.QUEUE_TIMEOUT)
        except Q.Empty:
          continue
          
        utils.msg_ud(msg, 'queue_info|run', oqueue.qsize())

        msg = model.apply(msg, rt)
        oqueue.put(msg)

    oqueue.close()
    oqueue.join_thread()
    params.SHOULD_FINISH.value = self.bname
    utils.eprint('runner: exit')
