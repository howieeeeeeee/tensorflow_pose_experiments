# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import os.path
import re
import time

import numpy as np
import tensorflow as tf

import image_processing
import scipy.misc

# import model
import model_2 as model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_steps', 20000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('set', 'train',
                           """Either 'train' or 'validation'.""")

tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")

tf.app.flags.DEFINE_float('learning_rate', 0.0000001,
                            """How many GPUs to use.""")

tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")
tf.app.flags.DEFINE_boolean('is_training', True,
                            """If set, dropout """)
                           
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '/data/huozengwei/train_dir/biwi_4/',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/data/huozengwei/train_dir/aflw_2016/aflw_1/',
                           """Directory where to read model checkpoints.""")

def train(dataset):
  """Train on dataset for a number of steps."""
  # with tf.Graph().as_default(), tf.device('/cpu:0'):
  with tf.Graph().as_default():

    # ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    # if ckpt and ckpt.model_checkpoint_path:
    #     global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

    global_step = tf.Variable(0,trainable=False)
    # global_step = tf.contrib.framework.get_or_create_global_step()

    num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus
    with tf.device('/cpu:0'):
      images, pitchs, yaws, rolls, names = image_processing.distorted_inputs(
        dataset,
        num_preprocess_threads=num_preprocess_threads)
    
    p = tf.expand_dims(pitchs,1)
    y = tf.expand_dims(yaws,1)
    r = tf.expand_dims(rolls,1)
    labels = tf.concat([p, y, r],1)

    train_output = model.inference(images,FLAGS.is_training)
    train_loss = model.losses(train_output, labels)    
    add_global = global_step.assign_add(1)     
    train_op = model.trainning(train_loss, FLAGS.learning_rate, global_step)
   
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    saver = tf.train.Saver()
    
    # ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    # if ckpt and ckpt.model_checkpoint_path:
    #   if os.path.isabs(ckpt.model_checkpoint_path):
    #     # Restores from checkpoint with absolute path.
    #     saver.restore(sess, ckpt.model_checkpoint_path)
    #   else:
    #     # Restores from checkpoint with relative path.
    #     saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
    #                                      ckpt.model_checkpoint_path))

      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/imagenet_train/model.ckpt-0,
      # extract global_step from it.
      # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      # print('Successfully loaded model from %s at step=%s.' %
      #       (ckpt.model_checkpoint_path, global_step))
    # else:
    #   print('No checkpoint file found')
    #   return

    sess.run(tf.global_variables_initializer())
    
    """
    these codes get the variable in conv1

    print(sess.run(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
    w = tf.contrib.framework.get_variables('conv1')
    t = tf.nn.l2_loss(w[0])
    print(sess.run(t))
    """

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
        for step in np.arange(FLAGS.max_steps):
            if coord.should_stop():
                    break
            _, _, tra_loss= sess.run([add_global, train_op, train_loss])
               
            if step % 50 == 0:
                print('Step %d, train loss = %.2f'  %(step, tra_loss))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            
            if step % 2000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()

    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    # try:
    #   print(sess.run(pitchs))
    # except Exception as e:
    #   coord.request_stop(e)
    # coord.request_stop()
    # coord.join(threads)
    # sess.close()
    

    # sv = tf.train.Supervisor()
    # with sv.managed_session() as sess:
    #   print(sess.sun(pitchs))
    
