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

import model
# import model_2 as model


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('set', 'train',
                           """Either 'train' or 'validation'.""")

# Flags governing the hardware employed for running TensorFlow.
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")

tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                            """How many GPUs to use.""")

tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '/data/huozengwei/train_dir/biwi_4/',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")


def train(dataset):
  """Train on dataset for a number of steps."""
  # with tf.Graph().as_default(), tf.device('/cpu:0'):
  with tf.Graph().as_default():

    global_step = tf.Variable(0,trainable=False)

    num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus
    with tf.device('/cpu:0'):
      images, pitchs, yaws, rolls, names = image_processing.distorted_inputs(
        dataset,
        num_preprocess_threads=num_preprocess_threads)
    
    p = tf.expand_dims(pitchs,1)
    y = tf.expand_dims(yaws,1)
    r = tf.expand_dims(rolls,1)
    labels = tf.concat([p, y, r],1)

    train_output = model.inference(images)
    train_loss = model.losses(train_output, labels) 

    add_global = global_step.assign_add(1)  
       
    train_op = model.trainning(train_loss, FLAGS.learning_rate, global_step)
   
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    saver = tf.train.Saver()
    
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
                gs = sess.run(global_step)
                print('Step %d, train loss = %.2f, global_step= %d'  %(step, tra_loss, gs))
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
    