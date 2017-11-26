from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import time


import numpy as np
import tensorflow as tf

import image_processing
import scipy.misc
import model
# import model_2 as model


FLAGS = tf.app.flags.FLAGS


# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation' or 'train'.""")

def evaluate(dataset,n):
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels from the dataset.
    images, pitchs, yaws, rolls, names = image_processing.inputs(dataset)
   
    if n==1:
      images_leftup = images[:,0:85,0:85,:]
      images_leftup = tf.image.resize_bilinear(images_leftup, [32, 32],
                                       align_corners=False)
      eval_output = model.inference(images_leftup)

    if n==2:
      images_rightup = images[:,14:99,0:85,:]
      images_rightup = tf.image.resize_bilinear(images_rightup, [32, 32],
                                       align_corners=False)
      eval_output = model.inference(images_rightup)

    if n==3:
      images_leftdown = images[:,0:85,14:99,:]
      images_leftdown = tf.image.resize_bilinear(images_leftdown, [32, 32],
                                       align_corners=False)
      eval_output = model.inference(images_leftdown)

    if n==4:
      images_rightdown = images[:,14:99,14:99,:]
      images_rightdown = tf.image.resize_bilinear(images_rightdown, [32, 32],
                                       align_corners=False)
      eval_output = model.inference(images_rightdown)

    if n==5:
	    images_center = images[:,0:7,92:99,:]
	    images_center = tf.image.resize_bilinear(images, [32, 32],
	                                       align_corners=False)
	    eval_output = model.inference(images_center)
    
    p = tf.expand_dims(pitchs,1)
    y = tf.expand_dims(yaws,1)
    r = tf.expand_dims(rolls,1)
    labels = tf.concat([p, y, r],1)
   
    # Calculate predictions.
 
    error_op = eval_output-labels
    # acc_op = tf.abs(eval_output-labels)

    saver = tf.train.Saver()

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                            graph_def=graph_def)

    with tf.Session() as sess:
	    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
	    if ckpt and ckpt.model_checkpoint_path:
	      if os.path.isabs(ckpt.model_checkpoint_path):
	        # Restores from checkpoint with absolute path.
	        saver.restore(sess, ckpt.model_checkpoint_path)
	      else:
	        # Restores from checkpoint with relative path.
	        saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
	                                         ckpt.model_checkpoint_path))

	      # Assuming model_checkpoint_path looks something like:
	      #   /my-favorite-path/imagenet_train/model.ckpt-0,
	      # extract global_step from it.
	      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
	      print('Successfully loaded model from %s at step=%s.' %
	            (ckpt.model_checkpoint_path, global_step))
	    else:
	      print('No checkpoint file found')
	      return

	    # Start the queue runners.
	    coord = tf.train.Coordinator()
	    try:
	      threads = []
	      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
	        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
	                                         start=True))

	      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
	      # Counts the number of correct predictions.
	      
	      total_sample_count = num_iter * FLAGS.batch_size
	      step = 0
	      
	      total_error = [0.0, 0.0, 0.0]
	      total_acc = [0.0, 0.0, 0.0]
	      
	      g_p_error = []

	      print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
	      start_time = time.time()
	      while step < num_iter and not coord.should_stop():
	        error = sess.run(error_op)
	        e = np.sum(np.abs(error),axis=0)
	        # acc_tmp = sess.run(acc_op)

	        # l = sess.run(labels)
	        # e = sess.run(eval_output)

	        # print(l)
	        # print(e)
	        # print(sess.run(names))
	        # break

	        g_p_error.extend(error)

	        acc_tmp = np.abs(error)
	        acc_tmp[acc_tmp<5] = 1
	        acc_tmp[acc_tmp>=5] = 0
	        acc_tmp_sum = np.sum(acc_tmp,axis=0)
	     
	        total_error += e
	        total_acc += acc_tmp_sum
	        step += 1
	        if step % 20 == 0:
	          duration = time.time() - start_time
	          sec_per_batch = duration / 20.0
	          examples_per_sec = FLAGS.batch_size / sec_per_batch
	          print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
	                'sec/batch)' % (datetime.now(), step, num_iter,
	                                examples_per_sec, sec_per_batch))
	          start_time = time.time()

	      # Compute precision @ 1.
	      mae = total_error / total_sample_count
	      acc = total_acc / total_sample_count

	      print('%s: pitch_mae = %.4f yaw_mae = %.4f roll_mae = %.4f pitch_acc = %.4f yaw_acc = %.4f roll_acc = %.4f [%d examples]' %
	            (datetime.now(), mae[0], mae[1], mae[2], acc[0], acc[1], acc[2], total_sample_count))

	      g_p = np.array(g_p_error)
	      # p = np.array(predict_label)
	  
	      np.savetxt("error_"+str(n)+".txt",g_p)
	      # np.savetxt("predict_label_"+str(n)+".txt",p)
	     
	      # summary = tf.Summary()
	      # summary.ParseFromString(sess.run(summary_op))
	      # summary.value.add(tag='Precision @ 1', simple_value=precision_at_1)
	      # summary.value.add(tag='Recall @ 5', simple_value=recall_at_5)
	      # summary_writer.add_summary(summary, global_step)

	    except Exception as e:  # pylint: disable=broad-except
	      coord.request_stop(e)

	    coord.request_stop()
	    coord.join(threads, stop_grace_period_secs=10)