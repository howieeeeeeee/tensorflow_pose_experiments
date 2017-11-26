# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

#%%
def inference(images):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    
    conv1 = tf.layers.conv2d(
        inputs=images,
        filters=16,
        kernel_size=[5, 5],
        padding='valid',
        activation=tf.nn.tanh,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.001),
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
        name='conv1')
    
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
        pool_size=[2,2],
        strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=20,
        kernel_size=[5, 5],
        padding='valid',
        activation=tf.nn.tanh,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.001),
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
        name='conv2')

    pool2 = tf.layers.max_pooling2d(inputs=conv2,
        pool_size=[2,2],
        strides=2)

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=20,
        kernel_size=[3, 3],
        padding='valid',
        activation=tf.nn.tanh,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.001),
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=120,
        kernel_size=[3, 3],
        padding='valid',
        activation=tf.nn.tanh,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.001),
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))

    conv4_flat = tf.reshape(conv4,[-1,1*1*120])

    dense1 = tf.layers.dense(inputs=conv4_flat,units=84,
        activation=tf.nn.tanh,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.001),
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))

    output = tf.layers.dense(inputs=dense1,units=3,
      kernel_regularizer = tf.contrib.layers.l2_regularizer(0.001),
      kernel_initializer = tf.truncated_normal_initializer(stddev=0.01))
    
    return output
   

#%%
def losses(output, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]
        
    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        squared_deltas = tf.square(output-labels)
        loss = tf.reduce_sum(squared_deltas)
        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = loss + regularization_loss
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

#%%
def trainning(loss, learning_rate):
    '''Training ops, the Op returned by this function is what must be passed to 
        'sess.run()' call to cause the model to train.
        
    Args:
        loss: loss tensor, from losses()
        
    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op

