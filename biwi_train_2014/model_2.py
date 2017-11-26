from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def _variable_on_cpu(name, shape, initializer):

    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype = dtype)
    return var
def _variable_with_weight_decay(name, shape, stddev, wd):

    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev,dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def inference(images):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                            shape=[5,5,1,16],
                                            stddev=5e-2,
                                            wd=0.1)
        conv = tf.nn.conv2d(images, kernel,[1,1,1,1], padding='VALID')
        biases = _variable_on_cpu('biases',[16], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.tanh(pre_activation, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1],
                            padding='VALID', name='pool1')
    
    # conv2 
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                            shape=[5,5,16,20],
                                            stddev=5e-2,
                                            wd=0.1)
        conv = tf.nn.conv2d(pool1, kernel, [1,1,1,1], padding='VALID')
        biases = _variable_on_cpu('biases',[20], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.tanh(pre_activation, name=scope.name)
   
    # pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1],
                            padding='VALID', name='pool2')
    
    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                            shape=[3,3,20,20],
                                            stddev=5e-2,
                                            wd=0.1)
        conv = tf.nn.conv2d(pool2, kernel, [1,1,1,1], padding='VALID')
        biases = _variable_on_cpu('biases',[20], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv,biases)
        conv3 = tf.nn.tanh(pre_activation, name=scope.name)
    
    # conv4
    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights',
                                            shape=[3,3,20,120],
                                            stddev=5e-2,
                                            wd=0.1)
        conv = tf.nn.conv2d(conv3, kernel, [1,1,1,1], padding='VALID')
        biases = _variable_on_cpu('biases',[120], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv,biases)
        conv4 = tf.nn.tanh(pre_activation, name=scope.name)
    

   # fc1
    with tf.variable_scope('fc1') as scope:
        reshape = tf.reshape(conv4,[-1,1*1*120])
        weights = _variable_with_weight_decay('weights',shape=[120,84],
                                                stddev=0.04,wd=0.004)
        biases = _variable_on_cpu('biases',[84],tf.constant_initializer(0.1))
        fc1 = tf.nn.tanh(tf.matmul(reshape,weights)+biases,name=scope.name)

    # fc2
    with tf.variable_scope('fc2') as scope:
        weights = _variable_with_weight_decay('weights',shape=[84,3],
                                                stddev=0.04,wd=0.004)
        biases = _variable_on_cpu('biases',[3],tf.constant_initializer(0.1))
        output = tf.matmul(fc1,weights)+biases

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
        square_loss = tf.reduce_sum(squared_deltas)
        
        tf.add_to_collection('losses',square_loss)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')

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
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op