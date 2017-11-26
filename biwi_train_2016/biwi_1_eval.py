from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf

import net_eval
from biwi_1_data import Biwi_1_Data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/data/huozengwei/tfrecord/biwi_4/',
                           """Path to the processed data, i.e. """
                           """TFRecord of Example protos.""")
tf.app.flags.DEFINE_string('eval_dir', '/data/huozengwei/eval_dir/biwi_2016/biwi_4/',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/data/huozengwei/train_dir/biwi_2016/biwi_4/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_examples', 4051,
                            """Number of examples to run. Note that the eval """
                            """3559 3867 4200 4051""")

def main(_):
  dataset = Biwi_1_Data(subset='validation')
  assert dataset.data_files()
  net_eval.evaluate(dataset)


if __name__ == '__main__':
  tf.app.run()