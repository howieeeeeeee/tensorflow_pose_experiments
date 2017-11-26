from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf

import net_eval_2 as net_eval
from biwi_1_data import Biwi_1_Data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/data/huozengwei/tfrecord/bu_1/',
                           """Path to the processed data, i.e. """
                           """TFRecord of Example protos.""")
tf.app.flags.DEFINE_string('eval_dir', '/data/huozengwei/eval_dir/bu_2014/bu_1/',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/data/huozengwei/train_dir/bu_2014/bu_1/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_examples', 1800,
                            """Number of examples to run. Note that the eval """
                            """4876 4877 4877 4877 4877""")

def main(_):
  dataset = Biwi_1_Data(subset='validation')
  assert dataset.data_files()
  net_eval.evaluate(dataset,5)



if __name__ == '__main__':
  tf.app.run()
