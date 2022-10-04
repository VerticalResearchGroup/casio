import cudaprofile
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

#tf.debugging.set_log_device_placement(True)

import sys
import os

CASIO=os.environ.get('CASIO')
sys.path.append(f'{CASIO}/utils')
import cudaprofile

from tensorflow_wrapper import benchmark_wrapper
import params


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    x = tf.placeholder(tf.float32, shape=(None, 128))
    w = tf.Variable(tf.random_normal([128, 64]))
    b = tf.Variable(tf.random_normal([64]))
    h = tf.nn.xw_plus_b(x, w, b)
    y = tf.nn.relu(h)
    loss = tf.reduce_mean(tf.square(y - tf.ones_like(y)))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    tf.global_variables_initializer().run()

    def roi(run_options=None, run_metadata=None):
      _ = sess.run(train_step, options=run_options, run_metadata=run_metadata)

    benchmark_wrapper('tftest', roi, sess)
