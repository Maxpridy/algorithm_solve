import tensorflow as tf
import numpy as np

dd = np.ones((3,4), dtype='float64')
idex = np.array([[0,1], [1,2], [2,3]])

data = tf.placeholder(tf.float64, [None, 4])
i = tf.placeholder(tf.int32, [None, None])

d = tf.scatter_add(data, tf.ones_like(i), i)

with tf.Session() as sess:

    _d = sess.run(d, feed_dict={data:dd, i:idex})
    print(_d)