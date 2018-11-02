'''
Resotre from ckpt and meta

'''

import tensorflow

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('/tmp/model.ckpt.meta')
    saver.restore(sess, "/tmp/model.ckpt")
