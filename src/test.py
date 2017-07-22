import numpy as np
import tensorflow as tf

x = tf.placeholder(dtype=tf.float32, shape=[1, 3], name='input')
W = tf.Variable(initial_value=[[1, 2, 3], [4, 5, 6]], dtype=tf.float32, name='weight')
b = tf.Variable(initial_value=[1, 2], dtype=tf.float32, name='bias')
y = tf.matmul(x, tf.transpose(W)) + b
res = tf.reduce_max(y)
grad = tf.gradients(res, x)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(y, feed_dict={x: [[1, 2, 3]]}))
print(sess.run([grad, res], feed_dict={x: [[1, 2, 3]]}))