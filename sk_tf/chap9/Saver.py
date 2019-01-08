import tensorflow as tf

theta = tf.Variable(tf.random_uniform([n+1,1], -1.0, 1.0), name='theta')
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

