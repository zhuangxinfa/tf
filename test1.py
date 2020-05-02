import tensorflow as tf
a = tf.Variable(1)
b = tf.constant(1)
aplus1 = tf.add(a,b)
update = tf.assign(a,aplus1)
init  = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        print (sess.run(update))