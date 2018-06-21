#encoding:utf8
import tensorflow as tf

W = tf.Variable([0.1],dtype=tf.float32)
b = tf.Variable([-0.1],dtype=tf.float32)

x = tf.placeholder(dtype=tf.float32)
linear_regression = W*x+b

y = tf.placeholder(dtype=tf.float32)
loss = tf.reduce_sum(tf.square(linear_regression-y))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

x_train = [1.0, 2.0, 3.0, 6.0, 8.0]
y_train = [4.8, 8.5, 10.4, 21.0, 25.3]
for i in range(10000):
    sess.run(train, {x: x_train, y: y_train})
    if i % 100 == 0:
        print('step:%s, w:%s, b:%s, loss:%s'%(i,sess.run(W), sess.run(b),sess.run(loss,{x:x_train, y:y_train})))

print('w:%s, b:%s, loss:%s'%(sess.run(W), sess.run(b),sess.run(loss,{x:x_train, y:y_train})))




