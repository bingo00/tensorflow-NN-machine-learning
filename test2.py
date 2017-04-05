# mainly using tensorflow to build up CNN

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#input data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# create the session
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# function
# create parameter w
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# create parameter b
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# create convolution layer, step 1, fill up 0
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# create pooling layer, kernel 2, step 2, fill up 0
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# first layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# all link layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# calculate the all link layer
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# keep_prob using for not overfitting (like stop training)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#1. cross_entropy calculation
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

#2. create optimizer (local minima typically rate 0.01%)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#3. accuracy calculation 
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#4. initial value of session
sess.run(tf.global_variables_initializer())

#5. looping
for i in range(2000):
    # take out 50 training sample at once
    batch = mnist.train.next_batch(50)
    # print message when loop for 100 multipe 
    if i % 100 == 0:
        # calculate accuracy
        train_accuracy = accuracy.eval(feed_dict={
           x: batch[0], y_: batch[1], keep_prob: 1.0})
        # print
        print("step %d, training accuracy %g" % (i, train_accuracy))
    # do the training module
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
# print the test data accuracy
print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
    }))