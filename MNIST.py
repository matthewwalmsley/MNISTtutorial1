# Import the MNIST data set
from tensorflow.examples.tutorials.mnist import input_data
# Import tensorflow
import tensorflow as tf

# Import data to minst
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create placeholder tensor for x, and variables for W and b
# A placeholder is a tensor that expects a value
# A variable is a modifiable tensor that can be modified by the computation
x = tf.placeholder(tf.float32, [None, 784]) # None means the dimension can be of any length
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Define our model.  matmul is a matrix multiplier function
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Set up a tensor of the correct answers
y_ = tf.placeholder(tf.float32, [None, 10])

# Implement the cross-entropy function -sum(y_ * log(y))
# See http://colah.github.io/posts/2015-09-Visual-Information/
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# reduce_sum adds the elements in the second dimension of y - this is set by reduction_indices = 1
# reduce_mean computes the mean over all examples in the batch

# A more stable version of this is included in the tutorial code
# cross_entropy = tf.reduce_mean(
#      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Define which optimiser we'll use, and the step
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

# Create a session and initialise the variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Run the training loop
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

# Assess the accuracy of our training
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
