import tensorflow as tf
import numpy as np
import pandas as pd
X_ = pd.read_csv('X.csv').values
Y_ = pd.read_csv('Y.csv').values


tf.reset_default_graph()
# These are the number of nodes we want to have in each layer,
# we can set these values to whatever we want.
num_nodes_hl1 = 32
num_nodes_hl2 = 32
num_nodes_hl3 = 32

# Number of output classes equals 10
# Our classes are also in one hot encoding
n_classes = 2

# The size of our imput data is 784
input_dim = np.array(X_).shape[1]

# We can set our batch size to whatever we can fit into memory
batch_size = 32

# Determine how many epochs we would like to use
hm_epoch = 3

# Set the height and wide of model
# These are x and y variables and they must be one dimensional each
# So the first dimension always equals None
# The second must equal the dimension of your X and y variable

x = tf.placeholder(tf.float32, shape=[None, input_dim], name='x_placeholder')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='y_placeholder')

# Initialize hidden layer's variables (Inputs, Weights, and Biases) and take random sample from batch training data

# Shape equals input data, and number of nodes
hidden_layer_1 = {'weights': tf.Variable(tf.random_normal(shape=[input_dim, num_nodes_hl1], name='hidden_layer_1')),
                  'biases': tf.Variable(tf.random_normal(shape=[num_nodes_hl1]))}

# The output shape of hidden_layer_1 is the num_nodes_hl1, so that is the input for the next layer
hidden_layer_2 = {'weights': tf.Variable(tf.random_normal(shape=[num_nodes_hl1, num_nodes_hl2], name='hidden_layer_2')),
                  'biases': tf.Variable(tf.random_normal(shape=[num_nodes_hl2]))}

hidden_layer_3 = {'weights': tf.Variable(tf.random_normal(shape=[num_nodes_hl2, num_nodes_hl3], name='hidden_layer_3')),
                  'biases': tf.Variable(tf.random_normal(shape=[num_nodes_hl3]))}

# Lastly we need the output layer, which must have the same output shape as our number of classes or our y shape
output_layer = {'weights': tf.Variable(tf.random_normal(shape=[num_nodes_hl3, n_classes], name='output_layer')),
                'biases': tf.Variable(tf.random_normal(shape=[n_classes]))}

# Build models function  (input * weights) + biases
# input = data
# weights = hidden_layer_n['weights']
# biases = hidden_layer_n['biases']

l1 = tf.add(tf.matmul(x, hidden_layer_1['weights']), hidden_layer_1['biases'])

# Add activation function to run this data
l1 = tf.nn.relu(l1, name='relu_l1')

l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
l2 = tf.nn.relu(l2, name='relu_l2')

l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
l3 = tf.nn.relu(l3, name='relu_l3')

y_predicted = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])


# The same reduce mean function but we also use softmax, a mulitinomial logistical classifier.

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_predicted, name='loss'))

optimizer = tf.train.AdamOptimizer().minimize(loss, name='optimizer')

# Initialize all global variables
init = tf.global_variables_initializer()

# Activate our sessions and repeat over for loop
with tf.Session() as sess:
    with tf.Graph().as_default():
        # compile all the tf assigned variables (tf.Variables)

        sess.run(init)

        # Initialize our TensorBoard writer for our NN
        writer = tf.summary.FileWriter('./graphs/new_nn', sess.graph)

        # Loop over the defined number of epochs
        for epoch in range(hm_epoch):
            # initialize loss at 0
            total_loss = 0

            # Initialize for loop for batches
            for step in range(int(len(X_) / batch_size)):
                randidx = np.random.randint(len(X_), size=batch_size)
                epoch_x = np.array(X_)[randidx]

                epoch_y = np.array(Y_)[randidx]

                # we use _ because it returns a tuple where the first value is always none, so we dont care about it
                _, l = sess.run([optimizer, loss], feed_dict={x: epoch_x, y: epoch_y})

                # add loss to epoch loss
                total_loss += l

            # Print out our total_loss at each Epoch
            print('Epoch ', epoch, ' completed out of ', hm_epoch, ', loss: ', total_loss)

        # close the writer when you're done using it
        writer.close()

