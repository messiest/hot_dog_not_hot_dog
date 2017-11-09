import tensorflow as tf
import numpy as np
import os

def remove_logs():
    logs_length = len(os.listdir('logs'))
    logs = os.listdir('logs')
    if  logs_length >= 1:
        for i in range(logs_length - 1):
            os.remove(logs[i])

def get_data():
    import pandas as pd
    from sklearn.model_selection import train_test_split

    X_ = pd.read_csv('X.csv').values
    Y_ = pd.read_csv('Y.csv').values

    X_ = np.array(X_).reshape(-1, 28, 28, 1)
    print(X_.shape)
    print(Y_.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.2, random_state=42, )

    print('Images Read')

    return X_train, X_test, y_train, y_test


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_nn(epochs=10, learning_rate=0.01):

    # Get data and TTS
    X_train, X_test, y_train, y_test = get_data()

    # Config
    model_path = "conv_model/model.ckpt"
    batch_size = 100
    logs_path = "logs"
    training_epochs = epochs
    learning_rate = learning_rate
    n_classes = 2
    input_dim = [None, 28, 28, 1]
    keep_rate = 0.8

    # Reset Graph
    tf.reset_default_graph()

    # Placeholders
    x = tf.placeholder('float', shape=input_dim, name='x_placeholder')
    y = tf.placeholder('float', shape=[None, n_classes], name='y_placeholder')

    # Weights
    W1_conv = tf.Variable(tf.random_normal([5, 5, 1, 32]), name='hidden_layer_1')
    W2_conv = tf.Variable(tf.random_normal([5, 5, 32, 64]), name='hidden_layer_2')
    W3 = tf.Variable(tf.random_normal([3136, 3136]), name='hidden_layer_3')
    W4 = tf.Variable(tf.random_normal([3136, 3136]), name='hidden_layer_3')
    W5 = tf.Variable(tf.random_normal([3136, 3136]), name='hidden_layer_3')
    out_w = tf.Variable(tf.random_normal([3136, n_classes]), name='hidden_layer_3')

    # bias
    b1_conv = tf.Variable(tf.random_normal([32]))
    b2_conv = tf.Variable(tf.random_normal([64]))
    b3 = tf.Variable(tf.random_normal([3136]))
    b4 = tf.Variable(tf.random_normal([3136]))
    b5 = tf.Variable(tf.random_normal([3136]))
    out_b = tf.Variable(tf.random_normal([n_classes]))

    # Convolutional Layers
    conv1 = tf.nn.relu(conv2d(x, W1_conv + b1_conv))
    conv1 = maxpool2d(conv1)
    conv1_norm = tf.nn.local_response_normalization(conv1)

    conv2 = tf.nn.relu(conv2d(conv1_norm, W2_conv + b2_conv))
    conv2 = maxpool2d(conv2)

    # Fully Conncected Layer
    fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc, W3 + b3))
    fc = tf.nn.dropout(fc, keep_rate)

    # Logistic Classifiers
    l4 = tf.matmul(fc, W4) + b4
    l4_actv = tf.nn.sigmoid(l4)

    l5 = tf.matmul(l4_actv, W5) + b5
    l5_actv = tf.nn.sigmoid(l5)
    final_logits = tf.matmul(l5_actv, W5) + b5

    # Predictions
    output = tf.add(tf.matmul(final_logits, out_w), out_b, name='output')

    # Cost function
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output, name='loss'))
    tf.summary.scalar("cost", cost)

    # Validation cost
    validation_cost = cost
    tf.summary.scalar("validation_cost", validation_cost)

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, name='optimizer')

    # Correct Predictions and Accuracy
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1), name='correct_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    tf.summary.scalar("accuracy", accuracy)

    # merge all summaries into a single "operation" which we can execute in a session
    summary_op = tf.summary.merge_all()

    # Save only one model
    saver = tf.train.Saver(max_to_keep=1)

    init = tf.global_variables_initializer()

    print('Graph Built')

    with tf.Session() as sess:
        print('Session Started')
        print('--------------- \n')
        # variables need to be initialized before we can use them

        sess.run(init)

        # create log writer object
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph(), max_queue=2)

        # perform training cycles
        for epoch in range(training_epochs):

            batch_count = int(len(X_train) / batch_size)
            total_loss = 0
            for i in range(int(len(X_train) / batch_size)):
                randidx = np.random.randint(len(X_train), size=batch_size)

                batch_x = np.array(X_train)[randidx]
                batch_y = np.array(y_train)[randidx]

                # perform the operations we defined earlier on batch
                _, summary, c = sess.run([optimizer, summary_op, cost], feed_dict={x: batch_x, y: batch_y})

                total_loss += c

                # write log
                writer.add_summary(summary, epoch * batch_count + i)

            print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: X_test, y: y_test}))

            print("Validation Loss:", sess.run(validation_cost, feed_dict={x: X_test, y: y_test}))

            print('Epoch ', epoch, ' completed out of ', training_epochs, ', loss: ', total_loss)

        # Save model
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

        return total_loss, accuracy


def main():
    total_loss, accuracy = convolutional_nn(epochs=5, learning_rate=.0000000000000000001)
    remove_logs()

if __name__ == "__main__":
    main()
