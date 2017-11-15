import os
import time

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.model_selection import train_test_split

start = time.time()


def get_data(X_, Y_):
    print('Y Shape: ', np.array(Y_).shape)
    print('X Shape: ', X_.shape)

    data_time = time.time() - start

    if data_time >= 60:
        data_time = data_time / 60
        print('Import Data Time: {} minutes'.format(round(data_time, 0)))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n')
    else:
        print('Import Data Time: {} seconds'.format(round(data_time, 0)))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n')

    return X_, Y_


def conv2d(x, W, padding):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)


def maxpool2d(x, padding):
    #                             size of window      movement of window
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding=padding)


def maxpool2d_dif(x, padding):
    #                             size of window       movement of window
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding=padding)


def run(X_, Y_, epochs=10, learning_rate=0.01):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    print('Starting Convolutional Neural Network')
    nn_start = time.time()

    X_ = np.array(X_).reshape(-1, 128, 128, 1)

    # Get data and TTS
    X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.2, random_state=42)

    # Config
    model_path = "conv_model/model.ckpt"
    batch_size = 64
    logs_path = "logs"
    training_epochs = epochs
    initial_learning_rate = learning_rate
    n_classes = 2
    input_dim = [None, 128, 128, 1]
    keep_rate = 0.3

    # Reset Graph
    tf.reset_default_graph()

    # global step
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # learning rate policy
    decay_steps = int(len(X_train) / batch_size)
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step,
                                               decay_steps,
                                               decay_rate=0.96,
                                               staircase=True,
                                               name='exponential_decay_learning_rate')

    # Placeholders
    x = tf.placeholder('float', shape=input_dim, name='x_placeholder')
    x = tf.reshape(x, [-1, 128, 128, 1])

    y = tf.placeholder('float', shape=[None, n_classes], name='y_placeholder')
    y = tf.reshape(y, [-1, 2])

    # Weights
    W1_conv = tf.Variable(tf.random_normal([8, 8, 1, 16]), name='hidden_layer_1')
    W2_conv = tf.Variable(tf.random_normal([5, 5, 16, 32]), name='hidden_layer_2')
    W3_conv = tf.Variable(tf.random_normal([5, 5, 32, 128]), name='hidden_layer_3')
    W4 = tf.Variable(tf.random_normal([512, 512]), name='hidden_layer_4')
    W5 = tf.Variable(tf.random_normal([512, 512]), name='hidden_layer_5')
    out_w = tf.Variable(tf.random_normal([512, n_classes]), name='hidden_layer_out')

    # bias
    b1_conv = tf.Variable(tf.random_normal([16]))
    b2_conv = tf.Variable(tf.random_normal([32]))
    b3_conv = tf.Variable(tf.random_normal([128]))
    b4 = tf.Variable(tf.random_normal([512]))
    b5 = tf.Variable(tf.random_normal([512]))
    out_b = tf.Variable(tf.random_normal([n_classes]))

    # Convolutional Layers
    conv1 = tf.nn.elu(conv2d(x, W1_conv + b1_conv, 'VALID'))
    conv1 = maxpool2d_dif(conv1, 'VALID')
    conv1_norm = tf.nn.local_response_normalization(conv1)

    conv2 = tf.nn.elu(conv2d(conv1_norm, W2_conv + b2_conv, 'SAME'))
    conv2 = maxpool2d(conv2, 'SAME')

    conv3 = tf.nn.elu(conv2d(conv2, W3_conv + b3_conv, 'SAME'))
    conv3 = maxpool2d(conv3, 'SAME')
    conv3 = tf.nn.dropout(conv3, 0.2)



    # Fully Conncected Layer
    fc = tf.reshape(conv3, [-1, 512])
    fc = tf.nn.dropout(fc, 0.5)
    fc = tf.nn.elu(tf.matmul(fc, W4 + b4))

    l5 = tf.matmul(fc, W5) + b5
    l5_actv = tf.nn.softmax(l5)
    final_logits = tf.matmul(l5_actv, W5) + b5

    # Predictions
    output = tf.add(tf.matmul(final_logits, out_w), out_b, name='output')

    # Cost function
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output, name='loss'))
    tf.summary.scalar("cost", cost)

    # auc = tf.metrics.auc(labels=y, predictions=output, name='auc')
    # tf.summary.scalar("auc", auc)
    # tf.summary.histogram('Hist Auc', auc)

    # Validation cost
    validation_cost = cost
    tf.summary.scalar("validation_cost", validation_cost)
    tf.summary.histogram('Hist Val Cost', validation_cost)

    # Correct Predictions and Accuracy
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1), name='correct_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.histogram('Hist Accuracy', accuracy)

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost,
                                                               # aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
                                                               global_step=global_step,
                                                               name='optimizer')

    # merge all summaries into a single "operation" which we can execute in a session
    summary_op = tf.summary.merge_all()

    # Save only one model
    saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto(inter_op_parallelism_threads=44,
                            intra_op_parallelism_threads=44)

    init = tf.global_variables_initializer()

    graph_time = time.time() - nn_start

    if graph_time >= 60:
        graph_time = graph_time / 60
        print('Graph Built: {} minutes'.format(round(graph_time, 0)))
    else:
        print('Graph Built: {} seconds'.format(round(graph_time, 0)))

    with tf.Session(config=config) as sess:
        print('Session Started')
        print('--------------- \n')
        # variables need to be initialized before we can use them

        sess.run(init)

        # create log writer object
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph(), max_queue=2)

        # perform training cycles
        for epoch in range(training_epochs):

            epoch_start = time.time()

            batch_count = int(len(X_train) / batch_size)
            print('Number of Batches: ', batch_count)
            total_loss = 0
            for step in range(batch_count):
                randidx = np.random.randint(len(X_train), size=batch_size)

                batch_x = np.array(X_train)[randidx]
                batch_y = np.array(y_train)[randidx]

                # perform the operations we defined earlier on batch
                _, summary, c, training_step, lr = sess.run([optimizer, summary_op, cost, global_step, learning_rate],
                                                            feed_dict={x: batch_x, y: batch_y})

                total_loss += c
                print('Current Step: ', step)

                # write log
                writer.add_summary(summary, global_step=step)

            print('~~~~~~~~~~~~~~~~~~~~\n')

            print('Current Learning Rate: ', lr)

            print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: X_test, y: y_test}))

            # print("Testing AUC:", sess.run(auc, feed_dict={x: X_test, y: y_test}))

            print("Validation Loss:", sess.run(validation_cost, feed_dict={x: X_test, y: y_test}))

            print('Epoch ', epoch + 1, ' completed out of ', training_epochs, ', loss: ', total_loss)

            epoch_time = time.time() - epoch_start
            if epoch_time >= 60:
                epoch_time = epoch_time / 60
                print('Epoch Time: {} minutes'.format(round(epoch_time, 0)))
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n')
            else:
                print('Epoch Time: {} seconds'.format(round(epoch_time, 0)))
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n')

        # Save model
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

        total_time = time.time() - start
        if total_time >= 60:
            total_time = total_time / 60
            print('Epoch Time: {} minutes'.format(round(total_time, 0)))
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n')
        else:
            print('Epoch Time: {} seconds'.format(round(total_time, 0)))
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n')

        return total_loss, accuracy
