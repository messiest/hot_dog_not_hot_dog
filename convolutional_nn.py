import tensorflow as tf
import numpy as np
import os
import time
# import image_bootstrapping as ib

start = time.time()

def remove_logs():
    logs_length = len(os.listdir('logs'))
    logs = os.listdir('logs')
    if  logs_length >= 1:
        for i in range(logs_length - 1):
            os.remove(logs[i])


def get_data():
    import pandas as pd
    from sklearn.model_selection import train_test_split

    Y_ = pd.read_csv('Y_notFlat.csv').values
    print('Y Done')
    print(np.array(Y_).shape)
    X_ = pd.read_csv('X_notFlat.csv').values
    print('X Done')

    # X_, Y_ = ib.loadData(128, 15000)
    print(X_.shape)


    X_ = np.array(X_).reshape(-1, 128, 128, 1)

    print(X_.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.2, random_state=42)

    data_time = time.time() - start

    print('Images Read: {} seconds'.format(round(data_time, 0)))

    return X_train, X_test, y_train, y_test


def conv2d(x, W, padding):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)



def maxpool2d(x, padding):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)

def maxpool2d_dif(x, padding):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding=padding)


def convolutional_nn(epochs=10, learning_rate=0.01):

    nn_start = time.time()

    # Get data and TTS
    X_train, X_test, y_train, y_test = get_data()

    # Config
    model_path = "conv_model/model.ckpt"
    batch_size = 1
    logs_path = "logs"
    training_epochs = epochs
    learning_rate = learning_rate
    n_classes = 128
    input_dim = [None, 128, 128, 1]
    keep_rate = 0.3

    # Reset Graph
    tf.reset_default_graph()

    # Placeholders
    x = tf.placeholder('float', shape=input_dim, name='x_placeholder')
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



    conv2 = tf.nn.elu(conv2d(conv1_norm, W2_conv + b2_conv,'SAME'))
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
    print(output)
    print(y)

    # Cost function
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output, name='loss'))
    print(y)
    tf.summary.scalar("cost", cost)

    # Validation cost
    validation_cost = cost
    tf.summary.scalar("validation_cost", validation_cost)


    # Correct Predictions and Accuracy
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1), name='correct_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    tf.summary.scalar("accuracy", accuracy)

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, name='optimizer')

    # merge all summaries into a single "operation" which we can execute in a session
    summary_op = tf.summary.merge_all()

    # Save only one model
    saver = tf.train.Saver(max_to_keep=1)

    init = tf.global_variables_initializer()

    graph_time = time.time() - nn_start

    print('Graph Built: {} seconds'.format(round(graph_time, 0)))


    with tf.Session() as sess:
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
            print(batch_count)
            total_loss = 0
            for i in range(batch_count):
                randidx = np.random.randint(len(X_train), size=batch_size)

                batch_x = np.array(X_train)[randidx]
                batch_y = np.array(y_train)[randidx]
                print(batch_y.shape)
                print(batch_x.shape)

                # perform the operations we defined earlier on batch
                _, summary, c = sess.run([optimizer, summary_op, cost], feed_dict={x: batch_x, y: batch_y})

                total_loss += c

                # write log
                writer.add_summary(summary, epoch * batch_count + i)

            print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: X_test, y: y_test}))


            print("Validation Loss:", sess.run(validation_cost, feed_dict={x: X_test, y: y_test}))

            print('Epoch ', epoch + 1, ' completed out of ', training_epochs, ', loss: ', total_loss)
            epoch_time = time.time() - epoch_start

            print('Epoch time: {} seconds'.format(round(epoch_time, 0)))
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n')

        # Save model
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

        total_time = time.time() - start

        print('Total time: {} seconds'.format(round(total_time, 0)))

        return total_loss, accuracy


def main():
    total_loss, accuracy = convolutional_nn()
    # remove_logs()

if __name__ == "__main__":
    main()
