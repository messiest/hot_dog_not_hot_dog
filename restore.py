import tensorflow as tf




init = tf.global_variables_initializer()
saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)

        # Restore model weights from previously saved model
        saver.restore(sess, model_path)
        print("Model restored from file: %s" % save_path)