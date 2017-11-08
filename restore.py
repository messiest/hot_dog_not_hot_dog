import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

model_path = "saved_model/model.ckpt"

X_ = pd.read_csv('X.csv').values
Y_ = pd.read_csv('Y.csv').values

X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.2, random_state=42)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)
    # access a variable from the saved Graph, and so on:

    graph = tf.get_default_graph()
    cost = graph.get_tensor_by_name("loss:0")
    accuracy = graph.get_tensor_by_name('accuracy:0')
    x = graph.get_tensor_by_name('x_placeholder:0')
    y = graph.get_tensor_by_name('y_placeholder:0')
    output = graph.get_tensor_by_name('output:0')
    correct_prediction = graph.get_tensor_by_name('correct_prediction:0')

    cp, a = sess.run([correct_prediction, accuracy], feed_dict={x: X_test, y: y_test})
    print(cp)
    print(a)