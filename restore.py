import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

model_path = "saved_model/model.ckpt"

X_ = pd.read_csv('X.csv').values
Y_ = pd.read_csv('Y.csv').values

X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.2, random_state=42)

init = tf.global_variables_initializer()

predictions =[]

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

    cp, a , o= sess.run([correct_prediction, accuracy, tf.argmax(output, 1)], feed_dict={x: X_test, y: y_test})

    predictions.append(o)




import itertools
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, roc_auc_score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

actuals = [y[1] for y in y_test]
actuals = np.array(actuals)

predictions = np.array(predictions).reshape(1090,)


print(predictions.shape)
print(actuals.shape)

y_pred = predictions
y_test = actuals
class_names = ['Not Hotdog', 'Hotdog']

# Set font and graph size
sns.set(font_scale=1.5)
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
def print_cm(cm, labels=class_names, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print
    "    " + empty_cell,
    for label in labels:
        print
        "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print
        "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print
            cell,
        print
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print
        '\n'
        print("Normalized confusion matrix")
    else:
        print
        '\n'
        print('Confusion matrix, without normalization')
    print_cm(cm, class_names)
    text_labels = [['True Negative', 'False Positive'],
                   ['False Negative', 'True Positive']]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i - 0.1, format(cm[i, j], fmt),
                 verticalalignment='bottom',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.text(j, i + 0.1, text_labels[i][j],
                 verticalalignment='top',
                 horizontalalignment="center",
                 fontsize=12,
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Print accuracy and precision
print('Accuracy: ', accuracy_score(y_test, y_pred, normalize=True))
print('Precision: ', precision_score(y_test, y_pred, average='macro'))
print('Roc-Auc: ', roc_auc_score(y_test, y_pred))
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()

