import convolutional_nn as cnn
import image_bootstrapping as bs
import restore_to_test as rest_test


def main():
    x, y = bs.load_data(128, 15000)
    x, y = cnn.get_data(x, y)
    cnn.convolutional_nn(epochs=5, learning_rate=0.0001)
    rest_test.


if __name__ == "__main__":
    main()
