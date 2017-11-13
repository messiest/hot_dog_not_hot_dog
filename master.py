import check
import convolutional_nn as cnn
import image_bootstrapping as bs
import restore_to_test as restore_test


def main():
    # scrape_images.read_images_to_folder()
    model_path = "conv_model/model.ckpt"
    x, y = bs.load_data(128, 15000)
    total_loss, accuracy = cnn.run(x, y, epochs=5, learning_rate=0.000000001)
    restore_test.run_test(x, y, model_path)
    check.run(model_path)


if __name__ == "__main__":
    main()
