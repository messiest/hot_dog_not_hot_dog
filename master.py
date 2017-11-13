import check
import convolutional_nn as cnn
import get_raw_images as scrape_images
import image_bootstrapping as bs
import remove_null_images as rm
import restore_to_test as restore_test

paths = ['./not-hotdog',
         './hotdog']

model_path = "conv_model/model.ckpt"

def main():
    scrape_images.read_images_to_folder()
    rm.remove_invalid(paths)
    x, y = bs.load_data(128, 15000)
    total_loss, accuracy = cnn.run(x, y, epochs=10, learning_rate=0.001)
    restore_test.run_test(x, y, model_path)
    check.run(model_path)


if __name__ == "__main__":
    main()
