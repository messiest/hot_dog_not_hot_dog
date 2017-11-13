import time

import check
import convolutional_nn as cnn
import get_raw_images as scrape_images
import image_augmentation as ia
import remove_null_images as rm
import restore_to_test as restore_test

start_time = time.time()
paths = ['not_hotdog',
         'hotdog']

model_path = "conv_model/model.ckpt"


def main():
    # This function will scrape images from Image.net
    scrape_images.read_images_to_folder()

    # Removing all of the invalid or null images
    rm.remove_invalid(paths)

    # Rotating and or blurring images to create larger and balanced classes
    x, y = ia.load_data(128, 15000)

    # Run convolutional neural network
    total_loss, accuracy = cnn.run(x, y, epochs=10, learning_rate=0.001)

    # Restore convolutional neural network and build confusion matrix
    restore_test.run_test(x, y, model_path)

    # Test the neural network on the hotdog and not_hotdog jpg
    check.run(model_path)

    end_time = time.time()
    print('Total Run Time {}'.format(round(end_time - start_time, 0)))


if __name__ == "__main__":
    main()
