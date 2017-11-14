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


def choices():
    while True:
        print(
            'Type "run_all" to run all scripts \nType "train" to just train model \nType "test" to test saved model \n')
        user_input = input('What would you like to run: ')
        # user_input = "'" + user_input + "'"

        if user_input == 'run_all':
            do_run_all()
        if user_input == 'train':
            do_run_train()

        if user_input == 'test':
            do_run_test()

        if user_input is not ('run_all', 'train', 'test'):
            print('Please enter either "run_all", "train", or "test" \n')
            print('Try again \n')
            continue

        else:
            break


def do_run_all():
    while True:
        try:
            epoch_param = input('How many epochs would you like to run? \n')
            epoch_param = int(epoch_param)
        except ValueError:
            print('Please enter valid integer')
            continue
        else:
            break
    while True:
        try:
            learn_param = input('What do you want your learning rate would you like to use? \n')
            learn_param = float(learn_param)
        except ValueError:
            print('Please enter valid float')
        else:
            break
    # This function will scrape images from Image.net
    scrape_images.read_images_to_folder()

    # # Removing all of the invalid or null images
    rm.remove_invalid(paths)

    # Rotating and or blurring images to create larger and balanced classes
    x, y = ia.load_data(128, 15000)

    # Run convolutional neural network
    total_loss, accuracy = cnn.run(x, y, epochs=epoch_param, learning_rate=learn_param)

    # Restore convolutional neural network and build confusion matrix
    restore_test.run_test(x, y, model_path)

    # Test the neural network on the hotdog and not_hotdog jpg
    check.run(model_path)

    end_time = time.time()
    print('Total Run Time {}'.format(round(end_time - start_time, 0)))


def do_run_train():
    while True:
        try:
            epoch_param = input('How many epochs would you like to run? \n')
            epoch_param = int(epoch_param)
        except ValueError:
            print('Please enter valid integer')
            continue
        else:
            break
    while True:
        try:
            learn_param = input('What do you want your learning rate would you like to use? \n')
            learn_param = float(learn_param)
        except ValueError:
            print('Please enter valid float')
        else:
            break

    # Rotating and or blurring images to create larger and balanced classes
    x, y = ia.load_data(128, 15000)

    # Run convolutional neural network
    total_loss, accuracy = cnn.run(x, y, epochs=epoch_param, learning_rate=learn_param)

    # Restore convolutional neural network and build confusion matrix
    restore_test.run_test(x, y, model_path)

    # Test the neural network on the hotdog and not_hotdog jpg
    check.run(model_path)

    end_time = time.time()
    print('Total Run Time {}'.format(round(end_time - start_time, 0)))


def do_run_test():
    # Rotating and or blurring images to create larger and balanced classes
    x, y = ia.load_data(128, 15000)

    # Restore convolutional neural network and build confusion matrix
    restore_test.run_test(x, y, model_path)

    # Test the neural network on the hotdog and not_hotdog jpg
    check.run(model_path)

    end_time = time.time()
    print('Total Run Time {}'.format(round(end_time - start_time, 0)))
