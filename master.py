import choices as run


def main():
<<<<<<< HEAD
    run.choices()
=======
<<<<<<< HEAD
    x, y = bs.load_data(128, 15000)
    x, y = cnn.get_data(x, y)
    cnn.convolutional_nn(epochs=5, learning_rate=0.0001)
    # rest_test.
=======
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
>>>>>>> 57281270cde3ce497c5976732e99bad20dd91ba5
>>>>>>> b3108a540dbc170f3fc5f70ab1382f5735815210


if __name__ == '__main__':
    main()
