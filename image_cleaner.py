# DEPRICATED


def preprocess_images():

    import os
    import numpy as np
    import cv2
    from sklearn.preprocessing import OneHotEncoder
    import time

    def input_parser(filenames):
        # convert the label to one-hot encoding
        # step 2
        images = []
        image_size = 28
        image = cv2.imread(filenames, cv2.IMREAD_GRAYSCALE)

        # Resizing the image to our desired size and
        # preprocessing will be done exactly as done during training
        image = cv2.resize(image, (image_size, image_size))
        images.append(image)
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')


        return images



    dirPaths = ['/Users/joeklein/PycharmProjects/hotdog_not_hotdog/hotdog']

    hotdog = []


    for dirPath in dirPaths:
        for img in os.listdir(dirPath):
            current_image_path = str(dirPath) + '/' + str(img)
            decoded = input_parser(current_image_path)
            hotdog.append([decoded.ravel(), 1])




    dirPaths = ['/Users/joeklein/PycharmProjects/hotdog_not_hotdog/not-hotdog']

    start = time.time()

    print('hotdogs done in: ', time.time() - start)

    for dirPath in dirPaths:
        for img in os.listdir(dirPath):
            current_image_path = str(dirPath) + '/' + str(img)
            decoded = input_parser(current_image_path)
            hotdog.append([decoded.ravel(), 0])

    print('not hotdogs done in: ', time.time() - start)

    X_ = [np.array(x[0]) for x in hotdog]

    Y_ = [y[1] for y in hotdog]

    Y_ = np.array(Y_).reshape(-1, 1)

    encode = OneHotEncoder()
    encode.fit(Y_)
    Y_ = encode.transform(Y_).toarray()

    print(np.array(X_).shape)
    print(np.array(Y_).shape)

    np.savetxt("X.csv", X_, delimiter=",")

    np.savetxt("Y.csv", Y_, delimiter=",")

    return X_, Y_


preprocess_images()
