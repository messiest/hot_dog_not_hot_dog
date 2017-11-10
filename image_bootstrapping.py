import numpy as np
import cv2
import glob
from sklearn.preprocessing import OneHotEncoder

def rotateImage(img, angle):
    (rows, cols) = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def loadRotateBlurImg(path, imgSize):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    angle = np.random.randint(0, 360)
    img = rotateImage(img, angle)
    img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, imgSize)
    return img

def loadRotate(path, imgSize):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    angle = np.random.randint(0, 360)
    img = rotateImage(img, angle)
    img = cv2.resize(img, imgSize)
    return img

def loadBlurImg(path, imgSize):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, imgSize)
    return img

def loadImg(path, imgSize):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, imgSize)
    return img


def loadImgClass(classPath, classLable, classSize, imgSize):
    x = []
    y = []

    #
    for path in classPath:
        img = np.array(loadImg(path, imgSize))
        x.append(img)
        y.append(classLable)

    while len(x) < classSize/4:
        randIdx = np.random.randint(0, len(classPath))
        img = np.array(loadRotateBlurImg(classPath[randIdx], imgSize))
        x.append(img)
        y.append(classLable)

    while len(x) < classSize/2:
        randIdx = np.random.randint(0, len(classPath))
        img = np.array(loadBlurImg(classPath[randIdx], imgSize))
        x.append(img)
        y.append(classLable)

    while len(x) < classSize * (3/4):
        randIdx = np.random.randint(0, len(classPath))
        img = np.array(loadRotate(classPath[randIdx], imgSize))
        x.append(img)
        y.append(classLable)

    while len(x) < classSize:
        randIdx = np.random.randint(0, len(classPath))
        img = np.array(loadImg(classPath[randIdx], imgSize))
        x.append(img)
        y.append(classLable)

    return x, y


def loadData(img_size, classSize):
    hotdogs = glob.glob('../hotdog/**/*.jpg', recursive=True)

    notHotdogs = glob.glob('../not-hotdog/**/*.jpg', recursive=True)

    imgSize = (img_size, img_size)


    xHotdog, yHotdog = loadImgClass(hotdogs, 0, classSize, imgSize)
    print('Hotdogs Bootstrapped')

    xNotHotdog, yNotHotdog = loadImgClass(notHotdogs, 1, classSize, imgSize)
    print('Not Hotdogs Bootstrapped')

    print("There are", len(xHotdog), "hotdog images")
    print("There are", len(xNotHotdog), "not hotdog images")

    X = np.array(xHotdog + xNotHotdog)
    y = np.array(yHotdog + yNotHotdog)

    y = np.array(y).reshape(-1, 1)

    encode = OneHotEncoder()
    encode.fit(y)
    y = encode.transform(y).toarray()

    print(np.array(X).shape)
    print(np.array(y).shape)

    import pandas as pd

    X = pd.DataFrame.from_records(X)
    X.to_csv("X_notFlat.csv")

    # np.savetxt("X_notFlat.csv", X, delimiter=",")

    np.savetxt("Y_notFlat.csv", y, delimiter=",")

    return X, y

print('Starting Bootstrap')
X_, Y_ = loadData(128, 15000)