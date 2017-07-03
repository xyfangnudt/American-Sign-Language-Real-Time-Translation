
import cv2
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from scipy.spatial import KDTree
import numpy as np


def to_gray(color_img):
    """

    Converts an image to grayscale

    :param color_img: Image that was read in with cv2.imread
    :type color_img: numpy.array
    :return: Image converted to grayscale
    :rtype: numpy.array

    """

    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray


def get_sift_features_from_file(sift, file):
    """

    :param file: File to get SIFT features from.
    :type file: str
    :return: Keypoints, Descriptors
    :rtype: numpy.array, numpy.array

    """

    image = cv2.imread(file)
    gray = to_gray(image)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def compute_bow(dictionary, features):
    """

    Computes a bag of words from a list of image features

    :param dictionary:
    :type dictionary:
    :param features:
    :type features:
    :return:
    :rtype:

    """

    kdtree = KDTree(dictionary)
    bins = dictionary.shape[0]
    hists = []
    for i, feature in enumerate(features):
        print(i)
        dis, indices = kdtree.query(feature)
        hist, _ = np.histogram(indices, bins=bins)
        hist = hist / len(indices) #normalizes
        hists.append(hist)
    return np.array(hists)


def main():

    image_list = glob.glob('dataset/**/*.jpeg', recursive=True)
    labels = [os.path.basename(os.path.dirname(file)) for file in image_list]
    x_train, x_test, y_train, y_test = train_test_split(image_list, labels, train_size=.8)

    dictionarySize = 500
    BOW = cv2.BOWKMeansTrainer(dictionarySize)
    sift = cv2.xfeatures2d.SIFT_create()

    train_features = []
    for i, x in enumerate(x_train):
        print(i)
        kp, des = get_sift_features_from_file(sift, x)
        train_features.append(des)
        BOW.add(des)

    test_features = []
    for j, x in enumerate(x_test):
        print(j)
        kp, des = get_sift_features_from_file(sift, x)
        test_features.append(des)

    dictionary = BOW.cluster()
    np.save('dictionary', dictionary)

    train_bow = compute_bow(dictionary, train_features)
    test_bow = compute_bow(dictionary, test_features)

    model = LinearSVC()
    model.fit(train_bow, y_train)
    print(model.score(test_bow, y_test))

    joblib.dump(model, 'model.pkl')

if __name__ == '__main__':
    main()



