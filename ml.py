
import cv2
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from scipy.spatial import KDTree
import numpy as np


def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

image_list = glob.glob('dataset/**/*.jpeg', recursive=True)
labels = [os.path.basename(os.path.dirname(file)) for file in image_list]
x_train, x_test, y_train, y_test = train_test_split(image_list, labels, train_size=.8)

dictionarySize = 200
BOW = cv2.BOWKMeansTrainer(dictionarySize)
sift = cv2.xfeatures2d.SIFT_create()

def get_sift_features_from_file(file):
    image = cv2.imread(file)
    gray = to_gray(image)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des

train_features = []
for i, x in enumerate(x_train):
    print(i)
    kp, des = get_sift_features_from_file(x)
    train_features.append(des)
    BOW.add(des)

test_features = []
for j, x in enumerate(x_test):
    print(j)
    kp, des = get_sift_features_from_file(x)
    test_features.append(des)

dictionary = BOW.cluster()
np.save('dictionary', dictionary)


def compute_bow(dictionary, features):
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

train_bow = compute_bow(dictionary, train_features)
test_bow = compute_bow(dictionary, test_features)

model = LinearSVC()
model.fit(train_bow, y_train)
model.score(test_bow, y_test)

joblib.dump(model, 'model.pkl')



