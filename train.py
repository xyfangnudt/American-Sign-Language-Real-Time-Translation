import os
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.constraints import maxnorm
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from skimage.io import imread
from skimage.color import rgb2gray

letter_map = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'i': 8,
    'j': 9,
    'k': 10,
    'l': 11,
    'm': 12,
    'n': 13,
    'o': 14,
    'p': 15,
    'q': 16,
    'r': 17,
    's': 18,
    't': 19,
    'u': 20,
    'v': 21,
    'w': 22,
    'x': 23,
    'y': 24,
    'z': 25,
    '0': 26,
    '1': 27,
    '2': 28,
    '3': 29,
    '4': 30,
    '5': 31,
    '6': 32,
    '7': 33,
    '8': 34,
    '9': 35,
}

n_letters = len(letter_map)

def build_dataset(dataset_path):
    """

    This function will generate the file paths and corresponding labels for this dataset.
    :param dataset_path: Dataset path.
    :type dataset_path: str
    :return: files and labels
    :rtype: list, list
    """
    x, y = [], []
    for root, subFolders, files in os.walk(dataset_path):
        for file in files:
            x.append(os.path.join(root, file))
            label = letter_map[os.path.basename(os.path.normpath(root))]
            y.append(label)

    return x, y

files, y = build_dataset('dataset')

x = []
for i, file in enumerate(files):
    print(i, 'out of', len(files))
    image = imread(file)
    image = rgb2gray(image)
    x.append(image)

x = np.array(x)
y = np.array(y)

width, height = 400, 400

x = np.reshape(x, (-1, width, height, 1))

x = x.astype('float32')
x /= 255

y = np_utils.to_categorical(y, n_letters)

x_train, x_test, y_train, y_test = train_test_split(x, y)

model = Sequential()

model.add(Convolution2D(32, (3, 3), input_shape=(width, height, 1), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(n_letters, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=32, epochs=100, verbose=1)

model.save('asl.h5')
