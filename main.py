import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras import models, layers

import util


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

lab2char = {0 : "\u304A",
            1 : "\u304D",
            2 : "\u3059",
            3 : "\u3064",
            4 : "\u306A",
            5 : "\u306F",
            6 : "\u307E",
            7 : "\u3084",
            8 : "\u308C",
            9 : "\u3092"}

X_train, X_test, y_train, y_test, train_size, test_size, img_size = util.load_and_split()
X_train, X_test = X_train / 255.0, X_test / 255.0


### Preprocessing ###
X_train_flat = np.reshape(X_train, (train_size, -1))
X_test_flat = np.reshape(X_test, (test_size, -1))
pcn = 550
pca = PCA(n_components=pcn)
print("[INFO] Processing PCA with {} components...".format(pcn))
X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.fit_transform(X_test_flat)
print("[INFO] PCA completed.")


### Classifiers ###

# ## Decision Tree Classifier - [Score: 0.6475]
# clf = DecisionTreeClassifier(criterion='entropy')
# print("[INFO] Training Decision Tree Classifier")
# print("[INFO] Training...")
# clf.fit(X_train_flat, y_train)
# print("[INFO] Score: {}".format(clf.score(X_test_flat, y_test)))

# ## Random Forest Classifier - [Score: 0.857]
# clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
# print("[INFO] Training Random Forest Classifier")
# print("[INFO] Training...")
# clf.fit(X_train_flat, y_train)
# print("[INFO] Score: {}".format(clf.score(X_test_flat, y_test)))

## K-Nearest Neighboors Classifier - [Too Long]
# k = 3
# clf = KNeighborsClassifier(n_neighbors=k)
# print("[INFO] Training K-Nearest Neighboors Classifier (k = {})".format(k))
# print("[INFO] Training...")
# clf.fit(X_train_pca, y_train)
# print("[INFO] Score: {}".format(clf.score(X_test_pca, y_test)))

# ## Naive Bayes Classifier - [Score: 0.4656]
# clf = GaussianNB()
# print("[INFO] Training Naive Bayes Classifier")
# print("[INFO] Training...")
# clf.fit(X_train_flat, y_train)
# print("[INFO] Score: {}".format(clf.score(X_test_flat, y_test)))

## Standard Vector Classifier - [Too Long]
# clf = SVC()
# print("[INFO] Training Standard Vector Classifier")
# print("[INFO] Training...")
# clf.fit(X_train_flat, y_train)
# print("[INFO] Score: {}".format(clf.score(X_test_flat, y_test)))




### Fully Connected Neural Network (h1=128, h2=10, epochs=5) - Score: 0.8902 ###
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# history = model.fit(X_train, y_train, epochs=5)
# model.evaluate(X_test, y_test)



### Fully Connected Neural Network (h1=128, h2=64, h3=10, epochs=20) - Score: 0.8984 ###
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(64, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# print("[INFO] Training...")
# history = model.fit(X_train, y_train, epochs=20)
# print("[INFO] Evaluation: ")
# model.evaluate(X_test, y_test)

### Fully Connected Neural Network - [accuracy: 0.9823 - val_accuracy: 0.9181] ###
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(550, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(225, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(64, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs/', histogram_freq=1)
# history = model.fit(X_train, y_train, epochs=10, callbacks=[tensorboard_callback], validation_data=(X_test, y_test))


### Convolutional Neural Network ###
# X_train = X_train.reshape(train_size, 28, 28, 1)
# X_test = X_test.reshape(test_size, 28, 28, 1)
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Conv2D(28, (3,3)),
#   tf.keras.layers.MaxPool2D((2,2)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(64, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# print("[INFO] Training...")
# history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
