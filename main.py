import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
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

# Flatten
X_train_flat = np.reshape(X_train, (train_size, -1))
X_test_flat = np.reshape(X_test, (test_size, -1))

# PCA
# pcn = 550
# pca = PCA(n_components=pcn)
# print("[INFO] Processing PCA with {} components...".format(pcn))
# X_train_pca = pca.fit_transform(X_train_flat)
# X_test_pca = pca.fit_transform(X_test_flat)
# print("[INFO] PCA completed.")

# Subsampling
# factor = 0.1
# dim = int(factor*len(X_train_flat))
# samples = np.random.randint(0,dim, size=dim)
# X = X_train_flat[samples]
# y = y_train[samples]


### Classifiers ###

# ## Decision Tree Classifier - [Score: 0.6475]
# clf = DecisionTreeClassifier(criterion='entropy')
# print("[INFO] Training Decision Tree Classifier")
# print("[INFO] Training...")
# clf.fit(X, y)
# print("[INFO] Score: {}".format(clf.score(X_test_flat, y_test)))

# ## Random Forest Classifier - [Score: 0.857]
# for est in [5, 10, 20, 50, 70, 100, 150, 200]:
#     clf = RandomForestClassifier(n_estimators=est, criterion='entropy',)
#     print("[INFO] Training Random Forest Classifier with {} estimators".format(est))
#     print("[INFO] Training...")
#     clf.fit(X_train_flat, y_train)
#     print("[INFO] Score: {}".format(clf.score(X_test_flat, y_test)))

## K-Nearest Neighboors Classifier - [0.8213 with full dataset and k = 1 subsampling = 0.1]
# scores = []
# for factor in [0.01]:
#     for k in range(1,17):
#         dim = int(factor*len(X_train_flat))
#         samples = np.random.randint(0,dim, size=dim)
#         X = X_train_flat[samples]
#         y = y_train[samples]
#         clf = KNeighborsClassifier(n_neighbors=k,)
#         print("[INFO] Training K-Nearest Neighboors Classifier (k = {}), Sub-Sampling factor: {}".format(k,factor))
#         clf.fit(X, y)
#         score = clf.score(X_test_flat, y_test)
#         scores.append(score)
#         print("[INFO] Score: {}".format(score))

# plt.plot(scores)
# plt.grid()
# plt.xlabel("K Neighbours")
# plt.ylabel("Accuracy")
# plt.show()


# ## Naive Bayes Classifier - [Score: 0.4656]
# clf = GaussianNB()
# print("[INFO] Training Naive Bayes Classifier")
# print("[INFO] Training...")
# clf.fit(X_train_flat, y_train)
# print("[INFO] Score: {}".format(clf.score(X_test_flat, y_test)))

## Standard Vector Classifier - [Too Long]
# factor = 0.001
# dim = int(factor*len(X_train_flat))
# samples = np.random.randint(0,dim, size=dim)
# X = X_train_flat[samples]
# y = y_train[samples]
# clf = LinearSVC()
# print("[INFO] Training Standard Vector Classifier")
# print("[INFO] Training...")
# clf.fit(X_train_flat, y_train)
# print("[INFO] Score: {}".format(clf.score(X_test_flat, y_test)))

### Multi Layer Perceptron (neural net) - [Score: 0.8719] ###
# for hl in [10, 25, 50, 75, 100, 125, 150, 175, 200]:
#     clf = MLPClassifier(hidden_layer_sizes=150)
#     print("[INFO] Training MLP Classifier with layer size: {}".format(hl))
#     clf.fit(X_train_flat, y_train)
#     print("[INFO] Score: {}".format(clf.score(X_test_flat, y_test)))


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
# history = model.fit(X_train, y_train, epochs=20, callbacks=[tensorboard_callback], validation_data=(X_test, y_test))


### Convolutional Neural Network - [accuracy: 0.9953 - val_accuracy: 0.9570] ###
X_train = X_train.reshape(train_size, 28, 28, 1)
X_test = X_test.reshape(test_size, 28, 28, 1)
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(28, (3,3)),
  tf.keras.layers.MaxPool2D((2,2)),
  tf.keras.layers.Conv2D(16, (2,2)),
  tf.keras.layers.MaxPool2D((2,2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(550, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(225, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.build((1, 28, 28, 1))
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs/', histogram_freq=1)
# history = model.fit(X_train, y_train, epochs=20, callbacks=[tensorboard_callback], validation_data=(X_test, y_test))
