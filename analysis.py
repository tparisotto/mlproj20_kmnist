import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import preprocessing
from sklearn.decomposition import PCA
from pyclustertend import hopkins

import util

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
X_train_flat = np.reshape(X_train, (train_size, -1))
X_test_flat = np.reshape(X_test, (test_size, -1))

# print("Train data shape:")
# print(X_train.shape)
# print("Test data shape:")
# print(X_test.shape)

cov_mat = np.cov(X_train_flat.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# print('Eigenvectors \n', eig_vecs)
# print('\nEigenvalues \n', eig_vals)
# X_train_std = preprocessing.StandardScaler().fit_transform(X_train_flat)
# cov_mat_std = np.cov(X_train_std.T)
# eig_vals_std, eig_vecs_std = np.linalg.eig(cov_mat_std)
# print('\nStandardized Eigenvectors \n', eig_vecs_std)
# print('\nStandardized Eigenvalues \n', eig_vals_std)

# fig, ax = plt.subplots(1,2, figsize=(20,6))
# ax[0].bar(range(1,28*28+1), eig_vals)
# ax[0].title.set_text("Eigenvalues")
# ax[0].grid()
# ax[1].bar(range(1,28*28+1), eig_vals_std)
# ax[1].title.set_text("Standardized Eigenvalues")
# ax[1].grid()
# plt.show()
# plt.waitforbuttonpress()

# trace = np.trace(cov_mat_std)
# variance_percentage = eig_vals_std / trace
# n_features = X_train_flat.shape[1]

# sorted_percentage = [0]
# for k in range(n_features):
#   pc_idx = variance_percentage.argsort()[-(k+1):][::-1]
#   first_k_vars = sum(variance_percentage[pc_idx])
#   sorted_percentage.append(first_k_vars)
# # print("How many variables we need?\n",sorted_percentage)
# # plt.figure(figsize=(16,6))
# plt.plot(range(n_features+1),sorted_percentage)
# plt.xlabel("Number of Variables")
# plt.ylabel("Cumulative Variance Percentage")
# plt.grid()
# # plt.axhline(y=0.99, color='r', linestyle='-')
# plt.show()
# plt.waitforbuttonpress()

subsampling_factors = [0.01]
data_sizes = []
scores = []

for factor in subsampling_factors:
  dim = int(factor*len(X_train_flat))
  X = pd.DataFrame(X_train_flat)
  X = X.sample(n=dim)
  data_sizes.append(dim)
  H = hopkins(X, dim)
  scores.append(1 - H)
  
results = pd.DataFrame({"Subsampling Factors" : subsampling_factors, "Number of samples" : data_sizes, "Scores" : scores})
print("Hopkins statistic for PCA reduced data")
print(results)