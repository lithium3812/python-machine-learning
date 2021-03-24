"""
Implement Linear Discriminant Analysis from scratch without sklearn
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('wine_data.csv')
X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# set digit precision shown in print
np.set_printoptions(precision=4)
# mean vectors
mean_vecs = []
for label in np.unique(y_train):
    mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
    print(f"mean vector for class {label}: {mean_vecs[label-1]}")

# label distribution
print("class label distribution:", np.bincount(y_train)[1:])
# number of features
d = X.shape[1]
# within-class scatter matrix
S_W = np.zeros((d,d))
for label, mv in zip(np.unique(y_train), mean_vecs):
    # scatter matrix for each class (covariance matrix)
    class_scatter = np.zeros((d,d))
    num_class_samples = X_train_std[y_train==label].shape[0]
    print(f"number of samples with class {label}: {num_class_samples}")
    for row in X_train_std[y_train==label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row-mv).dot((row-mv).T)/num_class_samples
    S_W += class_scatter
print("within-class scatter matrix:\n", S_W)

# overall mean (0 since they are standardized)
mean_overall = np.mean(X_train_std, axis=0)
print("overall mean:", mean_overall)
# between-class scatter matrix
S_B = np.zeros((d,d))
for label, mv in zip(np.unique(y_train), mean_vecs):
    mv, mean_overall = mv.reshape(d, 1), mean_overall.reshape(d, 1)
    num_class_samples = X_train_std[y_train==label].shape[0]
    S_B += num_class_samples*(mv - mean_overall).dot((mv - mean_overall).T)
print("between-class scatter matrix:", S_B)

# find linear discriminant
# comupute eigenvalues and eigenvectors of S_W^(-1)S_B
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
print("eigenvalues in descending order:\n")
for pair in eigen_pairs:
    print(pair[0])

# create the transformation matrix
W = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
print("transformation matrix:\n", W)

# plot effective 'discrinability' together with their cumulative sum
tot = sum(eigen_vals.real)
lin_discr = [val/tot for val in sorted(eigen_vals, reverse=True)]
cum_discr = np.cumsum(lin_discr)
print("linear discriminant:\n", lin_discr)
print("cumulant discriminant:\n", cum_discr)
plt.figure()
plt.bar(range(len(eigen_vals)), lin_discr, alpha=0.5, align='center', label='Indivisual discrinability')
plt.step(range(len(eigen_vals)), cum_discr, where='mid', label='Cumulative discrinability')
plt.xlabel('Linear Discriminants')
plt.ylabel('Discrinability ratio')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()

X_train_lda = X_train_std.dot(W)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
plt.figure()
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train==l, 0], X_train_lda[y_train==l, 1], c=c, label=l, marker=m)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()