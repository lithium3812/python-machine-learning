from sklearn.datasets import make_circles
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from kernel_pca import rbf_KPCA


X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_kpca, lambdas = rbf_KPCA(X, gamma=15, n_components=2)

plt.figure('half-moon dataset', figsize=(8,6))
plt.scatter(X[y==0, 0], X[y==0, 1], color='r', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='b', marker='o', alpha=0.5)
plt.tight_layout()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='r', marker='^', alpha=0.5)
ax[0].scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='b', marker='o', alpha=0.5)
ax[1].scatter(X_pca[y==0, 0], np.zeros((500,1))+0.02, color='r', marker='^', alpha=0.5)
ax[1].scatter(X_pca[y==1, 0], np.zeros((500,1))-0.02, color='b', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.tight_layout()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='r', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='b', marker='o', alpha=0.5)
ax[1].scatter(X_kpca[y==0, 0], np.zeros((500,1))+0.02, color='r', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((500,1))-0.02, color='b', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.tight_layout()

plt.show()