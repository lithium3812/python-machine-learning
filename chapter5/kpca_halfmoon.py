from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from kernel_pca import rbf_KPCA


# half-moon dataset of sklearn
X, y = make_moons(n_samples=100, random_state=123)
# sklearn PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
# Kernel PCA with RBF
X_kpca, lambdas = rbf_KPCA(X, gamma=15, n_components=2)

# original data
plt.figure('half-moon dataset', figsize=(8,6))
plt.scatter(X[y==0, 0], X[y==0, 1], color='r', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='b', marker='o', alpha=0.5)
plt.tight_layout()

# ordinary PCA
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='r', marker='^', alpha=0.5)
ax[0].scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='b', marker='o', alpha=0.5)
ax[1].scatter(X_pca[y==0, 0], np.zeros((50,1))+0.02, color='r', marker='^', alpha=0.5)
ax[1].scatter(X_pca[y==1, 0], np.zeros((50,1))-0.02, color='b', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.tight_layout()

# RBF Kernel PCA
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='r', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='b', marker='o', alpha=0.5)
ax[1].scatter(X_kpca[y==0, 0], np.zeros((50,1))+0.02, color='r', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((50,1))-0.02, color='b', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.tight_layout()

# demonstrate KPCA projection of a new data point
# use the 26th point in the dataset as an example
x_new = X[25]
print("Original data point:", x_new)
x_proj = X_kpca[25]
print("Projection in the initial KPCA:", x_proj)
# projection of a new data point
def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
    K = np.exp(-gamma*pair_dist)
    return K.dot(alphas/lambdas)

x_rproj = project_x(x_new, X, gamma=15, alphas=X_kpca, lambdas=lambdas)
print("Projection as a new data point:", x_rproj)

plt.figure('KPCA projection of a new data point', figsize=(8,6))
plt.scatter(X_kpca[y==0, 0], np.zeros((50,1)), color='r', marker='^', alpha=0.5)
plt.scatter(X_kpca[y==1, 0], np.zeros((50,1)), color='b', marker='o', alpha=0.5)
plt.scatter(x_proj[0], 0, color='b', label='original projection of point X[25]', marker='^', s=100)
plt.scatter(x_rproj[0], 0, color='g', label='projection of point X[25] as a new data point', marker='x', s=500)
plt.yticks([], [])
plt.xlabel('PC1')
plt.legend(scatterpoints=1)
plt.tight_layout()
plt.show()