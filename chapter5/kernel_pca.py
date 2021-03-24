"""
Kernel PCA
"""
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np


def rbf_KPCA(X, gamma, n_components):
    """
    Kernel PCA method with the Radial Basis Function
    """
    # 1. Compute RBS kernel matrix from the dataset
    # calculate pairwise squared Euclidean distances in the dataset
    sq_dists = pdist(X, 'sqeuclidean')
    # pdist retuens 1D array of the upper triangle matrix of the distance matrix. squareform transforms it to square matrix form
    mat_sq_dists = squareform(sq_dists)
    # compute the symmetric kernel matrix
    K = exp(-gamma*mat_sq_dists)
    # 2. Center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N, N))/N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    # 3.  collect the top-k eigenvectors of the centered kernel matrix
    # eigenpairs of the centered kernel matrix
    eigen_vals, eigen_vecs = eigh(K)
    # scipy.linalg.eigh returns in ascending order
    eigen_vals, eigen_vecs = eigen_vals[::-1], eigen_vecs[:, ::-1]
    # collect the top-k eigenvectors as the output
    X_pc = np.column_stack([eigen_vecs[:, i] for i in range(n_components)])
    # collect the corresponding eigenvalues
    lambdas = [eigen_vals[i] for i in range(n_components)]
    return X_pc, lambdas