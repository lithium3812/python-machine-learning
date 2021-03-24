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

cov_mat = np.cov(X_train_std.T)
print("Covariance matrix: \n", cov_mat)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print("Eigenvalues: \n", eigen_vals)

# Variance Explained Ratios
tot = sum(eigen_vals)
var_exp = [val/tot for val in eigen_vals]
# Cumulative Cariance Explained
cum_var_exp = np.cumsum(var_exp)

plt.figure(figsize=(8,6))
plt.bar(range(X.shape[1]), var_exp, alpha=0.5, align='center', label='Indivisual explained variance')
plt.step(range(X.shape[1]), cum_var_exp, where='mid', label='Cumulative Explained Variance')
plt.xlabel('Principal component index')
plt.ylabel('Variance Explained Ratios')
plt.legend(loc='best')
plt.tight_layout()

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
# Sort the tuples from high to low
eigen_pairs.sort(key=lambda x: x[0], reverse=True)
# Make projection matrix from eigenvectors
W = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print("Projection matrix:\n", W)
# Transform dataset to 2 principal components
X_train_pca = X_train_std.dot(W)
print("Shape of transformed dataset:", X_train_pca.shape)
# Plot samples onto the space of 2 principal components
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
plt.figure(figsize=(8,6))
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(x=X_train_pca[y_train==l, 0], y=X_train_pca[y_train==l, 1], c=c, label=l, marker=m)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()