import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


"""
Compare performances of 3 clustering algorithms: k-means, agglomerative, DBSCAN with halfmoon dataset
"""

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))

# K-means clustering
km = KMeans(n_clusters=2, random_state=0)
y_km = km.fit_predict(X)
ax1.scatter(X[y_km==0, 0], X[y_km==0, 1], c='lightblue', edgecolor='black', marker='o', s=40, label='cluster 1')
ax1.scatter(X[y_km==1, 0], X[y_km==1, 1], c='red', edgecolor='black', marker='s', s=40, label='cluster 2')
ax1.set_title('K-means clustering')

# Agglomeratice clustering
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
y_ac = ac.fit_predict(X)
ax2.scatter(X[y_ac==0, 0], X[y_ac==0, 1], c='lightblue', edgecolor='black', marker='o', s=40, label='cluster 1')
ax2.scatter(X[y_ac==1, 0], X[y_ac==1, 1], c='red', edgecolor='black', marker='s', s=40, label='cluster 2')
ax2.set_title('Agglomerative clustering')

# DBSCAN (density-based clustering)
db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
y_db = db.fit_predict(X)
ax3.scatter(X[y_db==0, 0], X[y_db==0, 1], c='lightblue', edgecolor='black', marker='o', s=40, label='cluster 1')
ax3.scatter(X[y_db==1, 0], X[y_db==1, 1], c='red', edgecolor='black', marker='s', s=40, label='cluster 2')
ax3.set_title('DBSCAN')

plt.legend()
plt.tight_layout()
plt.show()