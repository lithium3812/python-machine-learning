import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram


"""
Demonstrate agglomerative hierarchical clustering
"""

rng = np.random.default_rng(123)

# 3 features
variables = ['X', 'Y', 'Z']
# 5 samples
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']

X = rng.random([5, 3])*10
df = pd.DataFrame(X, columns=variables, index=labels)

# distance matrix of samples
row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)

# complete linkage agglomeration
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
row_clusters_df = pd.DataFrame(
    row_clusters, 
    columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'], 
    index=[f'cluster {i+1}' for i in range(row_clusters.shape[0])]
    )
print(row_clusters_df)

# Plot dendrogram
plt.figure()
row_dendr = dendrogram(row_clusters, labels=labels)
plt.ylabel('Euclidean distance')

"""
Attach the dendrogram to a heat map
"""

fig = plt.figure(figsize=(8,8), facecolor='white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
# Rortate the dendrogram to attach to the heatmap from the left
row_dendr = dendrogram(row_clusters, orientation='left')

# reorder the original dataframe of samples according to the clusters
df_rowclust = df.iloc[row_dendr['leaves'][::-1]]

# heatmap from the reordered dataframe
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')

# Aesthetics
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))

plt.show()
