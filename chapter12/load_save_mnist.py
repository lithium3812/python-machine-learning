from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


# Loading the dataset might take very long
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
y = y.values.astype(int)
X = ((X.values/255.) - .5)*2   # Standardize pixels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=123, stratify=y)

# Save the pixel data and labels compressed
np.savez_compressed('mnist_scaled.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# Load the compressed file back
mnist = np.load('mnist_scaled.npz')
X_train = mnist['X_train']
y_train = mnist['y_train']
X_test = mnist['X_test']
y_test = mnist['y_test']

# Just check if images are properly downloaded
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

