import numpy as np
import matplotlib.pyplot as plt
from numpy.random import Generator, PCG64
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from decision_boundary import plot_regions, plot_regions3D


rg = Generator(PCG64())
X_xor = rg.standard_normal((200, 2))
y_xor = np.where(np.logical_xor(X_xor[:, 0]>0, X_xor[:, 1]>0), 1, 0)
plt.figure('XOR gate samples', figsize=(8,6))
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], c='r', marker='s', label='0')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.xlabel('first element')
plt.ylabel('second element')
plt.legend(loc='best')
plt.tight_layout()

X_train, X_test, y_train, y_test = train_test_split(X_xor, y_xor, test_size=0.33, random_state=42)
X_combined = np.append(X_train, X_test, axis=0)
y_combined = np.append(y_train, y_test)
kernelSVM = SVC(kernel='rbf', random_state=1, gamma=0.1, C=10.0)
kernelSVM.fit(X_train, y_train)
y_pred_kernel = kernelSVM.predict(X_test)
score_kernel = accuracy_score(y_test, y_pred_kernel)
print("accuracy_kernel:", score_kernel)
plt.figure('kernel SVM', figsize=(8,6))
plot_regions(X_combined, y_combined, kernelSVM, test_idx=range(X_train.shape[0], X_xor.shape[0]))
plt.legend(loc='best')
plt.tight_layout()

X_xor = np.append(X_xor, np.multiply(X_xor[:, 0], X_xor[:, 1]).reshape(-1,1), axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_xor, y_xor, test_size=0.33, random_state=42)
X_combined = np.append(X_train, X_test, axis=0)
y_combined = np.append(y_train, y_test)
linearSVM = SVC(kernel='linear', C=10, random_state=1)
linearSVM.fit(X_train, y_train)
y_pred_linear = linearSVM.predict(X_test)
score_linear = accuracy_score(y_test, y_pred_linear)
print("accuracy_linear:", score_linear)

plot_regions3D(X_combined, y_combined, linearSVM, test_idx=range(X_train.shape[0], X_xor.shape[0]))
"""
ax.set_title("XOR gate samples kernel")
ax.set_xlabel("first element")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("second element")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("product of first & second element")
ax.w_zaxis.set_ticklabels([])
"""
plt.legend(loc='best')
plt.tight_layout()
plt.show()