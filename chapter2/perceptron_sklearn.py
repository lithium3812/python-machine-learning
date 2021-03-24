import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from decision_boundary import plot_regions


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
print("Label counts in y_train:", np.bincount(y_train))
print("Label counts in y_test:", np.bincount(y_test))

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

model = Perceptron()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))

'''
plt.figure('learning curve', figsize=(8, 6))
plt.plot(range(1, model.n_iter+1), model.errors_, marker='o')
plt.xlabel('epochs')
plt.ylabel('cost')
plt.tight_layout()
'''

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plt.figure('setosa & versicolor', figsize=(8, 6))
plot_regions(X=X_combined, y=y_combined,classifier=model,test_idx=range(len(y_train), len(y)))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()