from sklearn import datasets
from perceptron_classifier import Perceptron
import numpy as np
import matplotlib.pyplot as plt
import random


iris = datasets.load_iris()
X = iris.data[np.where(iris.target!=2), :2][0]
y = iris.target[np.where(iris.target!=2)]
y = np.where(y==0, -1, 1)
dataset = []
for feature, target in zip(X, y):
    sample = list(feature)
    sample.append(target)
    dataset.append(sample)
random.shuffle(dataset)
dataset = np.array(dataset)
X = dataset[:, :2]
y = dataset[:, 2]

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

model = Perceptron(n_iter=50)
model.fit(X, y)
predict = model.predict(X)
bound = -(model.w_[1]/model.w_[2])*np.linspace(x_min, x_max, 50) - model.w_[0]/model.w_[2]

plt.figure('setosa & versicolor', figsize=(8, 6))
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.plot(np.linspace(x_min, x_max, 50), bound)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()