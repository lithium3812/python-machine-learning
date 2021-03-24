from sklearn import datasets
from adaline_classifier import AdalineBGD, AdalineSGD
import numpy as np
import matplotlib.pyplot as plt
import random
from decision_boundary import plot_regions


iris = datasets.load_iris()
X = iris.data[np.where(iris.target!=2), :2][0]
y = iris.target[np.where(iris.target!=2)]
y = np.where(y==0, -1, 1)
# iris dataset is ordered by label and it ruins training, hence shuffle them
dataset = []
for feature, target in zip(X, y):
    sample = list(feature)
    sample.append(target)
    dataset.append(sample)
random.shuffle(dataset)
dataset = np.array(dataset)
X = dataset[:, :2]
y = dataset[:, 2]
# standardize features
X[:,0] = (X[:,0] - X[:,0].mean())/X[:,0].std()
X[:,1] = (X[:,1] - X[:,1].mean())/X[:,1].std()
"""
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
"""
model = AdalineSGD(eta=0.01, n_iter=15)
model.fit(X, y)
predict = model.predict(X)

# plot cost function at each training batch
plt.figure('training curve', figsize=(8, 6))
plt.plot(range(1, model.n_iter+1), model.cost_, marker='o')
plt.xlabel('epochs')
plt.ylabel('sum-squared-error')
plt.tight_layout()

# plot samples and decision boundary
plt.figure('setosa & versicolor', figsize=(8, 6))
plot_regions(X, y, model)
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc='upper left')
plt.show()
"""
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
"""