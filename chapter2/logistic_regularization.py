import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from decision_boundary import plot_regions


iris = datasets.load_iris()
X = iris.data[:, 2:]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# standardize features
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# combine train and test set again
X_combined = np.append(X_train, X_test, axis=0)
y_combined = np.append(y_train, y_test)

# plot coefficients over varying C parameter
weights, params, scores = [], [], []
for c in np.arange(-5, 5):
    model = LogisticRegression(C=10.**c, random_state=1, solver='lbfgs', multi_class='ovr')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    weights.append(model.coef_[1])
    params.append(10.**c)
    scores.append(accuracy_score(y_test, y_pred))
    plt.figure(figsize=(8, 6))
    plot_regions(X_combined, y_combined, model, test_idx=range(X_train.shape[0], X.shape[0]))
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc='upper left')
weights = np.array(weights)

plt.figure(figsize=(8, 6))
plt.plot(params, weights[:, 0], label='petal length [standardized]')
plt.plot(params, weights[:, 1], label='petal width [standardized]', linestyle='--')
plt.xlabel('C')
plt.ylabel('coefficients')
plt.legend(loc='upper left')
plt.xscale('log')

plt.figure(figsize=(8, 6))
plt.plot(params, scores, label='accuracy')
plt.xlabel('C')
plt.ylabel('accuracy score')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()