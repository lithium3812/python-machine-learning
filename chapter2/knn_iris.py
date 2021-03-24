import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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
model = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("accuracy:", score)

plt.figure(figsize=(8,6))
plot_regions(X_combined, y_combined, model, test_idx=range(X_train.shape[0], X.shape[0]))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
