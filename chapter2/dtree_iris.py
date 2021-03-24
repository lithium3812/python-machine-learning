import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from decision_boundary import plot_regions
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz


iris = datasets.load_iris()
X = iris.data[:, 2:]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
"""
# standardize features
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
"""
# combine train and test set again
X_combined = np.append(X_train, X_test, axis=0)
y_combined = np.append(y_train, y_test)

# plot coefficients over varying C parameter
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=1)
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

dot_data = export_graphviz(model, filled=True, rounded=True, class_names=['Setosa', 'Versicolor', 'Virginica'], feature_names=['petal length', 'petal width'], out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')