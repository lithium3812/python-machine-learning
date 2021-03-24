import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from decision_boundary import plot_regions


"""
Demonstrate Adaptive Boosting algorithm using the wine dataset
For simplicity use only class 2 and 3, and only two features Alcohol and OD280/OD315
"""
df = pd.read_csv('wine_data.csv')
df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 
'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

# drop class 1
df = df[df['Class label']!=1]
y = df['Class label'].values
X = df[['Alcohol', 'OD280/OD315 of diluted wines']].values
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Use decision tree as the base model
tree = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=1)
# AdaBoost with 500 trees
ada = AdaBoostClassifier(base_estimator=tree, n_estimators=500, learning_rate=0.1, random_state=1)

"""compare performances of a single tree and the AdaBoost model"""
# Single tree
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print(f"Decision tree train/test accuracies {tree_train: .2f}/{tree_test: .2f}")

# Bagging model
ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print(f"AdaBoost train/test accuracies {ada_train: .2f}/{ada_test: .2f}")

# Plot decision boundaries
for model, label in zip([tree, ada], ['Single tree', 'AdaBoost (500 trees)']):
    plt.figure(f'{label}')
    plot_regions(X, y, classifier=model)
    plt.xlabel('OD280/OD315 of diluted wines')
    plt.ylabel('Alcohol')
plt.show()