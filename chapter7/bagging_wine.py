import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from decision_boundary import plot_regions


"""
Demonstrate bagging algorithm using the wine dataset
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

# Take decision tree as the base model
tree = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=None)
# Bagging trees
bag = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=1)

"""compare performances of a single tree and the bagging model"""
# Single tree
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print(f"Decision tree train/test accuracies {tree_train: .2f}/{tree_test: .2f}")

# Bagging model
bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print(f"Bagging train/test accuracies {bag_train: .2f}/{bag_test: .2f}")

# Plot decision boundaries
for model, label in zip([tree, bag], ['Single tree', 'Bagging (500 trees)']):
    plt.figure(f'{label}')
    plot_regions(X, y, classifier=model)
    plt.xlabel('OD280/OD315 of diluted wines')
    plt.ylabel('Alcohol')
plt.show()
