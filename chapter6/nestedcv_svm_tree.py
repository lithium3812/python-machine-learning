import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# breast cancer dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
# 1st column is IDs
X = df.loc[:, 2:].values
# malignant or benign
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Pipeline of scaling, SVM
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))

# hyperparamer to grid search
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid_svc = [
    {'svc__C': param_range, 'svc__kernel': ['linear']},
    {'svc__C': param_range, 'svc__kernel': ['rbf'], 'svc__gamma': param_range}
    ]
param_grid_tree = [{'max_depth': [1,2,3,4,5,6,7,None]}]

# Grid search in training folds
gs_svm = GridSearchCV(estimator=pipe_svc, param_grid=param_grid_svc, scoring='accuracy', cv=2)
gs_tree = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), param_grid=param_grid_tree, scoring='accuracy', cv=2)
# Outer cross validation
scores_svm = cross_val_score(gs_svm, X_train, y_train, scoring='accuracy', cv=5)
scores_tree = cross_val_score(gs_tree, X_train, y_train, scoring='accuracy', cv=5)
print(f"CV accuracy with SVM: {np.mean(scores_svm): .3f} +/-{np.std(scores_svm): .3f}")
print(f"CV accuracy with tree: {np.mean(scores_tree): .3f} +/-{np.std(scores_tree): .3f}")