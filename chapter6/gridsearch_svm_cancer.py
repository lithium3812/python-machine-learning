import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


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
param_grid = [
    {'svc__C': param_range, 'svc__kernel': ['linear']},
    {'svc__C': param_range, 'svc__kernel': ['rbf'], 'svc__gamma': param_range}
    ]

# Grid search with 10 fold cross validation
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)
gs.fit(X_train, y_train)
print(f"Best score in grid search: {gs.best_score_: .3f}")
print("Best paramer combination: \n", gs.best_params_)
# Model with the best parameters are already re-trained with the whole train dataset
clf = gs.best_estimator_
score = clf.score(X_test, y_test)
print(f"Test accuracy by the best model: {score: .3f}")
