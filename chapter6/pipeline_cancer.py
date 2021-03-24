import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
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

# Pipeline of scaling, PCA, logistic regression
pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(random_state=1, solver='lbfgs'))
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
score = pipe_lr.score(X_test, y_test)
print(f"Test accuracy: {score: .3f}")

# Stratified Kfold cross validation
kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print(f"Fold: {k+1}, class dist.: {np.bincount(y_train[train])}, Accuracy: {score: .3f}")
# Kfold average accuracy and its variance
print(f"Average CV accuracy: {np.mean(scores): .3f} +/- {np.std(scores): .3f}")

# sklearn CV scorer
scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=-1)
print("CV accuracy scores for each fold:")
for score in scores:
    print(f"{score: .3f}")
print(f"Average CV accuracy: {np.mean(scores): .3f} +/- {np.std(scores): .3f}")