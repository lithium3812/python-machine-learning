import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, make_scorer
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
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
# Confusion Matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

# Precision
print(f"Precision: {precision_score(y_test, y_pred): .3f}")
# Recall
print(f"Recall: {recall_score(y_test, y_pred): .3f}")
# F1 score
print(f"F1: {f1_score(y_test, y_pred): .3f}")

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(5,5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')

"""
When we want to choose which label to treat as positive, we use make_scorer
Here we demonstrate grid search with F1 score as scorer and label 0 as target class (positive)
"""
c_gamma_range = [0.01, 0.1, 1.0, 10.0]
param_grid = [
    {'svc__C': c_gamma_range, 'svc__kernel': ['linear']},
    {'svc__C': c_gamma_range, 'svc__kernel': ['rbf'], 'svc__gamma': c_gamma_range}
]
scorer = make_scorer(f1_score, pos_label=0)
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring=scorer, cv=10)
gs = gs.fit(X_train, y_train)
print("---Grid search with F1 score and label 0 as the target---")
print(f"Best F1 score: {gs.best_score_: .3f}")
print(f"Best hyperparameters: {gs.best_params_}")

plt.show()