import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from majority_vote import MajorityVoteClassifier
from decision_boundary import plot_regions


"""
Demonstrate majority vote ensemble classifier made of logistic regression, decision tree, and KNN
Use sklearn iris dataset
Only use sepal width and petal length as features
Only use versicolor and virginica
"""
iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)

clf1 = LogisticRegression(penalty='l2', C=0.01, solver='lbfgs', random_state=1)
clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])

# Ensemble classifier with voting
clf_vote = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])

# Estimate ROC AUC of each method by 10-fold cross validation
clf_labels = ['Logistic regression', 'Decision tree', 'KNN', 'Majority voting']
all_clf = [pipe1, clf2, pipe3, clf_vote]
print('10-fold cross validation:\n')
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print(f"ROC AUC: {scores.mean(): .2f} +/-{scores.std(): .2f} [{label}]")


"""
Draw ROC curves
"""
plt.figure('ROC')
colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    # probability of class 1
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, threasholds = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=clr, linestyle=ls, label=f'{label} (AOC={roc_auc: .2f})')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], '--', color='gray', lw=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid(alpha=0.5)
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (TPR)')


"""
Boundary
"""
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)

for clf, label in zip(all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    plt.figure(f'{label}')
    plot_regions(X_train_std, y_train, clf)


"""
In hyperparameter tuning, we need to access parameters of each base classifier
get_params shows all internal parameters
"""
print("Parameters in the ensemble model:\n")
for k, v in clf_vote.get_params().items():
    print(f"{k}: {v}")


"""
grid search for inverse regularization C of LogisticRegression and max_depth of DecisionTree
"""
params = {'decisiontreeclassifier__max_depth': [1,2], 'pipeline-1__clf__C': [0.001, 0.1, 100.0]}
grid = GridSearchCV(estimator=clf_vote, param_grid=params, cv=10, scoring='roc_auc')
grid.fit(X_train, y_train)

# ROC AUC scores at each parameter setting
print("\nGrid Search results")
for i, _ in enumerate(grid.cv_results_['mean_test_score']):
    mean = grid.cv_results_['mean_test_score'][i]
    std = grid.cv_results_['std_test_score'][i]/2.0
    param_setting = grid.cv_results_['params'][i]
    print(f"{mean: .2f} +/-{std: .2f} {param_setting}")
# Best parameters
print("Best parameters:", grid.best_params_)
print(f"Accuracy: {grid.best_score_: .2f}")

plt.show()