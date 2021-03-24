import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve


# breast cancer dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
# 1st column is IDs
X = df.loc[:, 2:].values
# malignant or benign
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Pipeline of scaling, logistic regression
pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', random_state=1, solver='lbfgs', max_iter=10000))

"""
Learning curve
"""
train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(estimator=pipe_lr, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=-1, return_times=True)
# mean and diviation of scores for all folds
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# mean and diviation of fitting time
total_time = np.sum(fit_times, axis=0)
time_mean = np.mean(total_time)
time_std = np.std(total_time)
print(f"Time for fitting: {time_mean: .3f} +/-{time_std: .3f}")

plt.figure('Learning curve')
# Plot the mean of training accuracy
plt.plot(train_sizes, train_mean, color='b', marker='o', markersize=5, label='Training accuracy')
# Plot the diviation of training accuracy as region
plt.fill_between(train_sizes, train_mean+train_std, train_mean-train_std, alpha=0.15, color='b')
# Plot the mean of training accuracy
plt.plot(train_sizes, test_mean, color='g', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
# Plot the diviation of training accuracy as region
plt.fill_between(train_sizes, test_mean+test_std, test_mean-test_std, alpha=0.15, color='g')
plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])

"""
Validation curve
"""
# Values of the hyperparamer to evaluate score
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(estimator=pipe_lr, X=X_train, y=y_train, param_name='logisticregression__C', param_range=param_range, cv=10)
# mean and diviation of scores for all folds at each paramer
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure('Validation curve')
# Plot the mean of training accuracy
plt.plot(param_range, train_mean, color='b', marker='o', markersize=5, label='Training accuracy')
# Plot the diviation of training accuracy as region
plt.fill_between(param_range, train_mean+train_std, train_mean-train_std, alpha=0.15, color='b')
# Plot the mean of training accuracy
plt.plot(param_range, test_mean, color='g', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
# Plot the diviation of training accuracy as region
plt.fill_between(param_range, test_mean+test_std, test_mean-test_std, alpha=0.15, color='g')
plt.grid()
plt.xscale('log')
plt.xlabel('Paramer C')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])

plt.show()
