"""
Demonstrate resampling technique to deal with class imbalance
Reduce class 1 in the breast cancer data so that the imbalance is more distinct
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample


# breast cancer dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
# 1st column is IDs
X = df.loc[:, 2:].values
# malignant or benign
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

# Create a dataset that includes only 40 examples of class 1 (malignant)
X_imb = np.vstack((X[y==0], X[y==1][:40]))
y_imb = np.hstack((y[y==0], y[y==1][:40]))

# Strategy to predict all samples are class 0 (benign) achieves accuracy of approximately 90%
y_pred = np.zeros(y_imb.shape[0])
accuracy = np.mean(y_pred==y_imb)*100
print(f"accuracy by predicting all as class 0: {accuracy: .1f}%")

# Class 1 before resampling
print("Number of class 1 examples before:", X_imb[y_imb==1].shape[0])

# Resampling
X_upsampled, y_upsampled = resample(X_imb[y_imb==1], y_imb[y_imb==1], replace=True, n_samples=X_imb[y_imb==0].shape[0], random_state=123)

# Class 1 after resampling
print("Number of class 1 examples after:", X_upsampled[y_upsampled==1].shape[0])

# New dataset
X_bal = np.vstack((X[y==0], X_upsampled))
y_bal = np.hstack((y[y==0], y_upsampled))

# Majority vote strategy only achives accuracy 50% now
y_pred = np.zeros(y_bal.shape[0])
accuracy = np.mean(y_pred==y_bal)*100
print(f"accuracy by predicting all as class 0: {accuracy: .1f}%")
