"""
Implement Linear Discriminant Analysis with sklearn and apply logistic regression
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from decision_boundary import plot_regions


df = pd.read_csv('wine_data.csv')
X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# LDA from sklean
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

# logistic regression
lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
lr.fit(X_train_lda, y_train)
y_pred = lr.predict(X_test_lda)
score = accuracy_score(y_test, y_pred)
print("accuracy:", score)

# plot decision boundaries
X_combined = np.append(X_train_lda, X_test_lda, axis=0)
y_combined = np.append(y_train, y_test)

plot_regions(X_combined, y_combined, classifier=lr, test_idx=range(X_train.shape[0], X.shape[0]))
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()