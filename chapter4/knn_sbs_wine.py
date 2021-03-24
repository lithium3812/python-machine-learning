import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sequential_backward_selection import SBS


df = pd.read_csv('wine_data.csv')
X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

k3 = list(sbs.subsets_[10])
print(df.columns[1:][k3])

knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Trest accuracy:', knn.score(X_test_std, y_test))

knn.fit(X_train_std[:, k3], y_train)
print('Training accuracy with 3 features:', knn.score(X_train_std[:, k3], y_train))
print('Trest accuracy with 3 features:', knn.score(X_test_std[:, k3], y_test))

k_features = [len(k) for k in sbs.subsets_]

plt.plot(k_features, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.grid()
plt.tight_layout()
plt.show()