import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


"""
Modeling non-linear relation between LSTAT (percentage of lower status of the population) and MEDV (the average price of houses)
Compare the results of linear, quadratic, and cubic regression
"""

df = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book-3rd-edition/master/ch10/housing.data.txt', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

X = df[['LSTAT']].values
y = df['MEDV'].values

regr = LinearRegression()

# generate quadratic and cubic features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# features to draw fitting lines
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

# linear fit
regr = regr.fit(X, y)
y_linear_fit = regr.predict(X_fit)
r2_linear = r2_score(y, regr.predict(X))

# quadratic fit
regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
r2_quad = r2_score(y, regr.predict(X_quad))

# cubic fit
regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
r2_cubic = r2_score(y, regr.predict(X_cubic))

# Plot data and fitting lines, and R2 score
plt.figure()
plt.scatter(X, y, label='training points', color='lightgray')
plt.plot(X_fit, y_linear_fit, label=f'linear (d=1), $R^2={r2_linear: .2f}$', color='blue', lw=2, linestyle=':')
plt.plot(X_fit, y_quad_fit, label=f'quadratic (d=2), $R^2={r2_quad: .2f}$', color='red', lw=2, linestyle='-')
plt.plot(X_fit, y_cubic_fit, label=f'cubic (d=3), $R^2={r2_cubic: .2f}$', color='green', lw=2, linestyle='--')
plt.xlabel('lower status of population [LSTAT] %')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper right')

"""
Log transform to deal with the non-linear relation
"""

# log transform the feature
X_log = np.log(X)
y_sqrt = np.sqrt(y)

# features to draw fitting lines
X_fit = np.arange(X_log.min()-1, X_log.max()+1, 1)[:, np.newaxis]

# fit with transformed variables
regr = regr.fit(X_log, y_sqrt)
y_log_fit = regr.predict(X_fit)
r2_log = r2_score(y_sqrt, regr.predict(X_log))

# plot samples and fitting line on the transformed space
plt.figure()
plt.scatter(X_log, y_sqrt, label='training points', color='lightgray')
plt.plot(X_fit, y_log_fit, label=f'linear (d=1), $R^2={r2_log: .2f}$', color='blue', lw=2)
plt.xlabel('log(% of lower status of population [LSTAT])')
plt.ylabel('$\sqrt{Price \; in \; \$1000s \; [MEDV]}$')
plt.legend(loc='lower left')
plt.tight_layout()

plt.show()