"""
Estimate the error rate of ensemble classifiers with majority vote
Assume each classifier has the same error rate and they are independent from each other
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import math


# The error rate of 
def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier/2.))
    probs = [comb(n_classifier, k)*error**k*(1-error)**(n_classifier-k) for k in range(k_start, n_classifier+1)]
    return sum(probs)

# Plot over base error, ensemble of 11 classifiers
error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error) for error in error_range]
plt.plot(error_range, ens_errors, label='Ensemble error', lw=2)
plt.plot(error_range, error_range, '--', label='Base error', lw=2)
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid(alpha=0.5)
plt.show()
