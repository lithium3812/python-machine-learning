import numpy as np


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=42):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.w_ = np.random.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta*(target - self.predict(xi))
                self.w_[0] += update
                self.w_[1:] += update*xi
                errors += int(update != 0.0)
                #print("diff:", target - self.predict(xi))
            self.errors_.append(errors)
            print(self.w_)
            print(f"errors: {errors}")
        return
    
    def net_input(self, X):
        lin_com = np.dot(X, self.w_[1:]) + self.w_[0]
        return lin_com

    def predict(self, X):
        label = np.where(self.net_input(X) >= 0.0, 1, -1)
        return label
