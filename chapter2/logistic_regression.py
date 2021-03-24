import numpy as np


class LogisticGD(object):
    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=42):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.shuffle=shuffle

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        print("initial weights:", self.w_)
        self.cost_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = self._update_weights(X, y)
            self.cost_.append(cost)
        print("final weights:", self.w_)
        print("final cost:", cost)
        return
    
    def _shuffle(self, X, y):
        shuffled_idx = self.rng.permutation(len(y))
        return X[shuffled_idx], y[shuffled_idx]
    
    def _initialize_weights(self, m):
        self.rng = np.random.default_rng(self.random_state)
        self.w_ = self.rng.normal(loc=0.0, scale=0.01, size=1+m)
        self.w_initialized = True

    def _update_weights(self, X, y):
        output = self.activation(X)
        error = y - output
        self.w_[0] += self.eta*sum(error)
        self.w_[1:] += self.eta*X.T.dot(error)
        cost = -y.dot(np.log(output))-(1-y).dot(np.log(1-output))
        return cost
    
    def net_input(self, X):
        lin_com = np.dot(X, self.w_[1:]) + self.w_[0]
        return lin_com
    
    def activation(self, X):
        sigmoid = 1/(1+np.exp(-self.net_input(X)))
        return sigmoid

    def predict(self, X):
        output = self.activation(X)
        label = np.where(output >= 0.5, 1, 0)
        return label
