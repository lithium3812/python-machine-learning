import numpy as np


class AdalineBGD(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=42):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.w_ = np.random.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        print("initial weights:", self.w_)
        self.cost_ = []
        for _ in range(self.n_iter):
            output = self.activation(self.net_input(X))
            errors = y - output
            self.w_[0] += self.eta*sum(errors)
            self.w_[1:] += self.eta*X.T.dot(errors)
            cost = (y - output).dot(y - output)/2
            self.cost_.append(cost)
        print("final weights:", self.w_)
        print("final cost:", cost)
        return
    
    def net_input(self, X):
        lin_com = np.dot(X, self.w_[1:]) + self.w_[0]
        return lin_com
    
    def activation(self, X):
        return X

    def predict(self, X):
        label = np.where(self.net_input(X) >= 0.0, 1, -1)
        return label


class AdalineSGD(object):
    def __init__(self, eta=0.01, n_iter=15, shuffle=True, random_state=42):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.shuffle = shuffle

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        print("initial weights:", self.w_)
        self.cost_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost_list = []
            for xi, target in zip(X, y):
                cost = self._update_weights(xi, target)
                cost_list.append(cost)
            avg_cost = sum(cost_list)/len(y)
            self.cost_.append(avg_cost)
        print("final weights:", self.w_)
        print("final cost:", avg_cost)
        return self
    
    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape(1))
        if y.ravel().shape(0) > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(xi, target)
        return self
    
    def _shuffle(self, X, y):
        shuffled_idx = self.rng.permutation(len(y))
        return X[shuffled_idx], y[shuffled_idx]
    
    def _initialize_weights(self, m):
        self.rng = np.random.default_rng(self.random_state)
        self.w_ = self.rng.normal(loc=0.0, scale=0.01, size=1+m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_[0] += self.eta*error
        self.w_[1:] += self.eta*error*xi
        cost = 0.5*error**2
        return cost
        
    def net_input(self, X):
        lin_com = np.dot(X, self.w_[1:]) + self.w_[0]
        return lin_com
    
    def activation(self, X):
        return X

    def predict(self, X):
        label = np.where(self.net_input(X) >= 0.0, 1, -1)
        return label
