import numpy as np
import sys


class MLP(object):
    """ Feedforward neural network / Multi-layer perceptron
    
    Parameters:
        n_hidden (int, optional): number of hidden units. Defaults to 30.
        l2 (float, optional): Lambda value for L2-regularization. No reguralization if l2=0. Defaults to 0..
        epochs (int, optional): Number of passes over training dataset. Defaults to 100.
        eta (float, optional): Learning rate. Defaults to 0.001.
        shuffle (bool, optional): Shuffles training data every epoch if True to prevent cycles. Defaults to True.
        minibatch_size (int, optional): Number of training examples per minibatch. Defaults to 1.
        seed (int, optional): Random sead to initialize weights and shuffling. Defaults to None.
    """

    def __init__(self, n_hidden=30, l2=0., epochs=100, eta=0.001, shuffle=True, minibatch_size=1, seed=None):
        self.random = np.random.default_rng(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size
    
    def _onehot(self, y, n_classes):
        """Encode labels into one-hot representation

        Args:
            y (array, shape = n_examples): target values
            n_classes (int): Number of unique labels
        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)

        Args:
            z : Total input
        """
        return 1./ (1. + np.exp(-np.clip(z, -250, 250)))
    
    def _forward(self, X):
        """Compute forward propagation step

        Args:
            X (array): Input layer with original features.
        """
        # Step 1: net input of hidden layer
        z_h = np.dot(X, self.w_h) + self.b_h

        # Step 2: activation of hidden layer
        a_h = self._sigmoid(z_h)

        # Step 3: net input of output layer
        z_out = np.dot(a_h, self.w_out) + self.b_out

        # Step 4: activation output layer
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    
    def _compute_cost(self, y_enc, output):
        """Compute cost function

        Args:
            y_enc (array): One-hot encoded class labels
            output (array): Activation of the output layer
        """
        L2_term = self.l2*(np.sum(self.w_h**2.)+np.sum(self.w_out**2.))
        term1 = -y_enc*(np.log(output))
        term2 = (1. - y_enc)*np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term
        return cost
    
    def predict(self, X):
        """Predict class labels

        Args:
            X (array): Input layer with original features.
        """
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred
    
    def fit(self, X_train, y_train, X_valid, y_valid):
        """Learn weights from training data.

        Args:
            X_train (array): Input layer with original features.
            y_train (array): Target class labels.
            X_valid (array): Sample features for validation during training.
            y_valid (array): Sample labels for validation during training.
        """

        n_output = np.unique(y_train).shape[0]
        n_features = X_train.shape[1]
        
        #################################
        # Weight initialization
        #################################

        # weights for input -> hidden
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden))

        # weights for hidden -> output
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden, n_output))

        epoch_strlen = len(str(self.epochs))
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}

        y_train_enc = self._onehot(y_train, n_output)

        # iterate over training epochs
        for i in range(self.epochs):
            # iterate over minibaches
            indices = np.arange(X_train.shape[0])
            if self.shuffle:
                self.random.shuffle(indices)
            for start_idx in range(0, indices.shape[0] - self.minibatch_size + 1, self.minibatch_size):  # start indices of all batches
                end_idx = start_idx+self.minibatch_size
                batch_idx = indices[start_idx: end_idx]     # indices contained in this batch

                # forward propagation
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                ######################
                # Backpropagation
                ######################

                # error of output from the true labels
                delta_out = a_out - y_train_enc[batch_idx]

                sigmoid_derivative_h = a_h*(1. - a_h)

                # error of the hidden layers
                delta_h = np.dot(delta_out, self.w_out.T)*sigmoid_derivative_h

                grad_w_h = np.dot(X_train[batch_idx].T, delta_h)
                grad_b_h = np.sum(delta_h, axis=0)

                grad_w_out = np.dot(a_h.T, delta_out)
                grad_b_out = np.sum(delta_out, axis=0)

                # regularization and weight updates
                delta_w_h = grad_w_h + self.l2*self.w_h
                delta_b_h = grad_b_h    # bias is not reguralized
                self.w_h -= self.eta*delta_w_h
                self.b_h -= self.eta*delta_b_h

                delta_w_out = grad_w_out + self.l2*self.w_out
                delta_b_out = grad_b_out
                self.w_out -= self.eta*delta_w_out
                self.b_out -= self.eta*delta_b_out

            ######################
            # Evaluation
            ######################

            # Evaluation after each epoch during training
            z_h, a_h, z_out, a_out = self._forward(X_train)

            cost = self._compute_cost(y_enc=y_train_enc, output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = (np.sum(y_train==y_train_pred)).astype(np.float)/X_train.shape[0]
            valid_acc = (np.sum(y_valid==y_valid_pred)).astype(np.float)/X_valid.shape[0]

            sys.stderr.write(f'\r{i+1}/{self.epochs} | Cost: {cost: .2f} | Train/Valid Acc.: {train_acc*100: .2f}/{valid_acc*100: .2f}')
            sys.stderr.flush()

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self