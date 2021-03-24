import numpy as np
import matplotlib.pyplot as plt
from neuralnet import MLP


# Load the compressed MNIST data
mnist = np.load('mnist_scaled.npz')
X_train = mnist['X_train']
y_train = mnist['y_train']
X_test = mnist['X_test']
y_test = mnist['y_test']

# from-scratch implementation of Multi-Layer Perceptron
nn = MLP(n_hidden=100, l2=0.01, epochs=200, eta=0.0005, minibatch_size=100, shuffle=True, seed=1)
nn.fit(X_train=X_train[:55000], y_train=y_train[:55000], X_valid=X_train[55000:], y_valid=y_train[55000:])

# Performance on the test data
y_test_pred = nn.predict(X_test)
acc = np.sum(y_test==y_test_pred).astype(np.float)/ X_test.shape[0]
print("\n-----------------------------------")
print(f'Test accuracy: {acc*100: .2f}%')
print("-----------------------------------")

# Plot the cost function (negative maximum log-likelihood) over epochs
plt.figure('Cost')
plt.plot(range(nn.epochs), nn.eval_['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')

# Training & Validation accuracy
plt.figure('Accuracy')
plt.plot(range(nn.epochs), nn.eval_['train_acc'], label='training')
plt.plot(range(nn.epochs), nn.eval_['valid_acc'], label='validation', linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(loc='lower right')

# 25 examples of instances that our model failed to classify
miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title(f'{i+1}) True: {correct_lab[i]}/ Predict: {miscl_lab[i]}')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()

plt.show()