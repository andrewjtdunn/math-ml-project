import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

SEED = 123


class Multinomial_Logistic_Regression(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.005, max_epoch=3000):
        """ """

        self.learning_rate = learning_rate
        self.max_epoch = max_epoch

    def one_hot_encoding(self, y, c=3):
        """
        Generates one-hot encoding in a matrix

        Inputs:
            y (np array): array with multiple classes to predict
            c (int): number of unique values in c
                optional -- not calculating will speed up function

        Returns:
            y_encoded (np array): array with a column for every class, and
            a 1 for codings of that class, 0 otherwise
        """
        if not c:
            c = len(np.unique(y))
        y_encoded = np.zeros((len(y), c))
        y_encoded[np.arange(len(y)), y] = 1

        return y_encoded

    def softmax(self, z):
        """
        Calculates and returns the softmax function
        Note: np.exp is e to a power
        """
        # return np.exp(z) / np.sum(np.exp(z), axis=0)

        exp_mat = np.exp(z - np.max(z)) / np.sum(
            np.exp(z - np.max(z)), axis=1, keepdims=True
        )

        return exp_mat

    def fit(self, X, y):
        loss_array = []

        # define key matrix values
        c = len(np.unique(y))
        n, p = X.shape

        # set seed in numpy random
        np.random.seed(SEED)

        # initialize weights at random values
        self.weights = np.random.random((p, c))
        self.bias = np.random.random(c)

        for epoch in range(self.max_epoch):
            z = X @ self.weights + self.bias

            weights_gradient = (1 / n) * np.dot(
                X.T, self.softmax(z) - self.one_hot_encoding(y, c)
            )

            bias_gradient = (1 / n) * np.sum(
                self.softmax(z) - self.one_hot_encoding(y, c)
            )

            self.weights = self.weights - self.learning_rate * weights_gradient
            self.bias = self.bias - self.learning_rate * bias_gradient

            # Computing the loss
            loss = -np.mean(np.log(self.softmax(z)[np.arange(len(y)), y]))
            loss_array.append(loss)

            if epoch % 100 == 0:
                print("Epoch: {} , Loss: {}".format(epoch, loss))

        # # Print the weights and bias, which are saved within the class
        # print(f"{self.weights=}")
        # print(f"{self.bias=}")

    def predict(self, X, y_actual):
        """ """
        z = X @ self.weights + self.bias
        y_hat = self.softmax(z)

        # Store the highest probability class.
        predictions = np.argmax(y_hat, axis=1)

        # Print accuracy
        accuracy = np.sum(y_actual == predictions) / len(y_actual)
        print(f"{accuracy=}")

        return predictions
