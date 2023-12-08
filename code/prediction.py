import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

SEED = 123


class Multinomial_Logistic_Regression(BaseEstimator, ClassifierMixin):
    def __init__(self, verbose=False, learning_rate=0.005, max_epoch=3000):
        """ """

        self.verbose = verbose
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch

    def one_hot_encoding(self, y, c):
        """
        Generates one-hot encoding in a matrix

        Inputs:
            y (np array): array with multiple classes to predict
            c (int): number of unique values in c

        Returns:
            y_encoded (np array): array with a column for every class, and
            a 1 for codings of that class, 0 otherwise
        """

        y_encoded = np.zeros((len(y), c))
        y = y.astype(int)
        y_encoded[np.arange(len(y)), y] = 1

        return y_encoded

    def softmax(self, z):
        """
        Calculates and returns the softmax function
        Note: np.exp is e to a power
        """
        exp_mat = np.exp(z - np.max(z)) / np.sum(
            np.exp(z - np.max(z)), axis=1, keepdims=True
        )

        return exp_mat

    def fit(self, X, y):
        """
        to do: complete doc string
        """
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

            loss = -np.mean(np.log(self.softmax(z)[np.arange(len(y)), y.astype(int)]))
            loss_array.append(loss)

            if self.verbose:
                if epoch % 100 == 0:
                    print("Epoch: {} , Loss: {}".format(epoch, loss))

        # # Print the weights and bias, which are saved within the class
        # print(f"{self.weights=}")
        # print(f"{self.bias=}")
        print("Weights and bias are updated")
        return loss_array

    def predict(self, X, y):
        v_resp = []
        n = len(y)

        z = X @ self.weights + self.bias
        y_hat = self.softmax(z)
        prediction = np.argmax(y_hat, axis=1)

        accuracy_percent = sum(prediction == y) / n
        correct = accuracy_percent * 100
        incorrect = 100 - correct
        v_resp.append(correct)
        v_resp.append(incorrect)

        accuracy = balanced_accuracy_score(y, prediction)
        precision = precision_score(y, prediction, average="weighted")
        recall = recall_score(y, prediction, average="weighted")
        f1 = f1_score(y, prediction, average="weighted")

        return prediction, accuracy_percent, accuracy, precision, recall, f1
