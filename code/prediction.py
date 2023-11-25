import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

SEED = 123


class Multinomial_Logistic_Regression(BaseEstimator, ClassifierMixin):
    def __init__(self, X, y, learningRate=0.005, max_epoch=3000):
        """ """

        self.learningRate = learningRate
        self.max_epoch = max_epoch

        # What is this and how is it used?
        # self.weight = np.array(
        #     [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        # )

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

    def cost_derivate_gradient(self, n, Ti, Oi, X):
        result = -(np.dot(X.T, (Ti - Oi))) / n
        return result

    def function_cost_J(self, n, Ti, Oi):
        result = -(np.sum(Ti * np.log(Oi))) / n
        return result

    def softmax(self, z):
        """
        Calculates and returns the softmax function
        Note: np.exp is e to a power
        """
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    def fit(self, X, y):
        epochCount = 0
        n = len(X)
        gradientE = []
        v_epochs = []
        totalError = []
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
            # Ti = self.one_hot_encoding(y)
            # Z = np.matmul(X, self.weight)
            # Oi = self.softmax(Z)
            # erro = self.function_cost_J(n, Ti, Oi)
            # gradient = self.cost_derivate_gradient(n, Ti, Oi, X)
            # self.weight = self.weight - self.learningRate * gradient
            # if epochCount % 100 == 0:
            #     totalError.append(erro)
            #     gradientE.append(gradient)
            #     v_epochs.append(epochCount)
            #     print("Epoch ", epochCount, " Total Error:", "%.4f" % erro)

            # epochCount += 1

            z = X @ self.weights + self.bias
            weights_gradient = (1 / n) * np.dot(
                X.T, self.softmax(z) - self.one_hot_encoding(y, c)
            )
            bias_gradient = (1 / n) * np.sum(
                self.softmax(z) - self.one_hot_encoding(y, c)
            )

            self.weights = self.weights - self.learningRate * weights_gradient
            self.bias = self.bias - self.learningRate * bias_gradient

            # Computing the loss
            loss = -np.mean(np.log(self.softmax(z)[np.arange(len(y)), y]))
            loss_array.append(loss)

            if epochCount % 100 == 0:
                # totalError.append(erro)
                # gradientE.append(gradient)
                v_epochs.append(epochCount)
                print("Epoch: {} , Loss: {}".format(epoch, loss))

            epochCount += 1

        # self.show_err_graphic(v_epochs,totalError)
        # return self
        print("{self.weights}")
        print("{self.bias}")

    def predict(self, X, y_actual):
        z = X @ self.weights + self.bias
        y_hat = self.softmax(z)

        # Returning highest probability class.
        predictions = np.argmax(y_hat, axis=1)

        # print accuracy
        accuracy = np.sum(y_actual == predictions) / len(y_actual)
        print(accuracy)

        return predictions
