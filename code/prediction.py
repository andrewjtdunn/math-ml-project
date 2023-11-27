import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin

SEED = 123


class Multinomial_Logistic_Regression(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.005, max_epoch=3000):
        """ """

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

            # Computing the loss
            loss = -np.mean(np.log(self.softmax(z)[np.arange(len(y)), y]))
            loss_array.append(loss)

            if epoch % 100 == 0:
                print("Epoch: {} , Loss: {}".format(epoch, loss))

        # # Print the weights and bias, which are saved within the class
        # print(f"{self.weights=}")
        # print(f"{self.bias=}")
        print("Weights and bias are updated")

    def predict(self, X, y):
        acc_euth = 0
        acc_adoption = 0
        acc_foster = 0
        acc_return = 0
        acc_transfer = 0
        acc_no_outcome = 0

        v_resp = []
        n = len(y)

        z = X @ self.weights + self.bias
        y_hat = self.softmax(z)
        prediction = np.argmax(y_hat, axis=1)

        percent = sum(prediction == y) / n
        # print(" ID-Sample  | Class Classification |  Predicted |   Actual  ")
        for i in range(len(prediction)):
            if prediction[i] == 0:
                print(
                    " id :",
                    i,
                    "    | Euthanasia |  Output:",
                    prediction[i],
                    "   |",
                    y[i],
                )
            elif prediction[i] == 1:
                print(
                    " id :",
                    i,
                    "    | Adoption   |  Output:",
                    prediction[i],
                    "   |",
                    y[i],
                )
            elif prediction[i] == 2:
                print(
                    " id :",
                    i,
                    "    | Foster     |  Output:",
                    prediction[i],
                    "   |",
                    y[i],
                )
            elif prediction[i] == 3:
                print(
                    " id :",
                    i,
                    "    | Return     |  Output:",
                    prediction[i],
                    "   |",
                    y[i],
                )
            elif prediction[i] == 4:
                print(
                    " id :",
                    i,
                    "    | Transfer   |  Output:",
                    prediction[i],
                    "   |",
                    y[i],
                )
            elif prediction[i] == 5:
                print(
                    " id :",
                    i,
                    "    | Non-Outcome|  Output:",
                    prediction[i],
                    "   |",
                    y[i],
                )

        for i in range(len(prediction)):
            if (prediction[i] == y[i]) and (prediction[i] == 0):
                acc_euth += 1
            elif (prediction[i] == y[i]) and (prediction[i] == 1):
                acc_adoption += 1
            elif (prediction[i] == y[i]) and (prediction[i] == 2):
                acc_foster += 1
            elif (prediction[i] == y[i]) and (prediction[i] == 3):
                acc_return += 1
            elif (prediction[i] == y[i]) and (prediction[i] == 4):
                acc_transfer += 1
            elif (prediction[i] == y[i]) and (prediction[i] == 5):
                acc_no_outcome += 1

        correct = percent * 100
        incorrect = 100 - correct
        v_resp.append(correct)
        v_resp.append(incorrect)

        print(
            "Percent Total Correct:",
            correct,
            "\nNum Correct Euth:",
            acc_euth,
            "\nNum Correct Adopt:",
            acc_adoption,
            "\nNum Correct Foster:",
            acc_foster,
            "\nNum Correct Return to Owner:",
            acc_return,
            "\nNum Correct Transfer:",
            acc_transfer,
            "\nNum Correct No Outcome:",
            acc_no_outcome,
        )

        # add confusion matrix
        unique_labels = np.unique(np.concatenate([y, prediction]))
        cm = confusion_matrix(y, prediction, labels=unique_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
        disp.plot()
        plt.show()

        return
