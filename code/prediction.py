import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class Multinomial_Logistic_Regression(BaseEstimator, ClassifierMixin):
    def __init__(self, X, y, learningRate=0.005, max_epoch=3000):
        """ """

        self.learningRate = learningRate
        self.max_epoch = max_epoch
        self.X = X
        self.y = y

        # What is this and how is it used?
        # 4x3 array - each row corresponds to weights for a specific feature (4)
        # each column corresponds to weights for a specific class (3 classes)
        # weights are randomly initalized or set to small values close to zero, 
        # but different methods for initialization can
        # lead to different results, so the process of weight initalization is
        # commonly explored / experimented with
        
        self.weight = np.array(
            [[0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3], 
            [0.1, 0.2, 0.3]] # added an additional row of weights for bias
        )

    def one_hot_encoding(self, y):
        """
        Generates one-hot encoding in a matrix

        Inputs:
            y (np array): array with multiple classes to predict

        Returns:
            y_encoded (np array): array with a column for every class, and
            a 1 for codings of that class, 0 otherwise
        """
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
        # return np.exp(z) / np.sum(np.exp(z), axis=0)

        return (np.exp(z).T / np.sum(np.exp(z),axis=1)).T 

    def fit(self, X, y):
        epochCount = 0
        n = len(X)
        gradientE = []
        v_epochs = []
        totalError = []

        while epochCount < self.max_epoch:
            Ti = self.one_hot_encoding(y)
            Z = np.matmul(X, self.weight)
            Oi = self.softmax(Z)
            erro = self.function_cost_J(n, Ti, Oi)
            gradient = self.cost_derivate_gradient(n, Ti, Oi, X)
            self.weight = self.weight - self.learningRate * gradient
            if epochCount % 100 == 0:
                totalError.append(erro)
                gradientE.append(gradient)
                v_epochs.append(epochCount)
                print("Epoch ", epochCount, " Total Error:", "%.4f" % erro)

            epochCount += 1

        # self.show_err_graphic(v_epochs,totalError)
        return self

    def predict(self, X, y):
        acc_set = acc_vers = acc_virg = 0
        v_resp = []
        n = len(y)
        Z = np.matmul(X, self.weight)
        Oi = self.softmax(Z)
        prevision = np.argmax(Oi, axis=1)
        # self.show_probability(Oi)
        # print("")
        procent = sum(prevision == y) / n
        # print(" ID-Sample  | Class Classification |  Output |   Hoped output  ")
        # for i in range(len(prevision)):
        #     if(prevision[i] == 0): print(" id :",i,"          | Iris-Setosa        |  Output:",prevision[i],"   |",y[i])
        #     elif(prevision[i] == 1): print(" id :",i,"          | Iris-Versicolour   |  Output:",prevision[i],"   |",y[i])
        #     elif(prevision[i] == 2): print(" id :",i,"          | Iris-Virginica     |  Output:",prevision[i],"   |",y[i])

        for i in range(len(prevision)):
            if (prevision[i] == y[i]) and (prevision[i] == 0):
                acc_set += 1
            elif (prevision[i] == y[i]) and (prevision[i] == 1):
                acc_vers += 1
            elif (prevision[i] == y[i]) and (prevision[i] == 2):
                acc_virg += 1

        correct = procent * 100
        incorrect = 100 - correct
        v_resp.append(correct)
        v_resp.append(incorrect)
        # self.accuracy_graphic(v_resp)
        return "%.2f" % (correct), acc_set, acc_vers, acc_virg
