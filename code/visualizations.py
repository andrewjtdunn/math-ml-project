import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def scatter_graph(df):
    """ """
    scatter_matrix(df, alpha=0.5, figsize=(20, 20))
    plt.show()


def make_confusion_matrix(y, y_hat):
    """ """
    unique_labels = np.unique(np.concatenate([y, y_hat]))
    cm = confusion_matrix(y, y_hat, labels=unique_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    disp.plot()


def make_hyperparameter_graph(metrics, scores, xlabel):
    """ """
    accuracy_list = []
    precision_list = []
    recall_list = []
    fl_list = []

    for row in scores:
        accuracy_list.append(row[0])
        precision_list.append(row[1])
        recall_list.append(row[2])
        fl_list.append(row[3])

    # plot the stats
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))

    plt.plot(metrics, accuracy_list, label="accuracy", color="blue")
    plt.plot(metrics, precision_list, label="precision", color="red")
    plt.plot(metrics, recall_list, label="recall", color="darkgreen")
    plt.plot(metrics, fl_list, label="f1", color="thistle")

    ax.legend(loc="lower right", fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_xscale("log")
    ax.set_ylabel("Performance stats", fontsize=16)
    plt.show()
