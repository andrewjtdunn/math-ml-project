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
