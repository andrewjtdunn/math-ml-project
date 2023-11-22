import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


def scatter_graph(df):
    """ """
    scatter_matrix(df, alpha=0.5, figsize=(20, 20))
    plt.show()
