"""
Solution to Problem 5, Homework 2, COMS 4771 Machine Learning, Fall 2017
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def init_param(size, scale=.01):
    return scale * np.random.randn(*size).astype('float32')


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def plot_curve(data, model, save_path):
    sns.set()
    X = data.X.tolist()
    Y = data.Y.tolist()
    Y_hat = model.forward(data.X).tolist()
    plt.scatter(X, Y, color='b', marker='.', label='Given Y')
    plt.scatter(X, Y_hat, color='r', marker='.', label='Output Y')
    plt.savefig(save_path)
    plt.clf()
