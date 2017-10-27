"""
Solution to Problem 5, Homework 2, COMS 4771 Machine Learning, Fall 2017
"""

import random

from scipy.io import loadmat


class Data(object):
    """Data Controller"""
    def __init__(self, data_file):
        super(Data, self).__init__()
        raw = loadmat(data_file)
        self.X = raw['X'].astype(float).reshape(-1)
        self.Y = raw['Y'].astype(float).reshape(-1)
        self.cursor = -1
        self.size = self.X.shape[0]
        self.order = list(range(self.size))  # compatible with python3

    def reset(self):
        self.cursor = -1

    def next(self):
        self.cursor = (self.cursor + 1) % self.size
        if self.cursor == 0:
            random.shuffle(self.order)
        x = self.X[self.order[self.cursor]]
        y = self.Y[self.order[self.cursor]]
        return x, y
