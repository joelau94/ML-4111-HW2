"""
Solution to Problem 5, Homework 2, COMS 4771 Machine Learning, Fall 2017
"""

import random
import os

import numpy as np

from nn import TwoLayerFeedforward, SGD
from data import Data
from utils import plot_curve


random.seed(45)
np.random.seed(45)


def runNN(hidden_dim, data_file='./data/hw2data.mat'):
    dat = Data(data_file)
    mdl = TwoLayerFeedforward(1, hidden_dim, 1, name='NN-' + str(hidden_dim))
    model_path = './Q5/models/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    optim = SGD(1, 100000)
    print('Training: hidden_dim={}'.format(hidden_dim))
    optim(dat, mdl, model_path + mdl.name + '.mdl')
    plot_curve(dat, mdl, './fig/5-{}.png'.format(hidden_dim))


if __name__ == '__main__':
    runNN(32)
