"""
Solution to Problem 5, Homework 2, COMS 4771 Machine Learning, Fall 2017
"""

from NN import *
from data import *
from utils import *
from math import *
import numpy as np
import cPickle as pkl
from scipy.io import loadmat
import random
import os
import sys
import pdb

__author__ = "Zhuoran Liu <zl2621@columbia.edu>"
__date__ = "$Oct 21, 2017"


def runNN(hidden_dim, data_file='./data/hw2data.mat'):
    dat = Data('./data/hw2data.mat')
    mdl = TwoLayerFeedforward(1, hidden_dim, 1, name='NN-'+str(hidden_dim))
    model_path = './Q5/models/'    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    optim = SGD(1e-3, 1e-5)
    print('Training: hidden_dim={}'.format(hidden_dim))
    optim(dat, mdl, model_path+mdl.name+'.mdl')
    plot_curve(dat, mdl, './fig/5-{}.png'.format(hidden_dim))

if __name__ == '__main__':
    runNN(100)
