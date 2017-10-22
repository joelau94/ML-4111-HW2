"""
Solution to Problem 5, Homework 2, COMS 4771 Machine Learning, Fall 2017
"""

import numpy as np
import cPickle as pkl
from scipy.io import loadmat
import random
from utils import *
from math import *
import os
import sys
import pdb

__author__ = "Zhuoran Liu <zl2621@columbia.edu>"
__date__ = "$Oct 21, 2017"


class Module(object):
    """Super class"""
    def __init__(self, name=""):
        super(Module, self).__init__()
        self.name = name
        self.params = dict()

class sigmoidLinear(Module):
    """Module sigmoidLinear"""
    def __init__(self, input_dim, output_dim, use_bias=True, name=""):
        super(sigmoidLinear, self).__init__()
        self.name = name
        self.use_bias = use_bias
        self.W = init_param((input_dim, output_dim), scale=5.)
        self.params[self.name+'/W'] = self.W
        if self.use_bias:
            self.b = init_param((output_dim,))
            self.params[self.name+'/b'] = self.b

        self.input = None
        self.output = None

    def forward(self, _input):
        if _input.ndim <= 1:
            _input = _input.reshape(-1,1)
        self.input = _input
        self.output = sigmoid(np.matmul(self.input, self.W)) #(size,in)(in,out)
        if self.use_bias:
            self.output += self.b #(size,out)+(out,)=(size,out)
        return self.output

    def backward(self, dE_dy, lr):
        # (size,out)
        grad_b = dE_dy * self.output * (1. - self.output)
        #(in,size)(size,out)=(in,out)
        grad_W = np.matmul(self.input.T, grad_b)
        #(size,out)(out,in)=(size,in)
        grad_x = np.matmul(grad_b, self.W.T)

        # update params
        self.W -= lr * grad_W
        self.b -= lr * grad_b.reshape(-1)

        # backprop error
        return grad_x

class TwoLayerFeedforward(object):
    """TwoLayerFeedforward"""
    def __init__(self, input_dim, hidden_dim, output_dim, name=""):
        super(TwoLayerFeedforward, self).__init__()
        self.name = name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.params = {}

        # build NN
        self.L1 = sigmoidLinear(self.input_dim, self.hidden_dim,
            name=self.name+'/L1')
        self.L2 = sigmoidLinear(self.hidden_dim, self.output_dim,
            name=self.name+'/L2')
        self.params.update(self.L1.params)
        self.params.update(self.L2.params)

        self.output = None

    def save(self, model_file):
        np.savez(model_file, **self.params)

    def load(self, model_file):
        self.params.update(np.load(model_file))

    def forward(self, _input):
        self.output = self.L2.forward(self.L1.forward(_input))
        return self.output

    def backward(self, y_gold, lr):
        dE_dy = 2 * (self.output - y_gold)
        return self.L1.backward(self.L2.backward(dE_dy, lr), lr)


class SGD(object):
    """SGD"""
    def __init__(self, lr, eps):
        super(SGD, self).__init__()
        self.lr = lr
        self.eps = eps

    def __call__(self, data, model, model_file):
        old_error = 1.
        new_error = .5
        while abs((new_error - old_error) / old_error) > self.eps:
            X, Y = data.next()
            model.forward(X)
            Y_hat = model.backward(Y, self.lr)
            old_error = new_error
            new_error = ((Y_hat - Y)**2).sum()
            if data.cursor % 100 == 0:
                print('Training: data={}, error={}'.format(data.cursor, new_error))
                #pdb.set_trace()
        model.save(model_file)       