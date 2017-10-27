"""
Solution to Problem 4, Homework 2, COMS 4771 Machine Learning, Fall 2017
"""

import os
import sys
import pickle as pkl

import numpy as np
from scipy.io import loadmat


class Data(object):
    """Data Controller"""
    def __init__(self, data_file):
        super(Data, self).__init__()
        raw = loadmat(data_file)
        self.X = raw['X'].astype(float) / 256.
        self.Y = raw['Y'].astype(int).reshape(-1)
        self.cursor = -1
        self.size = self.X.shape[0]

    def reset(self):
        self.cursor = -1

    def next(self, digit_1, digit_2):
        self.cursor = (self.cursor + 1) % self.size
        Y = self.Y.tolist()
        while(Y[self.cursor]!=digit_1 and Y[self.cursor]!=digit_2):
            self.cursor = (self.cursor + 1) % self.size
        if Y[self.cursor] == digit_1:
            y = np.asarray([1.])
        elif Y[self.cursor] == digit_2:
            y = np.asarray([-1.])
        return self.X[self.cursor], y # shape=(784,), shape=(1,)

    def next1(self, digit_1, digit_2, w):
        '''
        :param w: shape (784,)
        '''
        ids = np.concatenate(( np.argwhere(self.Y == digit_1).reshape(-1),
                np.argwhere(self.Y == digit_2).reshape(-1) ))
        X = self.X[ids]
        Y = np.asarray([1. if y == digit_1 else -1.
            for y in self.Y[ids].tolist()]) # (size,)
        idx = np.argmin(Y * np.matmul(X, w)) # (size,) * ((size,784)(784,))
        return X[idx], Y[idx] # shape=(784,), shape=(1,)


class Perceptron(object):
    """Super class for all versions of Perceptron"""
    def __init__(self, digit_1, digit_2, w_dim):
        super(Perceptron, self).__init__()
        self.digit_1 = digit_1
        self.digit_2 = digit_2
        self.w = np.zeros(w_dim) # shape=(784,) for v0,1,2
        self.params = {'w': self.w}

    def save(self, model_file):
        np.savez(model_file, **self.params)

    def load(self, model_file):
        self.params = np.load(model_file)
        for k, v in self.params.iteritems():
            eval('self.'+k+'=v')

class Perceptron_v0(Perceptron):
    """Perceptron_v0"""

    def train(self, data, epochs):
        for _ in range(epochs):
            x, y = data.next(self.digit_1, self.digit_2)
            if y * np.dot(self.w, x) <= 0:
                self.w += y * x

    def test(self, x):
        '''
        :param x: shape(784,) or shape(size,784)
        '''
        result = np.sign(np.matmul(self.w[np.newaxis], x.T)) #(1,784)(784,size)
        result[result == 0.] = 1.
        return result.reshape(-1) # (size,)

class Perceptron_v1(Perceptron_v0):
    """Perceptron_v1"""

    def train(self, data, epochs):
        for _ in range(epochs):
            x, y = data.next1(self.digit_1, self.digit_2, self.w)
            if y * np.dot(self.w, x) <= 0:
                self.w += y * x
            else:
                break

class Perceptron_v2(Perceptron):
    """Perceptron_v2"""
    def __init__(self, digit_1, digit_2, w_dim):
        super(Perceptron_v2, self).__init__(digit_1, digit_2, w_dim)
        self.w = self.w[np.newaxis]
        self.c = np.asarray([0.])
        self.params['c'] = self.c

    def train(self, data, epochs):
        for _ in range(epochs):
            x, y = data.next(self.digit_1, self.digit_2)
            if y * np.dot(self.w[-1], x) <= 0:
                w_new = self.w[-1] + y * x
                self.w = np.concatenate(( self.w, w_new[np.newaxis] ))
                self.c = np.concatenate(( self.c, np.asarray([1.]) ))
            else:
                self.c[-1] = self.c[-1] + 1.

    def test(self, x):
        '''
        :param x: shape(784,) or shape(size,784)
        '''
        result = np.sign( (
            self.c.reshape((-1,1)) * np.matmul(self.w, x.T)
            ).sum(axis=0) ) # (k,1) * (k,784)(784,size) --sum--> (size,)
        result[result == 0.] = 1.
        return result # (size,)

class Perceptron_v3(Perceptron):
    """Perceptron_v3: Kernel Perceptron"""
    def __init__(self, digit_1, digit_2, w_dim, degree):
        '''
        w stores alpha, w_dim is meaningless here
        w will be override by a correct one in __initParams()
        '''
        super(Perceptron_v3, self).__init__(digit_1, digit_2, w_dim)
        self.degree = degree
        self.param_init = False

    def __initParams(self, data):
        np.random.seed(1)
        ids = np.concatenate(( np.argwhere(data.Y==self.digit_1).reshape(-1),
                np.argwhere(data.Y==self.digit_2).reshape(-1) ))
        np.random.shuffle(ids)
        self.X = data.X[ids]
        self.Y = np.asarray([1. if y == self.digit_1 else -1.
            for y in data.Y[ids].tolist()]) # (size,)
        self.w = np.zeros_like(self.Y)
        self.data_cursor = 0
        self.params.update({'w': self.w, 'X': self.X, 'Y':self.Y})
        self.param_init = True

    def train(self, data, epochs):
        if not self.param_init:
            self.__initParams(data)
        x_dim = self.X.shape[0]
        for e in range(self.data_cursor, self.data_cursor + epochs):
            x = self.X[e%x_dim]
            y = self.Y[e%x_dim]
            # pred = (self.w * self.Y) * (np.matmul(self.X, self.X.T) ** self.degree).T
            # self.w[np.sign(pred.sum(axis=0)) != self.Y] += 1.
            # pdb.set_trace()
            pred = (self.w * self.Y) * (np.matmul(self.X, x) ** self.degree).T
            if np.sign(pred.sum()) != y:
                self.w[e%x_dim] += 1.
        self.data_cursor += epochs

    def test(self, x):
        result = np.sign( (
            (self.w * self.Y).reshape(-1,1) * (np.matmul(self.X, x.T) ** self.degree)
            ).sum(axis=0) ) #(size,)
        result[result == 0.] = 1.
        return result # (size,)

        

class TenDigitClassifier(object):
    """TenDigitClassifier"""
    def __init__(self, perceptron_version, x_dim=784, degree=5):
        '''
        :param perceptron_version: '0', '1', '2', or '3'
        :param degree: only work for v3
        '''
        super(TenDigitClassifier, self).__init__()
        self.version = perceptron_version
        self.data = None
        if perceptron_version == '3':
            # BE CAREFUL! DON'T EXCHANGE i AND j HERE!!!
            self.perceptrons = eval('[[Perceptron_v'
                + self.version + '(i,j,' + str(x_dim) + ',' + str(degree)
                + ') for j in range(10)] for i in range(10)]')
        else:
            # BE CAREFUL! DON'T EXCHANGE i AND j HERE!!!
            self.perceptrons = eval('[[Perceptron_v'
                + self.version + '(i,j,' + str(x_dim)
                + ') for j in range(10)] for i in range(10)]')

    def getData(self, data):
        self.data = data

    def save(self, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        for i in range(10):
            for j in range(i+1,10): # <digit_i>-<digit_j>.npz
                model_file = os.path.join(model_path,'{}-{}.npz'.format(i,j))
                self.perceptrons[i][j].save(model_file)

    def load(self, model_path):
        if not os.path.exists(model_path):
            print('Non-existent path!')
            sys.exit(1)
        for i in range(10):
            for j in range(i+1,10): # <digit_i>-<digit_j>.npz
                model_file = os.path.join(model_path,'{}-{}.npz'.format(i,j))
                self.perceptrons[i][j].load(model_file)

    def train(self, epochs):
        if self.data is not None:
            for i in range(10):
                for j in range(i+1,10):
                    self.perceptrons[i][j].train(self.data, epochs)

    def test(self, x):
        '''
        :param x: shape(784,) or shape(size,784)
        '''
        if x.ndim == 1:
            votes = np.zeros((10,1))
        else:
            votes = np.zeros((10, x.shape[0]))
        for i in range(10):
            for j in range(i+1,10):
                prediction = self.perceptrons[i][j].test(x)
                votes[i][np.argwhere(prediction == 1.)] += 1.
                votes[j][np.argwhere(prediction == -1.)] += 1.
        result = votes.argmax(axis=0) # (x.shape[0],)
        return result
