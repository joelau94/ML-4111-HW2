"""
Solution to Problem 5, Homework 2, COMS 4771 Machine Learning, Fall 2017
"""

import abc

import numpy as np

from utils import init_param, sigmoid


class Module(object):
    """Super class"""
    def __init__(self, name=''):
        super(Module, self).__init__()
        self.name = name
        self.params = dict()

    @abc.abstractmethod
    def forward(self, x):
        return

    @abc.abstractmethod
    def backward(self, dx, lr):
        return


class SigmoidLinear(Module):
    """Module SigmoidLinear"""
    def __init__(self, input_dim, output_dim, use_bias=True, name=''):
        super(SigmoidLinear, self).__init__()
        self.name = name
        self.use_bias = use_bias
        self.W = init_param((input_dim, output_dim))
        self.params[self.name + '/W'] = self.W
        if self.use_bias:
            self.b = init_param((output_dim,))
            self.params[self.name + '/b'] = self.b

        self.input = None
        self.output = None

    def forward(self, _input):
        if _input.ndim <= 1:
            _input = _input.reshape(-1, 1)
        self.input = _input
        self.output = np.matmul(self.input, self.W)  # (size,in)(in,out)
        if self.use_bias:
            self.output += self.b  # (size,out)+(out,)=(size,out)
        self.output = sigmoid(self.output)
        return self.output

    def backward(self, dE_dy, lr):
        # (size,out)
        grad_b = dE_dy * self.output * (1. - self.output)
        # (in,size)(size,out)=(in,out)
        grad_W = np.matmul(self.input.T, grad_b) / self.input.shape[0]
        # (size,out)(out,in)=(size,in)
        grad_x = np.matmul(grad_b, self.W.T)

        # update params
        self.W -= lr * grad_W
        self.b -= lr * grad_b.mean(axis=0)

        # backprop error
        return grad_x


class TwoLayerFeedforward(object):
    """TwoLayerFeedforward"""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, input_dim, hidden_dim, output_dim, name=''):
        super(TwoLayerFeedforward, self).__init__()
        self.name = name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.params = {}

        # build NN
        self.L1 = SigmoidLinear(self.input_dim, self.hidden_dim,
                                name=self.name + '/L1')
        self.L2 = SigmoidLinear(self.hidden_dim, self.output_dim,
                                name=self.name + '/L2')
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
    # pylint: disable=too-few-public-methods

    def __init__(self, lr, epoch):
        super(SGD, self).__init__()
        self.lr = lr
        self.epoch = epoch

    def __call__(self, data, model, model_file):
        old_error = 1.
        new_error = .5
        # idx = 0
        for idx in range(1, self.epoch+1):
        # while abs((new_error - old_error) / old_error) > self.eps:
            # idx += 1
            X, Y = data.get()
            Y_hat = model.forward(X)
            model.backward(Y, self.lr)
            old_error = new_error
            new_error = ((Y_hat - Y) ** 2).mean()
            if idx % 100 == 0:
                print('iteration: {}, training error={}'.format(idx, new_error))
        model.save(model_file)
