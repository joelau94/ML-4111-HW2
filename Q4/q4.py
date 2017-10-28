"""
Solution to Problem 4, Homework 2, COMS 4771 Machine Learning, Fall 2017
"""

import os

from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import seaborn as sns

from perceptron_classifier import Data, TenDigitClassifier


def splitData(test_size, data_file, save_path='./data/'):
    data = loadmat(data_file)
    train_data = {
        'X': data['X'][:-test_size],
        'Y': data['Y'][:-test_size]
    }
    savemat(os.path.join(save_path, 'train_data.mat'),
            train_data, appendmat=False)
    test_data = {
        'X': data['X'][-test_size:],
        'Y': data['Y'][-test_size:]
    }
    savemat(os.path.join(save_path, 'test_data.mat'),
            test_data, appendmat=False)


def runClassifier(version, split=200, save_interval=500,
                  max_epoch=10000, data_path='./data/', degree=5):
    '''
    degree only make sense for v3
    '''
    # pylint: disable=too-many-arguments

    sns.set()
    train_data = Data(os.path.join(data_path, 'train_data.mat'))
    test_data = Data(os.path.join(data_path, 'test_data.mat'))
    if version == '3':
        classifier = TenDigitClassifier(str(version), degree=degree)
    else:
        classifier = TenDigitClassifier(str(version))
    classifier.getData(train_data)
    acc = []
    epochs = []
    model_path = './Q4/models/v{}/epoch_{}/'  # version of perceptron / epoch #
    for check_point in range(1, int(max_epoch / save_interval) + 1):
        epoch = check_point * save_interval
        epochs.append(epoch)
        print('Training: v={}, epoch={}'.format(version, epoch))
        classifier.train(save_interval)
        if version != '3':
            classifier.save(model_path.format(version, epoch))
        print('Testing: v={}, epoch={}'.format(version, epoch))
        test_result = classifier.test(test_data.X)
        acc.append((test_result == test_data.Y).sum() \
            / float(test_data.Y.shape[0]))
        print('Acc: {}'.format(acc[-1]))
    if version == '3':
        classifier.save(model_path.format(version, epoch))
    plt.plot(epochs, acc)
    if version == '3':
        plt.savefig('./fig/{}-{}.png'.format(version, degree))
    else:
        plt.savefig('./fig/{}-{}.png'.format(version, split))
    plt.clf()


if __name__ == '__main__':
    splitData(200, './data/hw1data.mat')
    runClassifier('0', split=200)
    runClassifier('1', split=200)
    runClassifier('2', split=200)
    splitData(500, './data/hw1data.mat')
    runClassifier('0', split=500)
    runClassifier('1', split=500)
    runClassifier('2', split=500)
    splitData(1000, './data/hw1data.mat')
    runClassifier('0', split=1000)
    runClassifier('1', split=1000)
    runClassifier('2', split=1000)
    runClassifier('3', save_interval=10, max_epoch=1000, degree=5)
    runClassifier('3', save_interval=10, max_epoch=1000, degree=7)
    runClassifier('3', save_interval=10, max_epoch=1000, degree=10)
