from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

class LogisticRegression(object):
    
    def __init__(self, input, n_in, nout):
        self.W = theano.shared(value=numpy.zeros(n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=numpy.zeros(n_out), dtype=theano.config.floatX), name='b', borrow=True)

        # Softmax
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        # Get maximal value
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        # The mean Log-Likelihood across the minibatch
        return -T.mean(T.log(self.p_y_given_x)[T.arrange(y.shape[0]), y)

    def errors(self, y):
        if y.ndim = != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

