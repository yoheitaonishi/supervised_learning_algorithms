from __future__ import print_function

import os
import sys
import timeit

import numpy

import theano
from theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

from logreg import LogisticRegression, load_data
from mlp import HiddenLayer


class LeNetConvPoolLayer(object):

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):

        # Debug 
        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) // numpy.prod(poolside))

        # Initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX), borrow=True)

        b_values = numpy.zeros((filter_share[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # Convolve input feature maps with filters
        conv_out = conv2d(input=input, filters=self.W, filter_shape=filter_shape, input_share=image_share)
        # Pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(input=conv_out, ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first reshape it to a tensor of shape (1, n_filters, 1, 1). 
        # Each bias will thus be broadcasted across mini-batches and feature map width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

        self.input = input

def evaluate_lenet5(learning_rate=0.1, n_epochs=200, dataset='mnist.pkl.gz', nkerns=[20, 50], batch_size=500):





