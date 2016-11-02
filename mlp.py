from __future__ import print_function

__docformat__ = 'restructedtext en'

import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
import logreg import LogisticRegression, load_data

#Typical hidden layer of a MLP: units are fully-connected and have sigmoidal activation function.
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):

        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden)) for tanh activation function.
        # optimal initialization of weights is dependent on the activation function used (among other things).
        # For example, results presented in [Xavier10] suggest that you should use 4 times larger initial weights for sigmoid compared to tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=-numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                # The output of uniform if converted using asarray to dtype theano.config.floatX so that the code is runable on GPU.
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        # We used a given non-linear function as the activation function of the hidden layer. 
        # By default this is tanh, but in many cases we might want to use something else.
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

class MLP(object):
    
    def __init__(self, rng, input, n_in, n_hidden, n_out):

        # The activation function can be replaced by sigmoid or any other nonlinear function.
        self.hiddenLayer = HiddenLayer(rng=rng, input=input, n_in=n_in, o_output=n_hidden, activation=T.tanh

        # The logistic regression layer gets as input the hidden units of the hidden layer
        self.logRegressionLayer = LogisticRegression(input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_out)
        
        # L1 norm ; one regularization option is to enforce L1 norm to be small
        ### addtion ###
        self.L1 = (abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W.sum()))
        
        # Square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
        ### addtion ###
        self.L2_sqr = ((self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum())

        # Negative log likelihood of the MLP is given by the negative log likelihood of the output of the model, computed in the logistic regression layer
        self.negative_log_likelihood = (self.logRegressionLayer.negative_log_likelihood)

        # The parameters of the model are the parameters of the two layer it is made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

    # The cost we minimize during training is the negative log likelihood of the model plus the regularization terms (L1 and L2); cost is expressed here symbolically
    cost = (classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr)

    # Compute the gradient of cost with respect to theta (sorted in params) the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, params) for params in classifier.params]






