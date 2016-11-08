from __future__ import print_function

__docformat__ = 'restructedtext en'

import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
from logreg import LogisticRegression, load_data

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
        self.params = [self.W, self.b]

class MLP(object):
    
    def __init__(self, rng, input, n_in, n_hidden, n_out):

        # The activation function can be replaced by sigmoid or any other nonlinear function.
        self.hiddenLayer = HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=n_hidden, activation=T.tanh)

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

        self.errors = self.logRegressionLayer.errors

        # The parameters of the model are the parameters of the two layer it is made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        self.input = input

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    
    print(dataset)
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print('building the model...')

    index = T.lscalar() # Index to a mini-batch
    x = T.matrix('x')   # The data is presented as rasterized images
    y = T.ivector('y')   # The labels are presented as 1D vector of int labels

    rng = numpy.random.RandomState(1234)

    # Construct the MLP class
    classifier = MLP(rng=rng, input=x, n_in=28 * 28, n_hidden=n_hidden, n_out=10)

    # The cost we minimize during training is the negative log likelihood of the model plus the regularization terms (L1 and L2); cost is expressed here symbolically
    cost = (classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr)

    test_model = theano.function(inputs=[index], outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size], 
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(inputs=[index], outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size], 
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # Compute the gradient of cost with respect to theta (sorted in params) the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, params) for params in classifier.params]

    # given two lists of the same length, A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4], 
    # zip generates a list C of same size, where each element is a pair formed from the two lists : C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)]

    # compiling a Theano function `train_model` that returns the cost, but in the same time updates the parameter of the model based on the rules defined in `updates`
    train_model = theano.function(inputs=[index], outputs=cost, updates=updates, 
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size], 
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print('train...')

    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()
    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                # If we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # Imporove patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss * improvement_threshold):
                        patience = max(patience, iter * patience_increase)
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # Test it on the test set
                        test_losses = [test_model(i) for i in range(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                        print(('epoch %i, minibatch %i/%i, test error of best model %f %%') % (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% obtained at iteration %i, with test performance %f %%') %
        (best_validation_loss * 100., best_iter + 1, test_score* 100.))
    print(('The code for file ' + os.path.split(__file__)[1] + ' run for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

if __name__ == '__main__':
    test_mlp()
