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
    
    def __init__(self, input, n_in, n_out):
        # Create W(0 matrix) and b(0 vector)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out), dtype=theano.config.floatX), name='b', borrow=True)

        # Softmax
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        # Getting maximal value
        # Because softmax function return each class's probability in order to classify many class 
        # Since you want to choose the highest pobability class as y_pred
        # In this case, Softmax function is activation function at output
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        # The mean Log-Likelihood across the minibatch
        # In multi-class logistic regression, it is very common to use the negative log likelihood as the loss
        # This is cost which we should decrease
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # Check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type))
        # Check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # The T.neq operator returns a vector of 0s and 1s, where 1 represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

# This method for loading data of MNIST
# After loading the date, creating test data, valid data and training data
def load_data(dataset):
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
    if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
        dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('loading data...')

    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX),borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX),borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    # Early-stopping combats overfitting by monitoring the model’s performance on a validation set.
    # If the model’s performance ceases to improve sufficiently on the validation set, 
    #or even degrades with further optimization, then the heuristic implemented here gives up on much further optimization.
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=600):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # Compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print('building the model...')

    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')    # data, presented as rasterized images
    y = T.ivector('y')   # labels, presented as 1D vector of [int] labels
    # You can check these means at this link
    # http://deeplearning.net/software/theano/library/tensor/basic.html

    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    # The cost we minimize during training is the negative log likelihood of the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    # Compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # Update the theta 
    updates = [(classifier.W, classifier.W - learning_rate * g_W), (classifier.b, classifier.b - learning_rate * g_b)]
    train_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print('training the model...')

    # Early stopping parameter
    # Early stopping is for preventing from over learning
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)
    best_validation_loss = numpy.inf
    test_score = 0.


    # Traning parameter
    done_looping = False
    epoch = 0

    start_time = timeit.default_timer()

    while(epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        # Minibatch learning
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss * improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('epoch %i, minibatch %i/%i, test error of best model %f %%') % (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

                if patience <= iter:
                    done_looping = True
                    break

    end_time = timeit.default_timer()
    print(('Optimazation complete with best validation score of %f %%,' 'with test performance %f %%') % (best_validation_loss * 100., test_score * 100.))
    print('The code run for %d epochs, with %f epoch/sec' % (epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

def predict():
    classifier = pickle.load('best_model.pkl')

    predict_model = theano.function(inputs=[classifier.input], outputs=classifier.y_pred)
    
    dataset = 'mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_z.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print('Predicted values for the first 10 examples in test set:')
    print(predicted_values)

if __name__ == '__main__':
    sgd_optimization_mnist()
