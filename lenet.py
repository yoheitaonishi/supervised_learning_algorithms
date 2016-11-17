import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d

import numpy

rng = numpy.random.RandomState(23455)

# Instantiate 4D tensor for input
input = T.tensor4(name='input')
# Instantiate shared value for weights
w_shp = (2, 3, 9, 9)
w_bound = numpy.sqrt(3 * 9 * 9)
W = theano.shared(numpy.asarray(rng.uniform(low=-1.0 / w_bound, high=1.0 / w_bound, size=w_shp), dtype=input.dtype), name='W')

# Biases are usually initialized to zero.
# However in this paticular application, applying the convolutional Layer to an image without Learning the parameters.
# therefore initializing them to "simulate" Learning.
b_shp = (2,)
b = theano.shared(numpy.asarray(rng.uniform(low=-.5, high=.5, size=b_shp), dtype=input.dtype), name='b')

# build symbolic expression that computes the convolution of input with filters in w
conv_out = conv2d(input, W)

output = T.nnet.sigmoid(conv_out, b.dimshuffle('x', 0, 'x', 'x'))
f = theano.function([input], output)









