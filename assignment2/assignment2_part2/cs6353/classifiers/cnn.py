from builtins import object
import numpy as np

from cs6353.layers import *
from cs6353.layer_utils import *


class ConvNet(object):
    """
   A simple convolutional network with the following architecture:

    [conv - bn - relu] x M - max_pool - affine - softmax
    
    "[conv - bn - relu] x M" means the "conv-bn-relu" architecture is repeated for
    M times, where M is implicitly defined by the convolution layers' parameters.
    
    For each convolution layer, we do downsampling of factor 2 by setting the stride
    to be 2. So we can have a large receptive field size.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[32], filter_sizes=[7],
                 num_classes=10, weight_scale=1e-3, reg=0.0,use_batch_norm=True, 
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer. It is a
          list whose length defines the number of convolution layers
        - filter_sizes: Width/height of filters to use in the convolutional layer. It
          is a list with the same length with num_filters
        - num_classes: Number of output classes
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - use_batch_norm: A boolean variable indicating whether to use batch normalization
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.num_layers = 1 + len(num_filters)

        ############################################################################
        # TODO: Initialize weights and biases for the simple convolutional         #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params.                                                 #
        #                                                                          #
        # IMPORTANT:                                                               #
        # 1. For this assignment, you can assume that the padding                  #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. You need to         #
        # carefully set the `pad` parameter for the convolution.                   #
        #                                                                          #
        # 2. For each convolution layer, we use stride of 2 to do downsampling.    #
        ############################################################################
        C, H, W = input_dim
        F = num_filters
        filter_size = filter_sizes[0]
        stride_conv = 2 
        pad = (filter_size - 1) // 2

        for i in range(len(F)):
            self.params['W' + str(i+1)] = weight_scale * np.random.randn(F[i], C, filter_size, filter_size)
            self.params['b' + str(i+1)] = np.zeros(F[i])
            self.params['gamma' + str(i+1)] = np.ones(F[i])
            self.params['beta' + str(i+1)] = np.zeros(F[i])
            C = F[i]

        self.params['W' + str(len(F)+1)] = weight_scale * np.random.randn(F[-1] * H * W // (4 ** len(F)), num_classes)
        self.params['b' + str(len(F)+1)] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        scores = None
        mode = 'test' if y is None else 'train'
        ############################################################################
        # TODO: Implement the forward pass for the simple convolutional net,       #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################

        W1, b1 = self.params['W1'], self.params['b1']
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']

        if len(self.params) == 2:
          W2, b2 = self.params['W2'], self.params['b2']
          gamma2, beta2 = self.params['gamma2'], self.params['beta2']

        if len(self.params) == 3:
          W2, b2 = self.params['W2'], self.params['b2']
          gamma2, beta2 = self.params['gamma2'], self.params['beta2']
          W3, b3 = self.params['W3'], self.params['b3']

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the simple convolutional net,      #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_naive(x, w, b, conv_param)
    b, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(b)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache
def conv_relu_forward(x, w, b, conv_param):
    a, conv_cache = conv_forward_naive(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache
def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    db, dgamma, dbeta = batchnorm_backward(da, bn_cache)
    dx, dw, db = conv_backward_naive(db, conv_cache)
    return dx, dw, db, dgamma, dbeta
def conv_relu_backward(dout, cache):
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_naive(da, conv_cache)
    return dx, dw, db