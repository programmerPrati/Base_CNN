import numpy as np

from cs6353.layers import *
from cs6353.fast_layers import *
from cs6353.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim

    # Initialize weights and biases for the convolutional layer
    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)

    # Calculate dimensions after pooling
    pool_output_H = H // 2
    pool_output_W = W // 2

    # Initialize weights and biases for the first affine layer
    self.params['W2'] = weight_scale * np.random.randn(num_filters * pool_output_H * pool_output_W, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)

    # Initialize weights and biases for the second affine layer
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)


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
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': int((filter_size - 1) / 2)}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################

    # Conv-relu-pool
    conv_out, conv_cache = conv_forward_naive(X, W1, b1, conv_param)
    relu_out, relu_cache = relu_forward(conv_out)
    pool_out, pool_cache = max_pool_forward_naive(relu_out, pool_param)

    # Flatten and affine-relu
    flat_pool_out = pool_out.reshape(X.shape[0], -1)  # Flatten the output
    affine_relu_out, affine_relu_cache = affine_relu_forward(flat_pool_out, W2, b2)

    # Final affine layer
    scores, affine_cache = affine_forward(affine_relu_out, W3, b3)


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################

    # Compute the loss and initialize gradient dictionary
    loss, grads = 0, {}

    # Softmax loss and regularization
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    loss = data_loss + reg_loss

    # Backward pass
    # Output layer
    daffine_relu, dW3, db3 = affine_backward(dscores, affine_cache)
    dW3 += self.reg * W3  # Regularization gradient
    grads['W3'], grads['b3'] = dW3, db3

    # Hidden affine-relu layer
    dflat_pool_out, dW2, db2 = affine_relu_backward(daffine_relu, affine_relu_cache)
    dW2 += self.reg * W2  # Regularization gradient
    grads['W2'], grads['b2'] = dW2, db2

    # Unflatten and conv-relu-pool backward
    dpool_out = dflat_pool_out.reshape(pool_out.shape)
    drelu_out = max_pool_backward_naive(dpool_out, pool_cache)
    dconv_out = relu_backward(drelu_out, relu_cache)
    dx, dW1, db1 = conv_backward_naive(dconv_out, conv_cache)
    dW1 += self.reg * W1  # Regularization gradient
    grads['W1'], grads['b1'] = dW1, db1

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads