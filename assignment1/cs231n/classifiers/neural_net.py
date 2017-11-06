from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    #print("the size of the W1 is :", input_sizehidden_size)

    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    #pass
    #
    # print("shape of W1,",W1.shape)
    # print("shape of X,",X.shape)
    # print("shape of b2,",b2.shape)
    # print(" b2 is :",b2)

    tem_b1=b1.reshape(b1.shape[0],1)
    #print("shape of b1,",tem_b1.shape)

    sc_=np.dot(W1.T, X.T)
    #print("shape of sc_,",sc_.shape)

    tem1=sc_+tem_b1
   # print("shape of tem1,",tem1.shape)
    tem1[tem1<=0]=0
    h1=tem1
   # print("shape of h1,",h1.shape)


    #print("shape of W2,",W2.shape)
    scores=np.dot(W2.T,h1).T+b2
   # print("shape of h2,",scores.shape)




    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    #pass
    #
    score_tmp = np.max(scores, axis=1)

    score_max = score_tmp.reshape(N, 1)
    exp_score = np.exp(scores - score_max)


    #exp_score = np.exp(scores)

    # score_sum = np.sum(score, axis=1)
    # print("  score.shape is :",score.shape)
    # print("  score_sum.shape is :",score_sum.shape)
    # print("  exp_score.shape is :",exp_score.shape)

    score_sum = np.sum(exp_score, axis=1)
    tem_prob = exp_score / score_sum.reshape(N, 1)
    pron = np.log(tem_prob)
    similar_array = np.zeros_like(pron)
    # print ("shape of the similar_array is :",similar_array.shape)
    similar_array[np.arange(N), y] = 1
    # print("similar array is :",similar_array)
    res_ = pron * similar_array;
    loss = -res_.sum() / N + 0.5*reg*(np.sum(W1*W1)+np.sum(W2*W2))


    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
   # pass

    bach_scores=tem_prob
    bach_scores[np.arange(N),y]-=1
    bach_scores/=N
    dw2=h1.dot(bach_scores)
    grads['W2']=dw2+reg*W2
    #a=np.sum(bach_scores, axis=0)
   # print("shape of a is :",a.shape)
    grads['b2'] = np.sum(bach_scores, axis=0)
    d_hid = np.dot(bach_scores, W2.T)
    # backprop the ReLU non-linearity
    d_hid[h1.T <= 0] = 0#因为我们采用的是relu 的激活函数所以写起来这里要注意，小于0 的位置要写为0;
    # finally into W,b
    grads['W1'] = np.dot(X.T, d_hid)+reg * W1
    grads['b1'] = np.sum(d_hid, axis=0)




    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      ## pass

      _batch_idx = np.random.choice(num_train, batch_size, replace=True)

      X_batch = X[_batch_idx, :]  # here we just select batch_size of the training set.
      y_batch = y[_batch_idx]



      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      #pass
      self.params['W2']= self.params['W2']-learning_rate* grads['W2']#here we try to update the parameters .
      self.params['b2']= self.params['b2']-learning_rate* grads['b2']
      self.params['W1']= self.params['W1']-learning_rate* grads['W1']
      self.params['b1']= self.params['b1']-learning_rate* grads['b1']
      #
      # W1 = W1 - learning_rate * grads['W1']
      # b1 = W1 - learning_rate * grads['b1']

      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################
      #
      # if verbose and it % 100 == 0:
      #   print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    #pass

    y1_out = X.dot(self.params['W1']) + self.params['b1']
    h1 = np.maximum(0, y1_out)
    scores = h1.dot(self.params['W2']) + self.params['b2']
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


