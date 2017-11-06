import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass





  num_test = X.shape[0]
  num_k_classes = W.shape[1]
  #print("the value of the num_test is :",num_test)
  #print("the value of the num_k_classes is :",num_k_classes)

  score = np.dot(X,W)

  score_tmp=np.max(score, axis=1)
  score_max = score_tmp.reshape(num_test, 1)
  tmp_exp=np.exp(score -  score_max)


  prob_tmp = tmp_exp /np.sum(tmp_exp, axis=1, keepdims=True)
  prob=-np.log(prob_tmp)
  #print("the shape of the num_k_classes is :",prob.shape)



  for i in xrange(num_test):
    for j in xrange(num_k_classes):
      if (j == y[i]):
        loss =loss+ prob[i, j]
        dW[:, j] = dW[:, j]+(1-prob_tmp[i, j])*X[i]
      else:
        dW[:, j] =dW[:, j]-prob_tmp[i, j]*X[i]

  loss = loss/num_test
  loss =loss+ 0.5*reg*np.sum((W **2 ))
  dW = -dW/num_test+reg*W




  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #
  # num_train = X.shape[0]
  # num_classes = W.shape[1]
  # f = X.dot(W)  # N by C
  # f_max = np.reshape(np.max(f, axis=1), (num_train, 1))
  # prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True)
  #
  # keepProb = np.zeros_like(prob)
  # keepProb[np.arange(num_train), y] = 1.0
  # loss += -np.sum(keepProb * np.log(prob)) / num_train + 0.5 * reg * np.sum(W * W)
  # dW += -np.dot(X.T, keepProb - prob) / num_train + reg * W


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass


  k_class = W.shape[1]
  num_test = X.shape[0]

  score = np.dot(X, W)
  exp_score = np.exp(score-np.max(score))

  #score_sum = np.sum(score, axis=1)
  # print("  score.shape is :",score.shape)
  # print("  score_sum.shape is :",score_sum.shape)
  # print("  exp_score.shape is :",exp_score.shape)

  score_sum = np.sum(exp_score, axis=1)
  tem_prob = exp_score / score_sum.reshape(num_test, 1)

  pron = np.log(tem_prob)

  similar = np.zeros_like(pron)
  # print ("shape of the similar is :",similar.shape)
  similar[np.arange(num_test), y] = 1
  # print("similar array is :",similar)
  res_ =pron * similar;
  loss=-res_.sum()/num_test + 0.5*reg*np.sum(W*W)




  tmp_dw = similar - tem_prob
  s_dw_V=np.dot(X.T,tmp_dw)
  dW=dW-s_dw_V/num_test+reg*W





  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

