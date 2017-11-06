import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  #print("shape of dW is :",dW.shape)
  #print("shape of X[i] is :",X[1].shape)

  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin=scores[j]-correct_class_score+1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, y[i]] += -X[i]
        dW[:, j] += X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW=dW/num_train
  # Add regularization to the loss.
  loss += reg*np.sum(W * W)
  dW=dW+reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #pass


  # print("shape of w is :",W.shape)
  # print("shape of X is ",X.shape)
  # print("shape of y is :",y.shape)
  num_classes = W.shape[1]
  num_train = X.shape[0]


  tem_score=np.dot(X,W)
  #print("shape of the tem_score is :",tem_score.shape)

  tem_score_y_i=tem_score[np.arange(num_train), y]
  #print("shape of the tem_score_y_i is :",tem_score_y_i.shape)
  tem_score_y_=tem_score_y_i.reshape(X.shape[0],1)
  margin=tem_score-tem_score_y_+1#calculate the margin,margin is very important for the loss.
  #print("shape of the score_m is :",margin.shape)

  margin[np.arange(num_train), y] = 0.0
  margin[margin <= 0] = 0.0
  loss += np.sum(margin) / num_train
  loss += 0.5*reg*np.sum(W*W)





  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #pass

  margin[margin > 0] = 1.0
  #print("shape of margin is : margin",margin.shape)

  num_error_classification = np.sum(margin, axis=1)#按行求和 对每一行都计算错误分类的个数

  #print("shape of the  row_sum is :",num_error_classification.shape)
  margin[np.arange(num_train), y] = -num_error_classification#在y_i 的位置上面进行赋值

  dW += np.dot(X.T, margin)
  dW=dW/num_train + reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
