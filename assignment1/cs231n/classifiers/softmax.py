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
  num_train = X.shape[0]
  num_classes = W.shape[1]
    
  for i in xrange(num_train):
    all_scores = np.zeros((num_classes,))
    for j in xrange(num_classes):
        all_scores[j] = np.dot(X[i],W[:,j])
        
    all_scores -= np.max(all_scores)
    P_yi = np.exp(all_scores[y[i]])/np.sum(np.exp(all_scores))
    loss += -np.log(P_yi)
    
    dW[:,y[i]] -= (1-P_yi)*X[i]
    for j in xrange(num_classes):
        if(j!=y[i]):
            P_j = np.exp(all_scores[j])/np.sum(np.exp(all_scores))
            dW[:,j] -= -P_j*X[i]

  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += reg * 2 * W
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = np.dot(X,W)
  max_scores = np.max(scores,axis=1)
  scores -= np.reshape(max_scores,(num_train,1))
  scores = np.exp(scores)
  scores_sum = np.sum(scores,axis=1)
  P_y = scores[np.arange(num_train),y]/scores_sum
  loss = -np.sum(np.log(P_y))/num_train + (reg*np.sum(W*W))
    
  P = -scores/np.reshape(scores_sum,(num_train,1)) 
  P[np.arange(num_train),y] += 1
  dW = -np.dot(X.T,P)/num_train + (reg*2*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

