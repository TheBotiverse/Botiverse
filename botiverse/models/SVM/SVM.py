#!/usr/bin/env python
# coding: utf-8

# # <font color="cyan">Support Vector Machine </font> From Scratch Implementation
# 
# In this notebook, we shall demonstrate implementing multiclass SVM from scratch using the dual formulation and quadratic programming.

# In[1]:


'''
This module implements and provides an interface for the multiclass Support Vector Machine (SVM) algorithm implemented from scratch.
'''
import numpy as np
from scipy.spatial import distance
import cvxopt
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import copy
import pickle


# ### SVM Class

# In[2]:


class SVM:
    '''
    This class implements the kernelized soft-margin Support Vector Machine algorithm.
    '''
    linear = lambda x, xࠤ , c=0: x @ xࠤ .T
    polynomial = lambda x, xࠤ , Q=5: (1 + x @ xࠤ.T)**Q
    rbf = lambda x, xࠤ , γ=10: np.exp(-γ * distance.cdist(x, xࠤ, 'sqeuclidean'))
    kernel_funs = {'linear': linear, 'polynomial': polynomial, 'rbf': rbf}
    
    def __init__(self, kernel='rbf', C=1, k=2):
        '''
        Initialize an instance of the SVM model.
        
        :param kernel: The kernel function to use. Can be 'linear', 'polynomial', or 'rbf'.
        :type kernel: str
        :param C: The regularization parameter.
        :type C: float
        :param k: The hyperparameter for the polynomial and rbf kernels. Ignored for the linear kernel.
        :type k: int
        '''
        self.kernel_str = kernel
        self.kernel = SVM.kernel_funs[kernel]
        self.C = C
        self.k = k
        self.X, y = None, None
        self.αs = None
        # for multi-class classification
        self.multiclass = False
        self.clfs = []                                  

SVMClass = lambda func: setattr(SVM, func.__name__, func) or func


# #### <font color="white">Solve the Dual Optimization Problem </font>

# We want to find the vector of Lagrange multipliers $\alpha = (\alpha_1, \alpha_2, ..., \alpha_n)$ that maximizes the following dual objective function:
# 
# $$
# \max_\alpha [ \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j (z_iz_j^T) ]
# \quad
# \text{subject to} \quad 
# 
# \sum_{i=1}^{n} \alpha_i y_i = 0, \;  0 \leq \alpha_i \leq C, \; \forall i = 1, 2, ..., n
# $$
# 
# where $z_i = \phi(x_i)$ and $z_j = \phi(x_j)$ are higher dimensional representations of $x_i$ and $x_j$ respectively given by the transform $\phi$.
# 
# <br>
# In matrix form, this becomes
# 
# $$\max_\alpha [ 1^Tα - \frac{1}{2} * α^T (YY^T * K) α ] \quad
# \text{subject to} \quad
# y^T \alpha = 0, \; 0 \leq \alpha_i \leq C, \; \forall i = 1, 2, \ldots, n
# $$
# where $K[i, j] = z_i^T z_j$ and $X$, $y$ are $(N,d)$ and $(N,1)$ dimensional training data matrices respectively.
# <br>
# CVXOPT solves
# 
# $$
# \min_x \frac{1}{2} x^T P x + q^T x \quad
# \text{subject to} \;  A x = b, \;G x ≤ h 
# $$
# 
# Hence, we have
# <br>
# $ x = \alpha $
# <br>
# $ P = YY^T * K $
# <br>
# $ q = -1^T $
# <br><br>
# and for the constraints
# <br>
# $ A = y^T $
# <br>
# $ b = 0 $
# <br>
# $ G = \begin{bmatrix} -I \\ I \end{bmatrix} $
# <br>
# $ h = \begin{bmatrix} 0 \\ C \end{bmatrix} $

# In[3]:


@SVMClass
def fit(self, X, y, eval_train=False):
    '''
    Fit the SVM model to the given data with N training examples and d features.
    
    :param X: The training data arranged as a float numpy array of shape (N, d)
    :type X: numpy.ndarray
    :param y: The training labels arranged as a numpy array of shape (N,) where each element is in {-1, 1} or {0, 1}.
    :type y: numpy.ndarray
    :param eval_train: Whether to print the training accuracy after training is done.
    :type eval_train: bool
    
    :note: This function assume C is not too small; otherwise, it becomes hard to distinguish support vectors.
    '''
    if len(np.unique(y)) > 2:
        self.multiclass = True
        return self.multi_fit(X, y, eval_train)
    
    if set(np.unique(y)) == {0, 1}: y[y == 0] = -1
    self.y = y.reshape(-1, 1).astype(np.double) # Has to be a column vector
    self.X = X
    m = X.shape[0]
    
    # compute the kernel over all possible pairs of (x, x') in the data
    self.K = self.kernel(X, X, self.k)
    
    # For 1/2 x^T P x + q^T x
    P = cvxopt.matrix(self.y @ self.y.T * self.K)
    q = cvxopt.matrix(-np.ones((m, 1)))
    
    # For Ax = b
    A = cvxopt.matrix(self.y.T)
    b = cvxopt.matrix(np.zeros(1))

    # For Gx <= h
    G = cvxopt.matrix(np.vstack((-np.identity(m),
                                 np.identity(m))))
    h = cvxopt.matrix(np.vstack((np.zeros((m,1)),
                                 np.ones((m,1)) * self.C)))

    # Solve    
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    self.αs = np.array(sol["x"])
        
    # Maps into support vectors
    self.is_sv = ((self.αs > 1e-3) & (self.αs <= self.C)).squeeze()
    self.margin_sv = np.argmax((1e-3 < self.αs) & (self.αs < self.C - 1e-2))
    
    if eval_train:  print(f"Model finished training with accuracy {self.evaluate(X, y)}")


# #### Multiclass Case with OVR

# In this, we train $K$ binary classifiers, where $K$ is the number of classes where each classifier perceives one class as +1 and all other classes as -1.

# In[4]:


@SVMClass
def multi_fit(self, X, y, eval_train=False):
    '''
    Fit k classifier for k classes.
    
    :param X: The training data arranged as a float numpy array of shape (N, d)
    :type X: numpy.ndarray
    :param y: The training labels arranged as a numpy array of shape (N,) where each element is an integer between 0 and k-1
    :type y: numpy.ndarray
    :param eval_train: Whether to print the training accuracy after training is done.
    :type eval_train: bool
    '''
    self.k = len(np.unique(y))      # number of classes
    # for each pair of classes
    for i in range(self.k):
        # get the data for the pair
        Xs, Ys = X, copy.copy(y)
        # change the labels to -1 and 1
        Ys[Ys!=i], Ys[Ys==i] = -1, +1
        # fit the classifier
        clf = SVM(kernel=self.kernel_str, C=self.C, k=self.k)
        clf.fit(Xs, Ys)
        # save the classifier
        self.clfs.append(clf)
    if eval_train:  print(f"Model finished training with accuracy {self.evaluate(X, y)}")


# #### SVM Prediction

# Given a new data point $x$, we predict its label $y$ by computing:
# 
# $$g(x) = \sum_{i=1}^{n} \alpha_i y_i k(x_i, x) + b$$
# 
# where $k(x_i, x)$ is the kernel function and $b= y_s - \sum_{i=1}^{n} \alpha_i y_i k(x_i, x_s)$ is the bias term. $s$ is the index of any margin support vector.
# 
# 

# In[5]:


@SVMClass
def predict(self, X_t):
    '''
    Predict the labels for given test data.
    
    :param X_t: The test data arranged as a float numpy array of shape (N, d)
    :return: The predicted labels arranged as a numpy array of shape (N,) where each element is in {-1, 1} or {0, 1}.

    :return: The predicted labels arranged as a numpy array of shape (N,) and the scores for the positive class.   
    :rtype: tuple 
    '''
    if self.multiclass: return self.multi_predict(X_t)
    xₛ, yₛ = self.X[self.margin_sv, np.newaxis], self.y[self.margin_sv]
    αs, y, X= self.αs[self.is_sv], self.y[self.is_sv], self.X[self.is_sv]

    b = yₛ - np.sum(αs * y * self.kernel(X, xₛ, self.k), axis=0)
    score = np.sum(αs * y * self.kernel(X, X_t, self.k), axis=0) + b
    return np.sign(score).astype(int), score


@SVMClass
def evaluate(self, X,y):
    '''
    Compare the predicted labels for given test data y with the actual labels by passing X to model.
    
    :param X: The test data arranged as a float numpy array of shape (N, d)
    :type X: numpy.ndarray
    :param y: The test labels arranged as a numpy array of shape (N,) where each element is an integer between 0 and k-1
    :type y: numpy.ndarray
    
    :return: The accuracy of the model on the given test data.
    :rtype: float
    '''
    outputs, _ = self.predict(X)
    accuracy = np.sum(outputs == y) / len(y)
    
    return round(accuracy, 2)


# #### Multiclass Case with OVR

# Each classifier compute the score of a class against all other classes. The class with the highest score is the predicted class.

# In[6]:


@SVMClass
def multi_predict(self, X):
    '''
    Predict the labels for given test data.
    
    :param X: The test data arranged as a float numpy array of shape (N, d)
    :type X: numpy.ndarray
    
    :return: The predicted labels arranged as a numpy array of shape (N,) where each element is an integer between 0 and k-1 and the corresponding highest scores.
    :rtype: tuple
    '''
    # get the predictions from all classifiers
    preds = np.zeros((X.shape[0], self.k))
    for i, clf in enumerate(self.clfs):
        _, preds[:, i] = clf.predict(X)
    

    # transform the predictions into probabilities using softmax
    preds = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)    
    
    # get the argmax and the corresponding score
    return np.argmax(preds, axis=1), np.max(preds, axis=1)


# ### Example

# In[7]:


# if running from notebook
if __name__ == '__main__':
    get_ipython().system('jupyter nbconvert --to script SVM.ipynb')

