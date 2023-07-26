#!/usr/bin/env python
# coding: utf-8

# # <font color="cyan">Feedforward Neural Network </font> From Scratch Implementation
# In this notebook, we will implement a feedforward neural network with arbitrary structure, loss and activations. You can find a full tutorial on this notebook [here](hhttps://medium.com/towards-data-science/backpropagation-the-natural-proof-946c5abf63b1).

# In[1]:


'''
This module implements and provides an interface for an arbitrary neural network architecture.
'''
try:
    from botiverse.models.NN.utils import split_data, batchify
except:
    pass
# check if running on notebook
from tqdm import tqdm
import numpy as np
import copy 
import pickle


# ### Define the Architecture

# In[2]:


class NeuralNet():
    '''
    Defines the hypothesis set and learning algorithm for a neural network.
    '''
    def __init__(self, structure, activation='sigmoid', optimizer='ADAM'):
        '''
        Initialize the hyperparameters, weights and biases for the network. 
        
        :param structure: A list of integers representing the number of neurons in each layer. For example, [2, 3, 1] is a network with 2 neurons in the input layer, 3 in the hidden layer, and 1 in the output layer.
        :type structure: list
        :param activation: The activation function to use. Can be 'sigmoid' or 'relu'.
        :type activation: str
        :param optimizer: The optimizer to use. Can be 'ADAM' or 'SGD'.
        :type optimizer: str
        '''
        ### Hyperparameters
        self.structure = structure
        self.num_layers = len(structure) 
        self.activation = activation
        self.optimizer = optimizer
        if activation == 'sigmoid':
            def σ(z): return 1.0/(1.0+np.exp(-z))                   # activation function
            def σࠤ(z): return σ(z)*(1-σ(z))                         # derivative of the activation function
     
        elif activation == 'relu':
            def σ(z): return np.maximum(0, z)
            def σࠤ(z): return np.greater(z, 0).astype(int)
        self.h = σ
        self.dh = σࠤ   
        
        def softmax(z): return np.exp(z-np.max(z))/np.sum(np.exp(z-np.max(z)), axis=0)
        self.g = softmax
        
        ### Parameters
        # Xaiver initialization
        #for each layer except the first y neurons randomize a (y, 1) vector for the bias vector:
        self.Bₙ = [np.random.randn(l, 1) * np.sqrt(2/l) for l in structure[1:]]
        #for each two consecutive layers with x and y neurons respectively randomize a (y,x) matrix for the weight matrix:
        self.Wₙ = [np.random.randn(l, next_l) * np.sqrt(2/l) for l, next_l in zip(structure[:-1], structure[1:])]
        
        ### Loss
        def cross_entropy(aᴺ, y): return -np.sum(y * np.log(aᴺ)) / aᴺ.shape[1]
        def cross_entropyࠤ(aᴺ, y): return (aᴺ-y)
        self.J = cross_entropy
        self.ᐁJ = cross_entropyࠤ

        ### Misc
        self.optimal_epochs = None
NNClass = lambda func: setattr(NeuralNet, func.__name__, func) or func


# ### Backpropagation

# In[3]:


@NNClass
def backprop(self, xₛ , yₛ ):
    '''
    Compute the loss gradients მJⳆმBₙₛ, მJⳆმWₙₛ given an observation (xₛ , yₛ ) where xₛ and yₛ are column vectors. 
    
    :param xₛ: The input vector of shape (d, 1).
    :type xₛ: numpy.ndarray
    :param yₛ: The output vector of shape (K, 1).
    :type yₛ: numpy.ndarray
    
    :return: A tuple of lists (მJⳆმBₙₛ, მJⳆმWₙₛ) where მJⳆმBₙₛ is a list of the gradients of the loss function with respect to the biases and მJⳆმWₙₛ is a list of the gradients of the loss function with respect to the weights.
    :rtype: tuple of lists
    '''
    h, dh, ᐁJ, g = self.h, self.dh, self.ᐁJ, self.g

    მJⳆმBₙₛ = [np.zeros(b.shape) for b in self.Bₙ]
    მJⳆმWₙₛ = [np.zeros(W.shape) for W in self.Wₙ]

    # forward pass (computing z for all layers)
    Zₙ = []                     # list to store all the z vectors, layer by layer
    Aₙ = []                     # list to store all the a vectors layer by layer

    for i, (b, W) in enumerate(zip(self.Bₙ, self.Wₙ)):
        z = W.T @ a + b if Zₙ else W.T @ xₛ  + b
        a = h(z) if i != self.num_layers-2 else g(z)
        Zₙ.append(z)
        Aₙ.append(a)

    #Zₙ and Aₙ are now ready.

    # backward pass (computing δ and consequently მJⳆმBₙₛ and მJⳆმWₙₛ layer by layer )
    H = self.num_layers-2
    for L in range(H, -1, -1):
        δ =  dh(Zₙ[L]) * (self.Wₙ[L+1] @ δ) if L != H else ᐁJ(Aₙ[L], yₛ ) 
        მJⳆმBₙₛ[L] = δ
        მJⳆმWₙₛ[L] = Aₙ[L-1] @ δ.T  if L != 0 else xₛ  @ δ.T
    
    return (მJⳆმBₙₛ, მJⳆმWₙₛ)


# ### Gradient Descent

# In[4]:


@NNClass
def SGD(self, x_batch, y_batch, λ, α=0.01):
    '''
    Given a minibatch (a list/numpy array of tuples (xₛ, yₛ )) this will update Bₙ and Wₙ by applying SGD with L2 regularization.
    
    :param x_batch: A list/numpy array of input vectors of shape (d, 1).
    :type x_batch: numpy.ndarray or list
    :param y_batch: A list/numpy array of output vectors of shape (K, 1).
    :type y_batch: numpy.ndarray or list
    :param λ: The learning rate.
    :type λ: float
    :param α: The regularization parameter.
    :type α: float
    '''
    მJⳆმBₙ = [np.zeros(b.shape) for b in self.Bₙ]
    მJⳆმWₙ = [np.zeros(W.shape) for W in self.Wₙ]

    for x, y in zip(x_batch, y_batch):
        მJⳆმBₙₛ, მJⳆმWₙₛ = self.backprop(x, y)
        მJⳆმBₙ = [მJⳆმb + მJⳆმbₛ for მJⳆმb, მJⳆმbₛ in zip(მJⳆმBₙ, მJⳆმBₙₛ)]  
        მJⳆმWₙ = [მJⳆმW + მJⳆმWₛ for მJⳆმW, მJⳆმWₛ in zip(მJⳆმWₙ, მJⳆმWₙₛ)]

    d = len(x_batch)
    self.Wₙ = [(1 - λ * α / d) * W - λ / d * მJⳆმW / np.linalg.norm(მJⳆმW) for W, მJⳆმW in zip(self.Wₙ, მJⳆმWₙ)]
    self.Bₙ = [(1 - λ * α / d) * b - λ / d * მJⳆმb / np.linalg.norm(მJⳆმb) for b, მJⳆმb in zip(self.Bₙ, მJⳆმBₙ)]


# In[5]:


@NNClass
def ADAM(self, x_batch, y_batch, λ, α=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    '''
    Given a minibatch (a list of tuples (xₛ, yₛ )) this will update Bₙ and Wₙ by applying Adam optimizer with L2 regularization.
    
    :param x_batch: A list/numpy array of input vectors of shape (d, 1).
    :type x_batch: numpy.ndarray or list
    :param y_batch: A list/numpy array of output vectors of shape (K, 1).
    :type y_batch: numpy.ndarray or list
    :param λ: The learning rate.
    :type λ: float
    :param α: The regularization parameter.
    :type α: float
    :param beta1: The exponential decay rate for the first moment estimates.
    :type beta1: float
    :param beta2: The exponential decay rate for the second-moment estimates.
    :type beta2: float
    :param epsilon: A small constant for numerical stability.
    :type epsilon: float
    '''
    mB = [np.zeros(b.shape) for b in self.Bₙ]           # momentum for Bₙ
    vB = [np.zeros(b.shape) for b in self.Bₙ]           # RMSprop for Bₙ
    mW = [np.zeros(W.shape) for W in self.Wₙ]           # momentum for Wₙ
    vW = [np.zeros(W.shape) for W in self.Wₙ]           # RMSprop for Wₙ

    for x, y in zip(x_batch, y_batch):
        mJdB, mJdW = self.backprop(x, y)
        # update the momentum and RMSprop for Wₙ and Bₙ
        mB = [beta1 * mb + (1 - beta1) * mJdb for mb, mJdb in zip(mB, mJdB)]
        vB = [beta2 * vb + (1 - beta2) * (mJdb ** 2) for vb, mJdb in zip(vB, mJdB)]
        mW = [beta1 * mw + (1 - beta1) * mJdw for mw, mJdw in zip(mW, mJdW)]
        vW = [beta2 * vw + (1 - beta2) * (mJdw ** 2) for vw, mJdw in zip(vW, mJdW)]

    # update the parameters Wₙ and Bₙ
    d = len(x_batch)
    self.Wₙ = [(1 - α * λ / d) * W - λ * mb / (np.sqrt(vb) + epsilon) for W, mb, vb in zip(self.Wₙ, mW, vW)]
    self.Bₙ = [(1 - α * λ / d) * b - λ * mb / (np.sqrt(vb) + epsilon) for b, mb, vb in zip(self.Bₙ, mB, vB)]


# ### Feedforward

# In[6]:


@NNClass
def feedforward(self, x ):
    h, g = self.h, self.g
    '''
    The forward pass of the network. Given an input x this will return the output of the network.
    
    :param x: The input vector of shape (d, 1) where d is the number of input features.
    :type x: numpy.ndarray
    
    :return: The output vector of shape (K, 1) where K is the number of classes.
    :rtype: numpy.ndarray
    '''
    a = x
    for i, (b, W) in enumerate(zip(self.Bₙ, self.Wₙ)):
        z = W.T @ a + b
        a = h(z) if i != self.num_layers-2 else g(z)
    ŷ = a
    return ŷ 


# ### Training Method

# In[7]:


@NNClass
def fit(self, X, y, batch_size=32, epochs=200, λ=30, α=0.01, optimizer='ADAM', val_split=0.0, patience=50, eval_train=False):
    '''
    For each epoch, go over each minibatch and perform a gradient descent update accordingly and evaluate the model on the training and validation sets if needed.
    
    :param X: The training data arranged as a float numpy array of shape (N, d)
    :type X: numpy.ndarray
    :param y: The training labels arranged as a numpy array of shape (N,) where each element is an integer between 0 and k-1
    :type y: numpy.ndarray
    :param batch_size: The size of the minibatches to use.
    :type batch_size: int
    :param epochs: The number of epochs to run.
    :type epochs: int
    :param λ: The learning rate.
    :type λ: float
    :param α: The regularization parameter.
    :type α: float
    :param optimizer: The optimizer to use. Can be 'ADAM' or 'SGD'.
    :type optimizer: str
    :param val_split: The ratio of the validation set to the training set. If given, per-epoch validation will be performed.
    :type val_split: float
    :param eval_train: Whether to evaluate the model on the training set.
    :type eval_train: bool
    
    '''
    if val_split:
        X, y, X_v, y_v = split_data(X, y, val_split)
    
    Xc, yc = copy.deepcopy(X), copy.deepcopy(y)
    X, y = batchify(X, y, batch_size)

    if self.optimal_epochs is not None:
        print(f"Using optimal epochs: {self.optimal_epochs} found from an earlier run.")
        epochs = self.optimal_epochs
        
    # training loop
    self.gradient_descent = self.SGD if optimizer == 'SGD' else self.ADAM
    train_acc, val_acc = '', ''
    pbar = tqdm(range(epochs))
    best_loss = np.inf
    val_loss = 0
    bad_epochs = 0
    for epoch in pbar:
        # randomly shuffle X, y
        p = np.random.permutation(len(X))
        X, y = X[p], y[p]
        for x_batch, y_batch in zip(X, y):
            self.gradient_descent(x_batch, y_batch, λ)          #update the parameters after learning from the mini_batch.
        if eval_train:    
            train_acc = self.evaluate(Xc, yc)
    
        # update bar
        desc1, desc2 = f"Train Acc: {train_acc}" if eval_train else '', f" | Val Acc: {val_acc}" if val_split else ''
        desc3 = f" | Val Loss: {val_loss}" if val_split else ''
        desc =  desc1 + desc2 + desc3
        pbar.set_description(desc)
        
        # early stopping
        if val_split:     
            val_loss = self.evaluate(X_v, y_v, loss=True)
            if val_loss <= best_loss:
                best_loss = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs == patience:
                    print(f"{patience} epochs have passed without improvement. Early stopping... \n")
                    self.optimal_epochs = epoch - patience + 2
                    self.optimal_epochs = int(self.optimal_epochs * (1 + val_split))
                    break
            val_acc = self.evaluate(X_v, y_v)        


# In[8]:


@NNClass
def predict(self, X):
    '''
    Predict the class of each example in X.
    
    :param X: The test data arranged as a float numpy array of shape (N, d)
    :type X: numpy.ndarray
    '''
    if len(X.shape) == 2:   X = X[..., np.newaxis]
    # return the predicted class and the probability of the predicted class
    return np.array([np.argmax(self.feedforward(x)) for x in X]), np.array([np.max(self.feedforward(x)) for x in X])
        
@NNClass
def evaluate(self, X,y, loss=False):
    '''
    Compare the one-hot vector y with the networks output yᴺ and calculate the accuracy.
    
    :param X: The test data arranged as a float numpy array of shape (N, d)
    :type X: numpy.ndarray
    :param y: The test labels arranged as a numpy array of shape (N,) where each element is an integer between 0 and k-1
    :type y: numpy.ndarray
    '''
    if len(X.shape) == 2:   X = X[..., np.newaxis]
    if loss:
        return np.sum([self.J(self.feedforward(x), y) for x, y in zip(X, y)]) / len(X)
    else:
        validation_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in zip(X, y)]   #the index is the number itself
        accuracy = sum(int(ŷ == y) for (ŷ, y) in validation_results) / len(X)
    
    return round(accuracy, 2)


# #### Saving and Loading Model

# In[9]:


@NNClass
def save(self, path):
    '''
    Save the model to the given path.
    
    :param path: The path to save the model to.
    :type path: str
    '''
    # store self.Bₙ, self.Wₙ, self.structure, self.activation, self.optimizer
    with open(path, 'wb') as f:
        # save self.Bₙ, self.Wₙ, self.structure, self.activation, self.optimizer
        class_dict = {key: value for key, value in self.__dict__.items() if key in ['Bn', 'Wn', 'structure', 'activation', 'optimizer']}
        f.write(pickle.dumps(class_dict))

@staticmethod        
@NNClass
def load(path):
    '''
    Load the model from the given path.
    
    :param path: The path to load the model from.
    :type path: str
    '''
    with open(path, 'rb') as f:
        class_dict = pickle.loads(f.read())
        model = NeuralNet(class_dict['structure'], class_dict['activation'], class_dict['optimizer'])
        # now use setattr to set Bₙ and Wₙ
        for key, value in class_dict.items():
            if key in ['Bn', 'Wn']:
                setattr(model, key, value)
        return model


# In[10]:


# if running from notebook
if __name__ == '__main__':
    get_ipython().system('jupyter nbconvert --to script NN.ipynb')

