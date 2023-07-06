#!/usr/bin/env python
# coding: utf-8

# # <font color="cyan">LSTM </font> From Scratch Implementation
# 
# In this notebook, we shall demonstrate implementing LSTM from scratch; up from linear layers.

# ### <font color='white'>LSTM Layer</font>
# 
# An LSTM layer takes in the input $x_t$ and the previous hidden and cell state $h_{t-1}$, $c_{t-1}$. It outputs the next hidden and cell state $h_t$, $c_t$ in the following fashion:
# 
# Define three gates and pass the input, previous hidden and cell state through them to get the outputs of the gates:
# 
# $$i_t = σ(W_{xi}x_t + W_{hi}h_{t-1} + b_i)$$
# 
# $$f_t = σ(W_{xf}x_t + W_{hf}h_{t-1} +  b_f)$$
# 
# $$o_t = σ(W_{xo}x_t + W_{ho}h_{t-1} + b_o)$$
# 
# Compute a candidate cell state and cell stage using another linear layer and input and forget gates:
# $$\hat{c} = tanh(W_{xg}x_t + W_{hg}h_{t-1} + b_g)$$
# 
# $$c_t = f_tc_{t-1} + i_tg_t$$
# 
# Compute the next hidden state using the computed cell state and output gate:
# $$h_t = o_ttanh(c_t)$$
# 
# 

# In[2]:


import torch
import torch.nn as nn
from torch import sigmoid, tanh as σ, tanh
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

class LSTMCell(nn.Module): 
    '''
    An interface for a single LSTM layer.
    '''
    def __init__(self, input_size, hidden_size):
        '''
        Initialize the parameters single LSTM layer given the size of the inputs and the required hidden size.
        :param input_size: The size of the input to the LSTM layer
        :param hidden_size: The size of the hidden state of the LSTM layer
        '''
        super(LSTMCell, self).__init__()
        # input and output dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # LSTM involves three gates: input, forget, output and tanh linear layer.
        # We can declare them as follows and chunk them later since their two weights are of same size.
        self.Wₓₕ = nn.Linear(input_size, hidden_size * 4)
        self.Wₕₕ = nn.Linear(hidden_size, hidden_size * 4)
        
        # initalize the weights using Xavier initialization
        σ = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():     w.data.uniform_(-σ, σ)

    def forward(self, input, h, c):
        '''
        Forward pass of the LSTM layer which passes in the input and previous states and returns the new hidden and cell states
        :param input: The input to the LSTM layer which is of shape (batch_size, input_size)
        :param h: The previous hidden state of the LSTM layer which is of shape (batch_size, hidden_size)
        :param c: The previous cell state of the LSTM layer which is of shape (batch_size, hidden_size)
        :return: The new hidden state and cell state of the LSTM layer which are of shapes (batch_size, hidden_size) each
        '''
        # chunk the weights and biases of the gates
        weights = self.Wₓₕ(input) + self.Wₕₕ(h)
        Γi, Γf, T, Γo = weights.chunk(4, 1)

        # Get gates (iₜ, fₜ,  oₜ)
        iₜ, fₜ, oₜ = σ(Γi), σ(Γf), σ(Γo)
        
        # compute candidate and new cell state
        ĉ = tanh(T)
        cₜ  = c * fₜ + iₜ * ĉ

        # compute new hidden state
        hₜ = oₜ * tanh(cₜ)

        return (hₜ, cₜ)


# ### LSTM Class
# 
# Given an input sequence, each token passes by all the layers and each layer has its own hidden state and cell state which is its output due to the previous token.

# In[3]:


class LSTMX(nn.Module):
    '''
    An interface for a multi-layer LSTM where the hidden state of each layer is of the same size.
    '''
    def __init__(self, input_size, hidden_size, num_layers=1):
        '''
        Initialize the parameters of an LSTM with an arbitrary number of layers all of which have the same hidden size.
        :param input_size: The size of the input to the LSTM layer
        :param hidden_size: The size of the hidden state of the LSTM layer
        :param num_layers: The number of LSTM layers stacked on top of each other (default: 1)
        '''
        super(LSTMX, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # initialize num_layers of LSTM layers
        self.lstm_layers = nn.ModuleList([LSTMCell(input_size, hidden_size)])
        for l in range(1, self.num_layers):
            self.lstm_layers.append(LSTMCell(self.hidden_size, self.hidden_size))


    def forward(self, input, hₒ=None):
        '''
        Forward pass of the LSTM which takes the input and initial hidden/cell state and returns the output
        due to the last token in the sequence.
        :param input: The input to the LSTM layer which is of shape (batch_size, seq_len, input_size)
        :param hₒ: The initial hidden and cell states of the LSTM layer which are of shape (num_layers, batch_size, hidden_size)
        :return: The output due to the last token in the sequence which is of shape (batch_size, hidden_size)
        '''
        # Input of shape (batch_size, seqence length , input_size)
        #
        # Output of shape (batch_size, output_size)
        batch_size, seq_len = input.size(0), input.size(1)
        
        if hₒ is None:
             v = Variable(input.new_zeros(self.num_layers, batch_size, self.hidden_size))
             hₒ = v.cuda() if torch.cuda.is_available() and hₒ else v
        
        # Will contain the output of the last layer for each token in the sequence
        outs = []

        H = [hₒ[l, :, :] for l in range(self.num_layers)]
        C = [hₒ[l, :, :] for l in range(self.num_layers)]
        # Now H[l], C[l] are the current hidden state for layer l (defined for every sequence in the batch)
        
        # for each token in the sequence
        for t in range(seq_len):
            # pass it through each layer
            token = input[:, t, :]
            for l in range(self.num_layers):
                lstm = self.lstm_layers[l]   # takes input (batch_size, input_size) and h, c of shape (batch_size, hidden_size)
                # layer takes the token or output from the previous layer and initial hidden, cell states
                H[l], C[l] = lstm(token, H[l], C[l]) if l == 0 else lstm(H[l - 1], H[l], C[l])

            # the output due to any token is that due to the last layer of the lstm
            outs.append(H[l])

        return outs[-1]


# In[4]:


class LSTMClassifier(nn.Module):
    '''
    LSTMClassifier class defines a high-level interface of the LSTM model for classification.
    '''
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = LSTMX(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.lstm(x)
        out = self.fc(out)
        return out
    
    def fit(self, X, y, hidden_size=64, λ=0.001, num_epochs=100, val_size=0.0):
        Xt = torch.from_numpy(X)
        yt = torch.from_numpy(y)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=λ)
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            outputs = self(Xt)
            loss = self.criterion(outputs.squeeze(), yt)
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, X):
        Xt = torch.from_numpy(X)
        outputs = self(Xt)
        outputs = torch.argmax(outputs, dim=1)
        return outputs.detach().numpy()
    
    def evaluate(self, X, y):
        Xt = torch.from_numpy(X)
        yt = torch.from_numpy(y)
        outputs = self(Xt)
        outputs = torch.argmax(outputs, dim=1)
        # compute the accuracy
        return (outputs == yt).sum().item() / len(yt)


# In[5]:


# if running from notebook
if __name__ == '__main__':
    get_ipython().system('jupyter nbconvert --to script LSTM.ipynb')

