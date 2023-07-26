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

# In[1]:


import torch
import torch.nn as nn
from torch import sigmoid, tanh as σ, tanh
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import os

class LSTMCell(nn.Module): 
    '''
    An interface for a single LSTM layer.
    '''
    def __init__(self, input_size, hidden_size):
        '''
        Initialize the parameters single LSTM layer given the size of the inputs and the required hidden size.
        
        :param input_size: The size of the input to the LSTM layer
        :type input_size: int
        :param hidden_size: The size of the hidden state of the LSTM layer
        :type hidden_size: int
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
        :type input: torch.Tensor
        :param h: The previous hidden state of the LSTM layer which is of shape (batch_size, hidden_size)
        :type h: torch.Tensor
        :param c: The previous cell state of the LSTM layer which is of shape (batch_size, hidden_size)
        :type c: torch.Tensor
        
        :return: The new hidden state and cell state of the LSTM layer which are of shapes (batch_size, hidden_size) each
        :rtype: tuple
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

# In[2]:


class LSTMX(nn.Module):
    '''
    An interface for a multi-layer LSTM where the hidden state of each layer is of the same size.
    '''
    def __init__(self, input_size, hidden_size, num_layers=1):
        '''
        Initialize the parameters of an LSTM with an arbitrary number of layers all of which have the same hidden size.
        
        :param input_size: The size of the input to the LSTM layer
        :type input_size: int
        :param hidden_size: The size of the hidden state of the LSTM layer
        :type hidden_size: int
        :param num_layers: The number of LSTM layers stacked on top of each other (default: 1)
        :type num_layers: int
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
        :type input: torch.Tensor
        :param hₒ: The initial hidden and cell states of the LSTM layer which are of shape (num_layers, batch_size, hidden_size)
        :type hₒ: torch.Tensor
        
        :return: The output due to the last token in the sequence which is of shape (batch_size, hidden_size)
        :rtype: torch.Tensor
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


# In[3]:


class LSTMClassifier(nn.Module):
    '''
    LSTMClassifier class defines a high-level interface of the LSTM model for classification.
    '''
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initialize the LSTMClassifier with the given parameters.
        
        :param input_size: The size of the input to the LSTM layer
        :type input_size: int
        :param hidden_size: The size of the hidden state of the LSTM layer
        :type hidden_size: int
        :param num_classes: The number of classes to classify the input into
        :type num_classes: int
        """
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = LSTMX(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        '''
        Forward pass of the LSTMClassifier which takes the input and passes it through all the LSTM layers and an output layer to produce an output.
        
        :param x: The input to the LSTMClassifier which is of shape (batch_size, seq_len, input_size)
        :type x: torch.Tensor
        
        :return: The output of the LSTMClassifier which is of shape (batch_size, num_classes)
        :rtype: torch.Tensor
        '''
        out = self.lstm(x)
        out = self.fc(out)
        return out
    
    
    def fit(self, X, y, λ=0.001, α=1e-3, max_epochs=100, patience=5, val_ratio=0.2):
        '''
        Fit the LSTMClassifier to the given data.
        
        :param X: The input data of shape (batch_size, seq_len, input_size)
        :type X: numpy.ndarray
        :param y: The labels of the data of shape (batch_size)
        :type y: numpy.ndarray
        :param hidden_size: The size of the hidden state of the LSTM layer (default: 64)
        :type hidden_size: int
        :param λ: The learning rate (default: 0.001)
        :type λ: float
        :param num_epochs: The number of epochs to train the model for (default: 100)
        :type num_epochs: int
        '''
        Xt = torch.from_numpy(X)
        yt = torch.from_numpy(y)
        if val_ratio:
            indices = torch.randperm(len(Xt))
            Xt, yt = Xt[indices], yt[indices]
            # split the data into train and validation sets
            val_size = int(val_ratio * len(Xt))
            Xt, Xv = Xt[:-val_size], Xt[-val_size:]
            yt, yv = yt[:-val_size], yt[-val_size:]
            self.Xv, self.yv = Xv, yv
                
        optimizer = torch.optim.Adam(self.parameters(), lr=λ, weight_decay=α)
        print("Training the LSTMClassifier...")
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        bad_epochs = 0
        val_accuracy = 0
        val_loss = 0
        best_loss = np.inf
        pbar = tqdm(range(max_epochs))
        for epoch in pbar:
            outputs = self(Xt)
            loss = self.criterion(outputs.squeeze(), yt)
            pbar.set_description(f"Epoch {epoch+1}/{max_epochs}, Loss: {loss.item()}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if val_ratio:
                # randomly shuffle the data
                val_accuracy = self.evaluate(Xv, yv)
                with torch.no_grad():
                    val_loss = self.criterion(self(Xv).squeeze(), yv)
                if val_loss < best_loss:
                    best_loss = val_loss
                    bad_epochs = 0
                    # save the model
                    torch.save(self.state_dict(), os.path.join(curr_dir, "LSTMClassifier.pt"))
                else:
                    bad_epochs += 1
                    if bad_epochs == patience:
                        print(f"{patience} epochs have passed without improvement. Early stopping...")
                        self.load_state_dict(torch.load(os.path.join(curr_dir, "LSTMClassifier.pt")))
                        break
                # every 5 epochs see
                pbar.set_postfix({"Validation Accuracy": val_accuracy})             
           

    def predict(self, X):
        '''
        Predict the labels of the given data by passing it through the LSTMClassifier.
        
        :param X: The input data of shape (batch_size, seq_len, input_size)
        :type X: numpy.ndarray
        
        :return: The predicted labels of the data of shape (batch_size)
        :rtype: numpy.ndarray
        '''
        Xt = torch.from_numpy(X)
        outputs = self(Xt)
        pred = torch.argmax(outputs, dim=1)
        softmax = nn.Softmax(dim=1)
        prob = torch.max(softmax(outputs), dim=1)
        return pred.detach().numpy(), prob.values.detach().numpy()
    
    def evaluate(self, Xt, yt):
        '''
        Evaluate the LSTMClassifier on the given data.
        
        :param X: The input data of shape (batch_size, seq_len, input_size)
        :type X: numpy.ndarray
        :param y: The labels of the data of shape (batch_size)
        :type y: numpy.ndarray
        
        :return: The accuracy of the LSTMClassifier on the given data
        :rtype: float
        '''
        # check ig they are torch tensors
        if not isinstance(Xt, torch.Tensor) or not isinstance(yt, torch.Tensor):
            Xt = torch.from_numpy(Xt)
            yt = torch.from_numpy(yt)
        outputs = self(Xt)
        outputs = torch.argmax(outputs, dim=1)
        # compute the accuracy
        return (outputs == yt).sum().item() / len(yt)
    
    def save(self, path):
        '''
        Save the LSTMClassifier to a file.
        
        :param path: The path to the file
        :type path: str
        '''
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        '''
        Load the LSTMClassifier from a file.
        
        :param path: The path to the file
        :type path: str
        '''
        self.load_state_dict(torch.load(path))


# In[4]:


# if running from notebook
if __name__ == '__main__':
    get_ipython().system('jupyter nbconvert --to script LSTM.ipynb')

