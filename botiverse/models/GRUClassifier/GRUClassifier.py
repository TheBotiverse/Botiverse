import torch
import torch.nn as nn

class BasicGRU(nn.Module):
    '''An interface for the basic GRU unit'''
    def __init__(self, input_size, dropout_p=0.1):
        """
        Constructs a BasicGRU instance with specific layer sizes and dropout probability.
        
        :param input_size: The size of the input to the model.
        :type input_size: int

        :param dropout_p: A regularization parameter.
        :type dropout_p: float

        :returns: None
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(BasicGRU, self).__init__()
        self.hidden_size = input_size
        # update gate
        self.W_z = nn.Linear(input_size, input_size)
        # reset gate
        self.W_r = nn.Linear(input_size, input_size)
        # new memory gate
        self.W_h = nn.Linear(input_size, input_size)
        # activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # dropout (for regularization)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, hidden):
        """
        Defines the computation performed by the model.

        :param input: The provided sequance input.
        :type input: Tensor
        
        :param hidden: The provided hidden state.
        :type hidden: Tensor

        :returns: New hidden state.
        :rtype: Tensor
        """
        combined = input + hidden
        # update activation vector calculation
        z = self.W_z(combined)
        z = self.sigmoid(z)
        # reset activation vector calculation
        r = self.W_r(combined)
        r = self.sigmoid(r)
        # new memory vector calculation
        h = self.W_h(input + r * hidden)
        h = self.tanh(h)
        # set the memory with the weighted sum of old and new memory
        hidden = (1 - z) * hidden + z * h
        # apply dropout
        hidden = self.dropout(hidden)
        return hidden

    # initialize the hidden state with zeros (initial memory)
    def initHidden(self, batch_size):
        """
        Creates a tensor of zeros for the hidden state initialization.

        :param batch_size: The size of the batch for which the hidden state is to be initialized.
        :type batch_size: int

        :returns: Tensor of zeros of the shape (batch_size, 1, hidden_size).
        :rtype: Tensor
        """
        return torch.zeros(batch_size, 1, self.hidden_size).to(self.device)

class GRUTextClassifier(nn.Module):
    ''' An interface for the GRU text classifier which uses a basic GRU unit with a linear output layer and an input embedding layer'''
    def __init__(self, vocabulary, embedding_size, output_size, dropout_p=0.1):
        """
        Constructs a GRUTextClassifier instance with specific hyperparameters.
 

        :param vocabulary: The size of vocabulary used in the Embedding layer.
        :type vocabulary: int
 
        :param embedding_size: The size of each embedding vector.
        :type embedding_size: int
 
        :param output_size: The size of the output from the model (number of classes).
        :type output_size: int

        :param dropout_p: A regularization parameter.
        :type dropout_p: float
 
        :returns: None
        """
        super(GRUTextClassifier, self).__init__()
        # the embedding layer
        self.embedding = nn.Embedding(vocabulary, embedding_size)
        # the GRU layer
        self.gru = BasicGRU(embedding_size, dropout_p)
        # the output layer
        self.h2o = nn.Linear(embedding_size, output_size)
        # activation function
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        """
        Defines the computation performed by the model.

        :param input: The model input.
        :type input: Tensor

        :returns: Output after the forward pass (classes probabilities).
        :rtype: Tensor
        """
        batch_size = input.size()[0]
        # get the first hidden state
        hidden = self.gru.initHidden(batch_size)
        # embed the input
        input_temp = self.embedding(input)
        # pass the input through the GRU layer for each token in the sequence
        for i in range(input_temp.size()[1]):
            hidden = self.gru(input_temp[:, i:i+1, :], hidden)
        # pass the last hidden state through the output layer
        output = self.h2o(hidden.squeeze(1))
        # apply softmax
        output = self.softmax(output)
        return output
