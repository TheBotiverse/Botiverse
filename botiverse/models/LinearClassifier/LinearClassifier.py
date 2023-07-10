from torch import nn

class LinearClassifier(nn.Module):
    '''An interface for a simple linear classifier intended to be used with the BERT model'''
    def __init__(self, input_embeddings_size, output_size):
        super(LinearClassifier, self).__init__()
        # input_embeddings_size as this is gets its input from the output of the BERT model
        """
        Constructs a LinearClassifier instance with specific layer sizes.

        :param input_embeddings_size: The size of each input embeddings from BERT.
        :type input_embeddings_size: int
 
        :param output_size: The size of the output from the model.
        :type output_size: int
 
        :returns: None
        """
        self.linear = nn.Linear(input_embeddings_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input):
        """
        Defines the computation performed by the model.

        :param input: The provided input from the BERT model.
        :type input: Tensor

        :returns: Output after the forward pass (classes probabilities).
        :rtype: Tensor
        """
        output = self.linear(input)
        output = self.softmax(output)
        return output