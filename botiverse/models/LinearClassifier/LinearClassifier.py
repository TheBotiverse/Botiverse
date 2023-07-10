from torch import nn

class LinearClassifier(nn.Module):
    def __init__(self, input_embeddings_size, output_size):
        super(LinearClassifier, self).__init__()
        # input_embeddings_size as this is gets its input from the output of the BERT model
        self.linear = nn.Linear(input_embeddings_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input):
        output = self.linear(input)
        output = self.softmax(output)
        return output