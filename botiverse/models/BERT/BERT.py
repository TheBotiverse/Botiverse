"""
Theis Module contains the BERT model architecture.
"""

import torch
import torch.nn as nn



# Embeddings
# 1. Cluster similar words together.
# 2. Preserve different relationships between words such as: semantic, syntactic, linear,
# and since BERT is bidirectional it will also preserve contextual relationships as well.
class Embeddings(nn.Module):
    """
    Embedding layer for BERT.

    This layer takes input_ids and token_type_ids as inputs and generates word embeddings
    using three types of embeddings: word, position, and token_type embeddings.

    :param config: BERT configuration.
    :type config: Config
    """
    def __init__(self, config):
        super(Embeddings, self).__init__()
        # Bert uses 3 types of embeddings: word, position, and token_type (segment type).
        # LayerNorm is used to normalize the sum of the embeddings.
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.padding_idx)
        self.position_embeddings = nn.Embedding(config.max_seq, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.token_types, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, token_type_ids): # input_ids: [batch_size, seq_len] token_type_ids: [batch_size, seq_len]
        """
        Forward pass of the Embeddings layer.

        :param input_ids: The input token IDs.
        :type input_ids: torch.Tensor
        :param token_type_ids: The token type IDs.
        :type token_type_ids: torch.Tensor
        :return: The generated embeddings.
        :rtype: torch.Tensor
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand_as(input_ids).to(device) # position_ids: [batch_size, seq_len]
        word_embeddings = self.word_embeddings(input_ids)  # word_embeddings: [batch_size, seq_len, hidden_size]
        position_embeddings = self.position_embeddings(position_ids) # position_embeddings: [batch_size, seq_len, hidden_size]
        token_type_embeddings = self.token_type_embeddings(token_type_ids) # token_type_embeddings: [batch_size, seq_len, hidden_size]
        embeddings = word_embeddings + position_embeddings + token_type_embeddings # embeddings: [batch_size, seq_len, hidden_size]
        # Normalize by subtracting the mean and dividing by the standard deviation calculated across the feature dimension
        # then multiply by a learned gain parameter and add to a learned bias parameter.
        embeddings = self.layer_norm(embeddings) # embeddings: [batch_size, seq_len, hidden_size]
        embeddings = self.dropout(embeddings) # embeddings: [batch_size, seq_len, hidden_size]
        return embeddings

# Encoder layer
class EncoderLayer(nn.Module):
    """
    Encoder layer for BERT.

    This layer contains self-attention, layer normalization, and position-wise feed-forward network.

    :param config: BERT configuration.
    :type config: Config
    """

    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(config)
        self.self_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_dropout = nn.Dropout(config.dropout)
        self.position_wise_feed_forward = PositionWiseFeedForward(config)
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_dropout = nn.Dropout(config.dropout)

    def forward(self, input, attention_mask):
        """
        Forward pass of the EncoderLayer.

        :param input: The input tensor.
        :type input: torch.Tensor
        :param attention_mask: The attention mask.
        :type attention_mask: torch.Tensor
        :return: The output tensor.
        :rtype: torch.Tensor
        """

        # Multi-head attention
        context, attention = self.self_attention(input, input, input, attention_mask) # context: [batch_size, seq_len, hidden_size] attention: [batch_size, heads, seq_len, seq_len]
        # Add and normalize
        context = self.self_dropout(context) # context: [batch_size, seq_len, hidden_size]
        output = self.self_layer_norm(input + context) # output: [batch_size, seq_len, hidden_size]
        # Position-wise feed-forward network
        context = self.position_wise_feed_forward(output) # context: [batch_size, seq_len, hidden_size]
        # Add and normalize
        context = self.ffn_dropout(context) # context: [batch_size, seq_len, hidden_size]
        output = self.ffn_layer_norm(output + context) # output: [batch_size, seq_len, hidden_size]
        return output, attention

# Multi-head attention
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer for BERT.

    This layer performs multi-head self-attention and returns the output context.

    :param config: BERT configuration.
    :type config: Config
    """
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.config = config
        self.w_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.w_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.w_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.w_o = nn.Linear(config.hidden_size, config.hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query, key, value, attention_mask): 
        """
        Forward pass of the MultiHeadAttention.

        :param query: The query tensor.
        :type query: torch.Tensor
        :param key: The key tensor.
        :type key: torch.Tensor
        :param value: The value tensor.
        :type value: torch.Tensor
        :param attention_mask: The attention mask.
        :type attention_mask: torch.Tensor
        :return: The output context.
        :rtype: torch.Tensor
        """                                                  
                
        # query: [batch_size, seq_len, hidden_size] key: [batch_size, seq_len, hidden_size]
        # value: [batch_size, seq_len, hidden_size] attention_mask: [batch_size, seq_len_q, seq_len_k]                                          

        batch_size, seq_len, hidden_size = query.size()

        query = self.w_q(query).view(batch_size, seq_len, self.config.heads, hidden_size // self.config.heads).transpose(1, 2) # query: [batch_size, heads, seq_len, hidden_size // heads]
        key = self.w_k(key).view(batch_size, seq_len, self.config.heads, hidden_size // self.config.heads).transpose(1, 2) # key: [batch_size, heads, seq_len, hidden_size // heads]
        value = self.w_v(value).view(batch_size, seq_len, self.config.heads, hidden_size // self.config.heads).transpose(1, 2) # value: [batch_size, heads, seq_len, hidden_size // heads]

        # Scaled dot-product attention
        attention = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(float(hidden_size // self.config.heads))) # attention: [batch_size, heads, seq_len, seq_len]
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.config.heads, 1, 1) # attention_mask: [batch_size, heads, seq_len_q, seq_len_k]
        attention_mask = (attention_mask == 0)
        attention.masked_fill_(attention_mask, -1e9) # attention: [batch_size, heads, seq_len, seq_len]
        attention = self.softmax(attention) # attention: [batch_size, heads, seq_len, seq_len]
        attention = self.dropout(attention) # attention: [batch_size, heads, seq_len, seq_len]
        context = torch.matmul(attention, value) # context: [batch_size, heads, seq_len, hidden_size // heads]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size) # context: [batch_size, seq_len, hidden_size]
        output = self.w_o(context) # output: [batch_size, seq_len, hidden_size]
        return output, attention

# Position-wise feed-forward network
class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network layer for BERT.

    This layer applies two linear transformations with a GELU activation function.

    :param config: BERT configuration.
    :type config: Config
    """
    def __init__(self, config):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.ff_size)
        self.linear2 = nn.Linear(config.ff_size, config.hidden_size)
        self.gelu = nn.GELU()

    def forward(self, input):
        """
        Forward pass of the PositionWiseFeedForward layer.

        :param input: The input tensor.
        :type input: torch.Tensor
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        output = self.linear1(input) # output: [batch_size, seq_len, ff_size]
        output = self.gelu(output) # output: [batch_size, seq_len, ff_size]
        output = self.linear2(output) # output: [batch_size, seq_len, hidden_size]
        return output

# Bert
# 1. Puts it all together.
class Bert(nn.Module):
    """
    BERT model implementation.

    This model combines the Embeddings layer, EncoderLayers, and linear transformation layers
    to perform BERT-based processing.

    :param config: BERT configuration.
    :type config: Config
    """
    def __init__(self, config):
        super(Bert, self).__init__()
        self.embeddings = Embeddings(config)
        self.encoder = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, input_ids, token_type_ids, attention_mask, return_dict=False): # input_ids: [batch_size, seq_len] token_type_ids: [batch_size, seq_len] attention_mask: [batch_size, seq_len]
        """
        Forward pass of the Bert model.

        :param input_ids: The input token IDs.
        :type input_ids: torch.Tensor
        :param token_type_ids: The token type IDs.
        :type token_type_ids: torch.Tensor
        :param attention_mask: The attention mask.
        :type attention_mask: torch.Tensor
        :param return_dict: Whether to return a dictionary or not, defaults to False.
        :type return_dict: bool
        :return: The sequence output and pooled output.
        :rtype: torch.Tensor, torch.Tensor
        """
        
        # Embedding
        output = self.embeddings(input_ids, token_type_ids) # output: [batch_size, seq_len, hidden_size]

        # Encoder
        attention_mask = attention_mask.unsqueeze(1).repeat(1, output.size(1), 1) # attention_mask: [batch_size, seq_len, seq_len]
        for encoder_layer in self.encoder:
            output, attention = encoder_layer(output, attention_mask) # output: [batch_size, seq_len, hidden_size] attention: [batch_size, heads, seq_len, seq_len]

        # Sequnce and pooled outputs
        sequence_output = output # sequence_output: [batch_size, seq_len, hidden_size]
        pooled_output = self.tanh(self.linear(sequence_output[:, 0])) # pooled_output: [batch_size, hidden_size]

        return sequence_output, pooled_output