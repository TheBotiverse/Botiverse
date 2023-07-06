import torch
import torch.nn as nn
from collections import OrderedDict
from transformers import BertModel


# Bert configuration
class BERTConfig(object):
    def __init__(self, vocab_size=30522, hidden_size=768, encoder_layers=12, heads=12, ff_size=3072, token_types=2, max_seq=512, padding_idx=0, layer_norm_eps=1e-12, dropout=0.1):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.encoder_layers = encoder_layers
        self.heads = heads
        self.ff_size = ff_size
        self.token_types = token_types
        self.max_seq = max_seq
        self.padding_idx = padding_idx
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout

config = BERTConfig()

# Embeddings
# 1. Cluster similar words together.
# 2. Preserve different relationships between words such as: semantic, syntactic, linear,
# and since BERT is bidirectional it will also preserve contextual relationships as well.
class Embeddings(nn.Module):
    def __init__(self):
        super(Embeddings, self).__init__()
        # Bert uses 3 types of embeddings: word, position, and token_type (segment type).
        # LayerNorm is used to normalize the sum of the embeddings.
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.padding_idx)
        self.position_embeddings = nn.Embedding(config.max_seq, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.token_types, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, token_type_ids): # input_ids: [batch_size, seq_len] token_type_ids: [batch_size, seq_len]
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
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention()
        self.self_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_dropout = nn.Dropout(config.dropout)
        self.position_wise_feed_forward = PositionWiseFeedForward()
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_dropout = nn.Dropout(config.dropout)

    def forward(self, input, attention_mask):
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
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.w_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.w_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.w_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.w_o = nn.Linear(config.hidden_size, config.hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query, key, value, attention_mask): # query: [batch_size, seq_len, hidden_size] key: [batch_size, seq_len, hidden_size]
                                                          # value: [batch_size, seq_len, hidden_size] attention_mask: [batch_size, seq_len_q, seq_len_k]

        batch_size, seq_len, hidden_size = query.size()

        query = self.w_q(query).view(batch_size, seq_len, config.heads, hidden_size // config.heads).transpose(1, 2) # query: [batch_size, heads, seq_len, hidden_size // heads]
        key = self.w_k(key).view(batch_size, seq_len, config.heads, hidden_size // config.heads).transpose(1, 2) # key: [batch_size, heads, seq_len, hidden_size // heads]
        value = self.w_v(value).view(batch_size, seq_len, config.heads, hidden_size // config.heads).transpose(1, 2) # value: [batch_size, heads, seq_len, hidden_size // heads]

        # Scaled dot-product attention
        attention = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(float(hidden_size // config.heads))) # attention: [batch_size, heads, seq_len, seq_len]
        attention_mask = attention_mask.unsqueeze(1).repeat(1, config.heads, 1, 1) # attention_mask: [batch_size, heads, seq_len_q, seq_len_k]
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
    def __init__(self):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.ff_size)
        self.linear2 = nn.Linear(config.ff_size, config.hidden_size)
        self.gelu = nn.GELU()

    def forward(self, input):
        output = self.linear1(input) # output: [batch_size, seq_len, ff_size]
        output = self.gelu(output) # output: [batch_size, seq_len, ff_size]
        output = self.linear2(output) # output: [batch_size, seq_len, hidden_size]
        return output

# Bert
# 1. Puts it all together.
class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.embeddings = Embeddings()
        self.encoder = nn.ModuleList([EncoderLayer() for _ in range(config.encoder_layers)])
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, input_ids, token_type_ids, attention_mask, return_dict=False): # input_ids: [batch_size, seq_len] token_type_ids: [batch_size, seq_len] attention_mask: [batch_size, seq_len]

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

# Load pre-trained weights from trnasformers library
def LoadPretrainedWeights(model):

    # Get pre-trained weights from transformers library
    pretrained_model = BertModel.from_pretrained('bert-base-uncased')
    state_dict = pretrained_model.state_dict()

    # Delete position_ids from the state_dict if available
    if 'embeddings.position_ids' in state_dict.keys():
        del state_dict['embeddings.position_ids']

    # Get the new weights keys from the model
    new_keys = list(model.state_dict().keys())

    # Get the weights from the state_dict
    old_keys = list(state_dict.keys())
    weights = list(state_dict.values())

    # Create a new state_dict with the new keys
    new_state_dict = OrderedDict()
    for i in range(len(new_keys)):
        new_state_dict[new_keys[i]] = weights[i]
        print(old_keys[i], '->', new_keys[i])

    model.load_state_dict(new_state_dict)

# Example comparing the outputs of the from scratch model to the pre-trained model from transformers library
import torch
from transformers import BertModel, BertTokenizer

def Example():

    # Build a BERT model from scratch
    model = Bert()
    LoadPretrainedWeights(model)

    # Load pre-trained weights from the Transformers library
    pretrained_weights = 'bert-base-uncased'
    pretrained_model = BertModel.from_pretrained(pretrained_weights)

    # Tokenize the input sequence
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    input_text = ["This is a sample input sequence.", "batyousef is awesome"]
    inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')

    # Set dropout to zero during inference
    model.eval()
    pretrained_model.eval()

    ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']

    # Pass the inputs through both models and compare the outputs
    with torch.no_grad():
        model_output1, model_output2 = model(ids, token_type_ids, attention_mask)
        pretrained_output1, pretrained_output2 = pretrained_model(ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids,
                                            return_dict=False
                                            )

    print(model_output1.size(), model_output1)
    print(pretrained_output1.size(), pretrained_output1)
    print()
    print()
    print(model_output2.size(), model_output2)
    print(pretrained_output2.size(), pretrained_output2)

    print(model)
    print(pretrained_model)