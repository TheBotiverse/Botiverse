"""
This module contains the configuration class for BERT.
"""

# Bert configuration
class BERTConfig(object):
    """
    Configuration class for BERT.

    This class holds the configuration parameters for the BERT model.

    :param vocab_size: The size of the vocabulary, defaults to 30522.
    :type vocab_size: int
    :param hidden_size: The hidden size of the BERT model, defaults to 768.
    :type hidden_size: int
    :param encoder_layers: The number of encoder layers in the BERT model, defaults to 12.
    :type encoder_layers: int
    :param heads: The number of attention heads in the BERT model, defaults to 12.
    :type heads: int
    :param ff_size: The size of the feed-forward layer in the BERT model, defaults to 3072.
    :type ff_size: int
    :param token_types: The number of token types in the BERT model, defaults to 2.
    :type token_types: int
    :param max_seq: The maximum sequence length in the BERT model, defaults to 512.
    :type max_seq: int
    :param padding_idx: The padding index used in the BERT model, defaults to 0.
    :type padding_idx: int
    :param layer_norm_eps: The epsilon value for layer normalization in the BERT model, defaults to 1e-12.
    :type layer_norm_eps: float
    :param dropout: The dropout rate in the BERT model, defaults to 0.1.
    :type dropout: float
    """
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