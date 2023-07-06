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