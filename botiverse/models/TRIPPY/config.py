"""
This Module has the configuration class for TRIPPY.
"""

import tokenizers
import os
import gdown

# Trippy configuration
class TRIPPYConfig(object):
    """
    Configuration class for TRIPPY.

    This class holds the configuration parameters for the TRIPPY model.

    :param max_len: The maximum sequence length, defaults to 128.
    :type max_len: int
    :param train_batch_size: The batch size for training, defaults to 32.
    :type train_batch_size: int
    :param dev_batch_size: The batch size for development evaluation, defaults to 1.
    :type dev_batch_size: int
    :param test_batch_size: The batch size for testing, defaults to 1.
    :type test_batch_size: int
    :param epochs: The number of training epochs, defaults to 15.
    :type epochs: int
    :param hid_dim: The hidden dimension size, defaults to 768.
    :type hid_dim: int
    :param n_oper: The number of operations, defaults to 7.
    :type n_oper: int
    :param dropout: The dropout rate, defaults to 0.3.
    :type dropout: float
    :param vocab_path: The path to the vocabulary file, defaults to 'vocab.txt'.
    :type vocab_path: str
    :param ignore_idx: The index value to ignore, defaults to -100.
    :type ignore_idx: int
    :param oper2id: The mapping of operation names to IDs, defaults to {'carryover' : 0, 'dontcare': 1, 'update':2, 'refer':3, 'yes':4, 'no':5, 'inform':6}.
    :type oper2id: dict[str, int]
    :param weight_decay: The weight decay value, defaults to 0.0.
    :type weight_decay: float
    :param lr: The learning rate, defaults to 1e-4.
    :type lr: float
    :param adam_epsilon: The epsilon value for Adam optimizer, defaults to 1e-6.
    :type adam_epsilon: float
    :param warmup_proportion: The proportion of warmup steps, defaults to 0.1.
    :type warmup_proportion: float
    :param multiwoz: The path to the MultiWOZ dataset, defaults to False.
    :type multiwoz: str
    """
    def __init__(self, 
                 max_len=128, 
                 train_batch_size=32, 
                 dev_batch_size=1, 
                 test_batch_size=1, 
                 epochs=15, 
                 hid_dim=768, 
                 n_oper=7, 
                 dropout=0.3,
                 vocab_path='vocab.txt',
                 ignore_idx=-100,
                 oper2id={'carryover' : 0, 'dontcare': 1, 'update':2, 'refer':3, 'yes':4, 'no':5, 'inform':6},
                 weight_decay=0.0,
                 lr=1e-4,
                 adam_epsilon=1e-6,
                 warmup_proportion=0.1,
                 multiwoz=False):

        self.max_len = max_len
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.hid_dim = hid_dim
        self.n_oper = n_oper
        self.dropout = dropout
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.vocab_path = os.path.join(cur_dir, vocab_path)
        if not os.path.exists(self.vocab_path):
            print("Downloading Vocab...")
            f_id = '1f2iOTT-QiFbIc1naqGVZWX5wPVo7gUMS' 
            gdown.download(f'https://drive.google.com/uc?export=download&confirm=pbef&id={f_id}', self.vocab_path, quiet=False)
            print("Done.")   
        self.tokenizer = tokenizers.BertWordPieceTokenizer(self.vocab_path, lowercase=True)
        self.ignore_idx = ignore_idx
        self.oper2id = oper2id
        self.weight_decay = weight_decay
        self.lr = lr
        self.adam_epsilon = adam_epsilon
        self.warmup_proportion = warmup_proportion
        self.multiwoz = multiwoz