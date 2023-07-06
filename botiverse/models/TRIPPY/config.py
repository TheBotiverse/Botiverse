import tokenizers
import os

# TO DO: DELETE THIS MULTIWOZ CONST
MULTIWOZ = True


# Trippy configuration
class TRIPPYConfig(object):
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
                 multiwoz=MULTIWOZ):

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
        self.tokenizer = tokenizers.BertWordPieceTokenizer(self.vocab_path, lowercase=True)
        self.ignore_idx = ignore_idx
        self.oper2id = oper2id
        self.weight_decay = weight_decay
        self.lr = lr
        self.adam_epsilon = adam_epsilon
        self.warmup_proportion = warmup_proportion
        self.multiwoz = multiwoz


# Other config variables for experiments with different datasets

# # Multi-woz
# ONTOLOGY_PATH = './Dataset/fixed/ontology.json'
# LABEL_MAPS_PATH = './Dataset/fixed/label_maps.json'
# TRAIN_DATA_PATH = './Dataset/fixed/train_dials.json'
# DEV_DATA_PATH = './Dataset/fixed/dev_dials.json'
# TEST_DATA_PATH = './Dataset/fixed/test_dials.json'
# MODEL_PATH = './Models/model.pt'
# EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
# # EXPERIMENT_DOMAINS = ["restaurant"]
# NON_REFERABLE_SLOTS = ['hotel-stars', 'hotel-internet', 'hotel-parking']
# NON_REFERABLE_PAIRS = [('hotel-book_people','hotel-book_stay'), ('restaurant-book_people','hotel-book_stay')]
# MULTIWOZ = True


# # Woz
# ONTOLOGY_PATH = './Woz2/fixed/ontology.json'
# LABEL_MAPS_PATH = './Woz2/fixed/label_maps.json'
# TRAIN_DATA_PATH = './Woz2/fixed/train_dials.json'
# DEV_DATA_PATH = './Woz2/fixed/dev_dials.json'
# TEST_DATA_PATH = './Woz2/fixed/test_dials.json'
# MODEL_PATH = './Models/model.pt'
# EXPERIMENT_DOMAINS = ["restaurant"]
# NON_REFERABLE_SLOTS = []
# NON_REFERABLE_PAIRS = []
# MULTIWOZ = False