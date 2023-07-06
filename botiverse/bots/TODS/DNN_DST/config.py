import tokenizers
import os

# Deep DST config
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
DEV_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 15
HID_DIM = 768
N_OPER = 7
DROPOUT = 0.3

curr_dir = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(curr_dir, 'vocab.txt')

TOKENIZER = tokenizers.BertWordPieceTokenizer(VOCAB_PATH, lowercase=True)
IGNORE_IDX = -100
OPER2ID = {'carryover' : 0, 'dontcare': 1, 'update':2, 'refer':3, 'yes':4, 'no':5, 'inform':6}
WEIGHT_DECAY = 0.0
LR = 1e-4 #1e-5 #1e-4 #2e-5
ADAM_EPSILON = 1e-6
WARMUP_PROPORTION = 0.1 #0.0 #0.1



# TO DO: DELETE THIS MULTIWOZ CONST
MULTIWOZ = True


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