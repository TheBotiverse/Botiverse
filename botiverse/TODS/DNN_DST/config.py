import tokenizers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
DEV_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 5
HID_DIM = 768
N_OPER = 6
DROPOUT = 0.3
VOCAB_PATH = 'botiverse/TODS/DNN_DST/vocab.txt'
MODEL_PATH = './model.pt'
TOKENIZER = tokenizers.BertWordPieceTokenizer(VOCAB_PATH, lowercase=True)
IGNORE_IDX = -100
OPER2ID = {'carryover' : 0, 'dontcare': 1, 'update':2, 'delete':3, 'yes':4, 'no':5}