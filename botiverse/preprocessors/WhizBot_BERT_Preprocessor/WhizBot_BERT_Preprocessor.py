import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

class WhizBot_BERT_Preprocessor:
    def __init__(self, file_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.file_path = file_path
        self.text_col = "text"
        self.label_col = "label"
        self.data = pd.read_json(self.file_path)
        # make a dictionary of all the tag and responses
        self.responces = {}
        for i in range(len(self.data)):
            self.responces[self.data['tag'][i]] = self.data['responses'][i]
        # remove the responses column from the dataframe
        self.data = self.data.drop('responses', axis=1)
        # stretch the dataframe to have one row per pattern
        self.data = self.data.explode('patterns')
        # rename the columns tag->label and patterns->text
        self.data = self.data.rename(columns={'tag': 'label', 'patterns': 'text'})
        # remove all rows with empty text
        self.data = self.data.dropna()
        self.data = self.data.reset_index(drop=True)
        # load the pretrained model and freeze its parameters (not to be trained)
        self.bert = AutoModel.from_pretrained("bert-base-multilingual-cased").to(self.device)
        for param in self.bert.parameters():
            param.requires_grad = False

    def process(self):
        # text processing
        self.data['text'] = self.data['text'].apply(self.clean_string)
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        # tokenize the data
        tqdm.pandas(desc='Tokenizing strings')
        self.data['text'] = self.data['text'].progress_apply(self.tokenize_string)
        # embed the tokens
        tqdm.pandas(desc='Embedding tokens')
        self.data['text'] = self.data['text'].progress_apply(self.embed_tokens)
        # make a dictionary of all labels
        self.label_dict = {}
        for i in range(len(self.data)):
            if self.data['label'][i] not in self.label_dict:
                self.label_dict[self.data['label'][i]] = len(self.label_dict)
        # now replace all labels with their corresponding index
        self.data['label'] = self.data['label'].apply(lambda x: self.label_dict[x])
        # now convert to tensors
        self.data['label'] = self.data['label'].apply(lambda x: torch.tensor([x]))
        return self.data

    # process a single string
    def clean_string(self, string):
        string = string.lower()
        string = string.replace('[^\w\s]','')
        string = string.encode("ascii", "ignore").decode()
        return string

    # tokenize a single string
    def tokenize_string(self, string):
        tokens_obj = self.tokenizer(string, padding=True, truncation=True, max_length=256, return_tensors="pt")
        return tokens_obj

    # as the model is frozen, we can make the training way faster by precomputing the embeddings, with that it will only be computed once
    def embed_tokens(self, tokens_obj):
        # average of all the output embeddings (to capture the whole sentence meaning)
        embeddings = self.bert(input_ids=tokens_obj['input_ids'].to(self.device), attention_mask=tokens_obj['attention_mask'].to(self.device),return_dict=False)[0].mean(dim=1).detach().cpu()
        return embeddings

    # the whole preprocessing pipeline
    def process_string(self, string):
        string = self.clean_string(string)
        tokens_obj = self.tokenize_string(string)
        embeddings = self.embed_tokens(tokens_obj)
        return embeddings