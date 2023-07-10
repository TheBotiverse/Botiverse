import pandas as pd
from transformers import AutoTokenizer
import numpy as np

class ConverseBot_Preprocessor:
    def __init__(self, dataset=None):# dataset structure is an array of multiturn conversations (each multiturn conversation is an array of strings)[["hi","hello","how are you?"], ["good","how about you?","i am fine"]]
        self.data = dataset
        # create the t5 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    def process(self):
        # separate conversations into text and target
        target_array = []
        text_array = []
        for conversation in self.data:
            # get length of conversation
            conversation_length = len(conversation)
            # get a random index to split the conversation
            split_index = np.random.randint(0, conversation_length//2)*2+1
            # split the conversation into input and target
            conternt = conversation[:split_index]
            target = conversation[split_index]
            conternt = "[C]".join(conternt)
            # add the input and target to the data
            text_array.append(conternt)
            target_array.append(target)
        # create a dataframe
        self.data = pd.DataFrame({'text': text_array, 'target': target_array})
        # clean the text
        self.data['text'] = self.data['text'].apply(self.clean_string)
        self.data['target'] = self.data['target'].apply(self.clean_string)
        # tokenize the data with padding to 512 and truncation to 512 (from the end)
        self.data['text'] = self.data['text'].apply(self.tokenize_string)
        self.data['target'] = self.data['target'].apply(self.tokenize_string, target=True)
        # get text ids and attention masks
        self.data['text_input_ids'] = self.data['text'].apply(lambda x: x['input_ids'])
        self.data['text_attention_mask'] = self.data['text'].apply(lambda x: x['attention_mask'])
        self.data = self.data.drop(columns=['text'])
        # get traget ids
        self.data['target'] = self.data['target'].apply(lambda x: x['input_ids'])
        return self.data

    # process a single string
    def clean_string(self, string):
        string = string.replace("\n", " ")
        # remove spaces at the beginning and end of the string
        string = string.strip()
        # remove spaces before and after punctuation
        string = string.replace(" .", ".")
        string = string.replace(" ,", ",")
        string = string.replace(" !", "!")
        string = string.replace(" ?", "?")
        string = string.replace(" '", "'")
        string = string.replace(" ’", "’")
        return string

    # tokenize a single string
    def tokenize_string(self, string, target=False):
        tokens_obj = self.tokenizer(string, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        if target:
            tokens_obj['input_ids'][tokens_obj['input_ids'] == self.tokenizer.pad_token_id] = -100
        return tokens_obj

    # decode a single string
    def decode_tokens(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    # clean then tokenize a single string
    def process_string(self, string):
        string = self.clean_string(string)
        tokens_vector = self.tokenize_string(string)
        return tokens_vector