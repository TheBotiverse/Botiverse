import pandas as pd
import torch
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE

class WhizBot_GRU_Preprocessor:
    '''An interface that provides the required preprocessing for the WhizBot_GRU bot'''
    def __init__(self, file_path):
        """
        Constructs a WhizBot_GRU_Preprocessor instance with the file path of the dataset.

        :param file_path: Path to the .json file if the dataset.
        :type file_path: str

        :returns: None
        """
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

    def process(self):
        """
        loads the data, cleans it, tokenizes it, pads it and removes outlier sequances.

        :returns: Processed data.
        :rtype: DataFrame
        """
        # text processing
        self.data['text'] = self.data['text'].apply(self.clean_string)
        # byte pair encoding tokenizer
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = Whitespace()
        # padding token is set to 0
        trainer = BpeTrainer(vocab_size=2000, min_frequency=2, special_tokens=["<pad>"])
        self.data['text'].to_csv("text.txt", index=False, header=False)
        self.tokenizer.train(["text.txt"], trainer)
        # tokenize the data
        self.data['text'] = self.data['text'].apply(self.tokenize_string)
        # remove outliers (too long or too short sequences)
        median = self.data['text'].apply(lambda x: len(x)).median()
        iqr = self.data['text'].apply(lambda x: len(x)).quantile(0.75) - self.data['text'].apply(lambda x: len(x)).quantile(0.25)
        self.data = self.data[self.data['text'].apply(lambda x: len(x) >= median - 1.5 * iqr)]
        self.data = self.data[self.data['text'].apply(lambda x: len(x) <= median + 1.5 * iqr)]
        self.data = self.data.reset_index(drop=True)
        # pad the text sequences
        padding_id = self.tokenizer.token_to_id("<pad>")
        self.longest_sequence = self.data['text'].apply(lambda x: len(x)).max()
        self.data['text'] = self.data['text'].apply(lambda x: torch.cat((x, torch.zeros(self.longest_sequence - len(x)).fill_(padding_id).long())))
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
        """
        Cleans the given text string by removing punctuation, converting to lowercase and removing non-ascii characters.

        :param string: Provided string.
        :type string: str

        :returns: Cleaned string.
        :rtype: str
        """
        string = string.lower()
        string = string.replace('[^\w\s]','')
        string = string.encode("ascii", "ignore").decode()
        return string

    # tokenize a single string
    def tokenize_string(self, string):
        """
        Tokenizes the given text string.

        :param string: Provided string.
        :type string: str

        :returns: Tokenizens Id.
        :rtype: Tensor
        """
        tokens_vector = self.tokenizer.encode(string).ids
        tokens_vector = torch.tensor(tokens_vector)
        return tokens_vector

    # process a single string
    def process_string(self, string):
        """
        Cleans and tokenizes a given text string, the pads it to the longest sequence length (for batch processing).

        :param string: Provided string.
        :type string: str

        :returns: Processed padded tokens ids.
        :rtype: Tensor
        """
        string = self.clean_string(string)
        tokens_vector = self.tokenize_string(string)
        padded_tokens_vector = torch.cat((tokens_vector, torch.zeros(self.longest_sequence - len(tokens_vector)).long()))
        return padded_tokens_vector

    # pad a sequence (to make batch compatible)
    def pad_sequence(self, sequence):
        """
        Pads a given sequence to make it compatible with batch processing.

        :param sequence: Provided sequence.
        :type sequence: Tensor

        :returns: Padded sequence.
        :rtype: Tensor
        """
        return torch.nn.utils.rnn.pad_sequence(sequence.tolist(), batch_first=True, padding_value=self.tokenizer.token_to_id("[PAD]"))
