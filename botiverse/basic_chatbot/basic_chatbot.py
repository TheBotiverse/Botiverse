import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, dataloader
from gensim.utils import tokenize
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import numpy as np

from botiverse.models import SVM
from botiverse.models import NeuralNet

import gdown
import os


class basic_chatbot:

    def __init__(self, machine='NN', repr='tf-idf'):
        """
        Instantiate a basic chat bot model that uses a classic feedforward neural network.
        Data can be then used to train the chatbot model.
        
        :param name: The chatbot's name.
        :type name: string
        """
        self.model = None
        self.machine = machine
        self.repr = repr
        self.glove_dict = None
        if repr == 'glove' or 'tf-idf-glove': self.load_glove_vectors()
        self.tf = None
        self.idf = None
        self.classes = None
        
        
    # Load GLOVE word vectors
    def load_glove_vectors(self, force_download=False):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        path = curr_dir + '/glove.6B.50d.txt'
        if not os.path.exists(path) or force_download:
            print("GLoVE embeddings not found. Downloading now...")
            f_id = '1BOSO0rR3ZzjWlv5WYzCux6ZluBP_vNDv' 
            gdown.download(f'https://drive.google.com/uc?export=download&confirm=pbef&id={f_id}', curr_dir + '/glove.6B.50d.txt', quiet=False)
            print("Done.")   
        
        glove_dict = {}         # dictionary mapping words to their GloVe vector representation
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                # In each line, the first value is the word, the rest are the values of the vector
                word = values[0]            
                vector = np.asarray(values[1:], dtype='float32')
                glove_dict[word] = vector
        self.glove_dict = glove_dict


    def setup_glove_vectors(self, sentence_list):
        # make a numpy array of the sentence vectors
        X = np.zeros((len(sentence_list), 50))
        for i, sentence in enumerate(sentence_list):
            if self.repr == 'glove':
                X[i] = self.get_glove_vector(sentence)
            elif self.repr == 'tf-idf-glove':
                X[i] = self.get_tf_idf_glove_vector(sentence)
                
        return X
        
    # Calculate average vector
    def get_tf_idf_glove_vector(self, sentence):
        
        tokens = list(tokenize(sentence, to_lower=True))
        tokens_s = [stemmer.stem(word.lower()) for word in tokens if word not in ['?', '!', '.', ',']]

        # get the idf of each word in the sentence
        weights = np.zeros((len(tokens)))
        for i, word in enumerate(tokens_s):
            # if word in self.all_words:
            if word in self.all_words and tokens[i] in self.glove_dict:
                weights[i] = self.idf[self.all_words.index(word)]
        
        # normalize the weights
        weights = weights / np.sum(weights)
        
        # get the weighted average of the glove vectors
        avg_vector = np.zeros(50)
        for i, word in enumerate(tokens):
            if word in self.glove_dict:
                avg_vector += weights[i] * self.glove_dict[word]
        
        avg_vector = avg_vector[np.newaxis, :]
        return avg_vector

    # Calculate average vector
    def get_glove_vector(self, sentence):
        tokens = list(tokenize(sentence, to_lower=True))
        vector_sum = np.zeros(50)  
        num_glove_tokens = 0            # num of words that occur in glove vocab
        for token in tokens:
            if token in self.glove_dict:
                vector_sum += self.glove_dict[token]
                num_glove_tokens += 1

        avg_vector = np.zeros_like(self.glove_dict['a'])  if num_glove_tokens == 0 else vector_sum / num_glove_tokens
        avg_vector = avg_vector[np.newaxis, :]
        return avg_vector

        
    def setup_tf_idf(self, sentence_list, all_words):
        '''
        Given a list of tokenized sentences, return a table of tf-idf vectors (one for each sentence)
        '''
        # Compute the normalized frequency of each word in the document
        tf_table = np.zeros((len(sentence_list), len(all_words)), dtype=np.float64)
        for i, sentence in enumerate(sentence_list):
            sentence = list(tokenize(sentence, to_lower=True))
            sentence = [stemmer.stem(word.lower()) for word in sentence if word not in ['?', '!', '.', ',']]
            sentence_length = len(sentence)
            for word in sentence:
                word_index = all_words.index(word)
                tf_table[i, word_index] += 1 / sentence_length
        
        # Get the number of documents in which each word appears
        df = np.sum(tf_table > 0, axis=0)
        N = len(sentence_list)
        idf_col = np.log(N/(df+1))
        
        # compute the tf-table by the idf column (gets broadcasted)
        self.tf, self.idf = tf_table, idf_col
        tfidf_table = tf_table * idf_col
        
        return tfidf_table

    def get_tf_idf(self, sentence):
        '''
        Given a sentence, return its tf-idf vector.
        '''
        sentence = list(tokenize(sentence, to_lower=True))
        sentence = [stemmer.stem(word.lower()) for word in sentence if word not in ['?', '!', '.', ',']]
        # compute the tf-idf vector for the prompt
        tf_idf = np.zeros((1, len(self.all_words)), dtype=np.float64)
        for word in sentence:
            # get its tf
            if word not in self.all_words:  continue
            word_index = self.all_words.index(word)
            tf_idf[0, word_index] += 1 / len(sentence)
        tf_idf *= self.idf
        return tf_idf

    def setup_data(self):
        """
        Given JSON data, set up the data for training by converting it to a list of sentences and their corresponding classes.
        """  
        all_words = []
        classes = []
        sentence_list = []                             # sentence_table[i] is a tuple (list of words, class)
        y = []
        for intent in self.raw_data['FAQ']:             #this is a list of dictionaries. each has a tag (class), list of patterns and list of responses.
            tag = intent['tag']
            classes.append(tag)
            for pattern in intent['patterns']:
                if self.repr == 'tf-idf' or self.repr == 'tf-idf-glove':
                    all_words += list(tokenize(pattern, to_lower=True))     
                sentence_list.append(pattern)
                y.append(tag)

        # stem and lower each word
        all_words = [stemmer.stem(word.lower()) for word in all_words if word not in ['?', '!', '.', ',']]
        
        # remove duplicates and sort alphabetically
        all_words = sorted(set(all_words))
        classes = sorted(set(classes))
        
        self.all_words = all_words
        self.classes = classes

        if self.repr == 'tf-idf':        
            X = self.setup_tf_idf(sentence_list, all_words)
        elif self.repr == 'glove':
            X = self.setup_glove_vectors(sentence_list)
        elif self.repr == 'tf-idf-glove':
            _ = self.setup_tf_idf(sentence_list, all_words)
            X = self.setup_glove_vectors(sentence_list)
            

        # convert each class to its index
        for i, tag in enumerate(y):
            y[i] = classes.index(tag)
        y = np.array(y)
        return X, y

    def train(self, path):
        """
        Train the chatbot model with the given JSON data.
        
        :param data: A stringfied JSON object containing the training data 
        :type number: string
    
        :return: None
        :rtype: NoneType
        """
        with open(path, 'r') as f:
            self.raw_data = json.load(f) 

        X, y = self.setup_data()
        if self.machine == 'NN':
            self.model = NeuralNet(structure=[X.shape[1], 12, len(self.classes)], activation='sigmoid')
            self.model.fit(X, y, batch_size=1, epochs=1000, λ = 0.04, eval_train=True)
        elif self.machine == 'SVM':
            self.model = SVM(kernel='linear', C=700)
            self.model.fit(X, y, eval_train=True)
            print("meh")
        
    def infer(self, prompt, confidence=None):
        """
        Infer a suitable response to the given prompt.
        
        :param promp: The user's prompt
        :type number: string
    
        :return: The chatbot's response
        :rtype: string
        """
        if confidence is None: confidence = 1.5/len(self.classes)
        vector = self.get_tf_idf(prompt) if self.repr == 'tf-idf' else self.get_glove_vector(prompt) if self.repr == 'glove' else self.get_tf_idf_glove_vector(prompt)
        # predict the class of the prompt
        tag_idx, tag_prob = self.model.predict(vector)
        tag_idx, tag_prob = tag_idx[0], tag_prob[0]
        tag = self.classes[tag_idx]
        if tag_prob < confidence: return "Could you rephrase that?"
        for intent in self.raw_data['FAQ']:
            if tag == intent["tag"]:
                return np.random.choice(intent['responses'])