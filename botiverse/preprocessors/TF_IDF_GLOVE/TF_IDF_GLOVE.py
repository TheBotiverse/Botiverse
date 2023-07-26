from gensim.utils import tokenize
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

from botiverse.preprocessors import GloVe, TF_IDF

class TF_IDF_GLOVE():
    '''
    An interface for transforming sentences into idf-glove vectors by weighting word GloVe vectors by their tf-idf.
    '''
    def __init__(self, force_download=False):
        '''
        Initialize the GloVe and TF-IDF transformer and download the embeddings if needed.
        
        :param force_download: If True, download the embeddings even if they already exist.
        :type force_download: bool
        '''
        self.glove = GloVe(force_download)
        self.glove_dict = self.glove.glove_dict
        self.tf_idf = TF_IDF()
        #self.glove_dict = None
        self.tf = None
        self.idf = None
        self.all_words = None


    def transform_list(self, sentence_list, all_words):
        '''
        Given a list of tokenized sentences, return a table of idf-GloVe vectors (one for each sentence) in the form of a numpy array.
        This also initializes the tf and idf tables of the class for use in the transform() method.
        
        :param sentence_list: A list of tokenized sentences
        :type sentence_list: list
        :param all_words: A list of all the words in the corpus
        :type all_words: list
        
        :return: A 2D numpy array of idf-GloVe vectors 
        :rtype: numpy.ndarray
        '''
        self.all_words = all_words
        # just to set tf and idf
        self.tf_idf.transform_list(sentence_list, all_words)
        self.tf, self.idf = self.tf_idf.tf, self.tf_idf.idf
        # make a numpy array of the sentence vectors
        X = np.zeros((len(sentence_list), 50))
        for i, sentence in enumerate(sentence_list):
                X[i] = self.transform(sentence)
        return X

    def transform(self, sentence):
        '''
        Given a sentence, return its idf-GloVe vector as a numpy array by weighting the GloVe vectors of the words in the sentence by their idf then averaging.
        
        :param sentence: A string of words
        :type sentence: str
        
        :return: A numpy array of the idf-GloVe vector
        :rtype: numpy.ndarray
        
        '''
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
