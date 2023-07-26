from gensim.utils import tokenize
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


class BoW():
    '''
    An interface for transforming sentences into bag-of-words vectors.
    '''
    def __init__(self, binary=False):
        '''
        Initialize the BoW transformer.
        
        :param binary: Whether to use binary BoW vectors instead of frequency BoW vectors.
        :type binary: bool
        '''
        
        self.all_words = None
        self.binary = binary
        
    def transform_list(self, sentence_list, all_words):
        '''
        Given a list of tokenized sentences, return a table of BoW vectors (one for each sentence) in the form of a numpy array.
        
        :param sentence_list: A list of tokenized sentences.
        :type sentence_list: list
        :param all_words: A list of all the words in the vocabulary.
        :type all_words: list
        
        :return: A table of BoW vectors (one for each sentence) in the form of a numpy array.
        :rtype: numpy.ndarray
        '''
        self.all_words = all_words
        BoWs = np.zeros((len(sentence_list), len(all_words)), dtype=np.float64)
        for i, sentence in enumerate(sentence_list):
            sentence = list(tokenize(sentence, to_lower=True))
            sentence = [stemmer.stem(word.lower()) for word in sentence if word not in ['?', '!', '.', ',']]
            for word in sentence:
                word_index = all_words.index(word)
                BoWs[i, word_index] += 1 if not self.binary else 1
        return BoWs
    
    def transform(self, sentence):
        '''
        Given a sentence, return its BoW vector as a numpy array.
        
        :param sentence: A string of words
        :type sentence: str
        
        :return: A BoW vector for the given sentence.
        :rtype: numpy.ndarray
        '''
        sentence = list(tokenize(sentence, to_lower=True))
        sentence = [stemmer.stem(word.lower()) for word in sentence if word not in ['?', '!', '.', ',']]
        BoW = np.zeros((1, len(self.all_words)), dtype=np.float64)
        for word in sentence:
            if word not in self.all_words:  continue
            word_index = self.all_words.index(word)
            BoW[0, word_index] += 1 if not self.binary else 1
        return BoW