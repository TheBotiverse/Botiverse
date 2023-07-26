
from gensim.utils import tokenize
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

class TF_IDF():
    '''
    An interface for transforming sentences into tf-idf vectors.
    '''
    def __init__(self):
        self.tf = None
        self.idf = None
        self.all_words = None
    
    def transform_list(self, sentence_list, all_words):
        '''
        Given a list of tokenized sentences, return a table of tf-idf vectors (one for each sentence) in the form of a numpy array.
        
        :param sentence_list: A list of tokenized sentences
        :type sentence_list: list
        :param all_words: A list of all the words in the corpus
        :type all_words: list
        
        :return: A numpy array of tf-idf vectors
        :rtype: numpy.ndarray
        '''
        self.all_words = all_words
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
    

    def transform(self, sentence):
        '''
        Given a sentence, return its tf-idf vector as a numpy array.
        
        :param sentence: A string of words
        :type sentence: str
        
        :return: A numpy array of the tf-idf vector
        :rtype: numpy.ndarray
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
    
    
