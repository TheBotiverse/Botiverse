import os
import gdown
from gensim.utils import tokenize
import numpy as np


class GloVe():
    '''
    An interface for transforming words into GloVe vectors.
    '''
    def __init__(self, force_download=False):
        '''
        Initialize the GloVe transformer and download the embeddings if needed.
        
        :param force_download: If True, download the embeddings even if they already exist.
        :type force_download: bool
        '''
        self.glove_dict = None
        self.load_glove_vectors(force_download=force_download)

    # Load GLOVE word vectors
    def load_glove_vectors(self, force_download):
        '''
        Load GloVe vectors from gensim into the class. By default uses 50D vectors.
        
        :param force_download: If True, download the embeddings even if they already exist.
        :type force_download: bool
        '''
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


    def transform_list(self, sentence_list, **kwargs):
        '''
        Transform a sentence list into a numpy array of GloVe vectors
        
        :param sentence_list: A list of sentences
        :type sentence_list: list
        
        :return: A numpy array of GloVe vectors
        :rtype: numpy.ndarray
        '''
        # make a numpy array of the sentence vectors
        X = np.zeros((len(sentence_list), 50))
        for i, sentence in enumerate(sentence_list):
                X[i] = self.transform(sentence)
        return X
    
    def transform(self, sentence):
        '''
        Transform a sentence into a GloVe vector by averaging the word vectors in it.
        
        :param sentence: a string of words
        :type sentence: str
        
        :return: the corresponding GloVe vector
        :rtype: numpy.ndarray
        '''
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