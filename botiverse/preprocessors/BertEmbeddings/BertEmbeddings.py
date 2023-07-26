from transformers import BertTokenizer, BertModel
from botiverse.models.BERT.config import BERTConfig
from botiverse.models.BERT.utils import LoadPretrainedWeights
from botiverse.models import Bert
import torch
import numpy as np


#out = BertEmbedder().embedd(['hello world', 'hello world'])

class BertEmbedder():
    '''An interface for converting given text into BERT embeddings.'''
    def __init__(self):
        '''Load the pre-trained model and tokenizer'''
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = Bert(BERTConfig())
        LoadPretrainedWeights(self.bert)
        self.model = BertModel.from_pretrained('bert-base-uncased')
    
    def embed(self, sentences, random_state=42):
        '''
        Convert the given sentences into BERT embeddings.
        
        :param sentences: A list of sentences to convert into BERT embeddings.
        :type sentences: list
        :param random_state: The random state to use for reproducibility.
        :type random_state: int
        
        :return: A list of BERT embeddings for the given sentences.
        :rtype: list
        '''
        torch.manual_seed(random_state)
        tokss = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        emb, clst = self.bert(**tokss)
        attention_mask = tokss['attention_mask']
        # exclude padding tokens
        emb = emb * attention_mask.unsqueeze(-1)
        emb = emb.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        return emb.squeeze()

    
    def closest_sentence(self, new_sentence,  sentence_list, retun_ind=False):
        '''
        Given a list of sentences and a new sentence, return the sentence from the list that is closest to the new sentence.
        
        :param new_sentence: The new sentence to compare to the list of sentences.
        :type new_sentence: str
        :param sentence_list: A list of sentences to compare the new sentence to.
        :type sentence_list: list
        :param retun_ind: Whether to return the index of the closest sentence instead of the sentence itself.
        :type retun_ind: bool
        
        :return: The sentence from the list that is closest to the new sentence and its score.
        :rtype: str, float
        '''
        new_sentence_embedding = self.embed(new_sentence)
        sentence_list_embeddings = [self.embed(sentence) for sentence in sentence_list]
        cosine_sim = lambda x, y: torch.dot(x, y) / (torch.norm(x) * torch.norm(y))
        scores = [cosine_sim(new_sentence_embedding, sentence_embedding) for sentence_embedding in sentence_list_embeddings]
        softmax = lambda x: torch.exp(x) / torch.sum(torch.exp(x))
        scores = softmax(torch.tensor(scores)).detach().numpy()
        return sentence_list[np.argmax(scores)] if not retun_ind else np.argmax(scores), np.max(scores)
    
    
try:
    from sentence_transformers import SentenceTransformer, util
except:
    pass

class BertSentenceEmbedder():
    '''
    An interface for converting given text into sentence BERT embeddings.
    '''
    def __init__(self):
        '''
        Load the pre-trained model and tokenizer
        '''
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def embed(self, sentences):
        '''
        Convert the given sentences into BERT embeddings.
        
        :param sentences: A list of sentences to convert into BERT embeddings.
        :type sentences: list
        
        :return: A list of BERT embeddings for the given sentences.
        :rtype: list
        '''
        return self.model.encode(sentences, convert_to_tensor=True)
    
    def closest_sentence(self, new_sentence,  sentence_list, retun_ind=False):
        '''
        Given a list of sentences and a new sentence, return the sentence from the list that is closest to the new sentence.
        
        :param new_sentence: The new sentence to compare to the list of sentences.
        :type new_sentence: str
        :param sentence_list: A list of sentences to compare the new sentence to.
        :type sentence_list: list
        :param retun_ind: Whether to return the index of the closest sentence instead of the sentence itself.
        :type retun_ind: bool
        '''
        new_sentence_embedding = self.embed(new_sentence)
        sentence_list_embeddings = self.embed(sentence_list)
        cosine_sim = lambda x, y: torch.dot(x, y) / (torch.norm(x) * torch.norm(y))
        scores = [cosine_sim(new_sentence_embedding, sentence_embedding) for sentence_embedding in sentence_list_embeddings]
        softmax = lambda x: torch.exp(x) / torch.sum(torch.exp(x))
        scores = softmax(torch.tensor(scores)).detach().numpy()        
        return sentence_list[np.argmax(scores)] if not retun_ind else np.argmax(scores), np.max(scores)
