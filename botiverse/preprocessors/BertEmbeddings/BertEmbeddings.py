from transformers import BertTokenizer, BertModel
import torch
import numpy as np

class BertEmbedder():
    '''An interface for converting given text into BERT embeddings.'''
    def __init__(self):
        '''Load the pre-trained model and tokenizer'''
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def embed(self, sentences):
        '''
        Convert the given sentence into a BERT embedding by passing it through the model.
        :param sentence: The sentence to be converted into a BERT embedding.
        '''
        emb = self.model(**self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)).last_hidden_state.mean(dim=1)
        return emb.squeeze()
    
    
    def closest_sentence(self, new_sentence,  sentence_list, retun_ind):
        '''
        Given a list of sentences and a new sentence, return the sentence from the list that is closest to the new sentence.
        :param new_sentence: The new sentence to compare to the list of sentences.
        :param sentence_list: A list of sentences to compare the new sentence to.
        :return: The sentence from the list that is closest to the new sentence and its score.
        '''
        new_sentence_embedding = self.embed(new_sentence)
        sentence_list_embeddings = self.embed(sentence_list)
        cosine_sim = lambda x, y: torch.dot(x, y) / (torch.norm(x) * torch.norm(y))
        scores = [cosine_sim(new_sentence_embedding, sentence_embedding) for sentence_embedding in sentence_list_embeddings]
        softmax = lambda x: torch.exp(x) / torch.sum(torch.exp(x))
        scores = softmax(torch.tensor(scores)).detach().numpy()
        return sentence_list[np.argmax(scores)] if not retun_ind else np.argmax(scores), np.max(scores)