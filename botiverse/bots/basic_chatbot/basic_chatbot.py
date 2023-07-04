import numpy as np
import json
from gensim.utils import tokenize
import numpy as np
from botiverse.models import SVM, NeuralNet
from botiverse.preprocessors import GloVe, TF_IDF, TF_IDF_GLOVE
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

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
        
        if repr == 'glove': 
            self.transformer = GloVe()
        if repr == 'tf-idf':
            self.transformer = TF_IDF()
        if repr == 'tf-idf-glove':
            self.transformer = TF_IDF_GLOVE()
            
        self.tf = None
        self.idf = None
        self.classes = None
        
        

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

        X = self.transformer.transform_list(sentence_list, all_words=all_words)

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
        
    def infer(self, prompt, confidence=None):
        """
        Infer a suitable response to the given prompt.
        
        :param promp: The user's prompt
        :type number: string
    
        :return: The chatbot's response
        :rtype: string
        """
        if confidence is None: confidence = 1.5/len(self.classes)
        vector = self.transformer.transform(prompt) 
        # predict the class of the prompt
        tag_idx, tag_prob = self.model.predict(vector)
        tag_idx, tag_prob = tag_idx[0], tag_prob[0]
        tag = self.classes[tag_idx]
        if tag_prob < confidence: return "Could you rephrase that?"
        for intent in self.raw_data['FAQ']:
            if tag == intent["tag"]:
                return np.random.choice(intent['responses'])