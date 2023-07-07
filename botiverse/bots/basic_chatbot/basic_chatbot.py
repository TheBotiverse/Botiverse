import numpy as np
import json
from gensim.utils import tokenize
import numpy as np
from botiverse.models import SVM, NeuralNet
from botiverse.preprocessors import GloVe, TF_IDF, TF_IDF_GLOVE
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


class basic_chatbot:
    '''
    An interface for a basic chatbot model suitable for small datasets such as FAQs. Note that the
    underlying model is not sequential (either an NN or an SVM).
    '''
    def __init__(self, machine='NN', repr='tf-idf'):
        """
        Instantiate a basic chat bot model that uses a classic feedforward neural network.
        Data can be then used to train the chatbot model.
        
        :param name: The chatbot's name.
        :type name: string
        :param machine: The machine learning model to use. Either 'NN' or 'SVM', else will assume to be provided in fit.
        :type machine: string
        :param repr: The representation to use. Either 'glove', 'tf-idf', or 'tf-idf-glove'.
        :type repr: string
        """
        self.model = None
        self.machine = machine
        self.repr = repr
        
        # if machine or transform is not a string, then assume it is a model
        if type(machine) != str: self.model = machine
        if type(repr) != str: self.transformer = repr
        
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
        for intent in self.raw_data:                    #this is a list of dictionaries. each has a tag (class), list of patterns and list of responses.
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

    def train(self, path, max_epochs=None, early_stop=False, **kwargs):
        """
        Train the chatbot model with the given JSON data.
        
        :param data: A stringfied JSON object containing the training data 
        :type number: string
        :param early_stop: Whether to use early stopping or not
        :type early_stop: bool
        :param provided_model: A model to use instead of the default one
        :type provided_model: Object
        :param provided_params: A dictionary of parameters to use instead of the default ones
        :type provided_params: dict
    
        :return: None
        :rtype: NoneType
        """
        with open(path, 'r') as f:
            self.raw_data = json.load(f) 

        X, y = self.setup_data()
            
        if self.machine == 'NN':
            self.model = NeuralNet(structure=[X.shape[1], 12, len(self.classes)], activation='sigmoid')
            max_epochs = max_epochs if max_epochs is not None else 50 * len(self.classes)
            if early_stop:
                self.model.fit(X, y, batch_size=1, epochs=max_epochs, λ = 0.02, eval_train=True, val_split=0.2, patience=100)
                self.model.fit(X, y, batch_size=1, epochs=max_epochs, λ = 0.02, eval_train=True, val_split=0.0)
            else:
                self.model.fit(X, y, batch_size=1, epochs=max_epochs, λ = 0.02, eval_train=True, val_split=0.0)
                
        elif self.machine == 'SVM':
            self.model = SVM(kernel='linear', C=700)
            self.model.fit(X, y, eval_train=True)
        else:
                self.model.fit(X, y, **kwargs)


    def save(self, path):
        '''
        Save the model to a file.
        :param path: The path to the file
        '''
        if self.machine == 'SVM':
            print("Could Not Save: SVM model is for experimentation only and does not allow saving yet.")
        else:
            self.model.save(path+'.bot')
    
    def load(self, load_path, data_path):
        '''
        Load the model from a file.
        :param path: The path to the file
        '''
        if self.machine == 'SVM':
            print("Could Not Load: SVM model is for experimentation only and does not allow loading yet.")
        else:
            self.model = NeuralNet.load(load_path + '.bot')
            with open(data_path, 'r') as f:
                self.raw_data = json.load(f)
                self.classes = sorted(set([intent['tag'] for intent in self.raw_data]))
                
    def infer(self, prompt, confidence=None):
        """
        Infer a suitable response to the given prompt.
        
        :param promp: The user's prompt
        :type number: string
    
        :return: The chatbot's response
        :rtype: string
        """
        if confidence is None: confidence = 2/len(self.classes)
        vector = self.transformer.transform(prompt) 
        # predict the class of the prompt
        tag_idx, tag_prob = self.model.predict(vector)
        tag_idx, tag_prob = tag_idx[0], tag_prob[0]
        tag = self.classes[tag_idx]
        if tag_prob < confidence: return "Could you rephrase that?"
        for intent in self.raw_data:
            if tag == intent["tag"]:
                return np.random.choice(intent['responses'])