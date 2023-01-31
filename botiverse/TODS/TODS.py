import numpy as np
import torch

class TODS:
    """
    Instantiate a basic chat bot model that uses a classic feedforward neural network.
    Data can be then used to train the chatbot model.
    
    :param name: The chatbot's name.
    :type name: string
    """
    
    def __init__(self, name):
        self.name = name
        self.model = lambda x: f"My name is {self.name} and I don't know how to respond to that."


    def train(self, data):
        """
        Train the chatbot model with the given JSON data.
        
        :param data: A stringfied JSON object containing the training data 
        :type number: string
    
        :return: None
        :rtype: NoneType
        """
        rubbish = data
    
    def infer(self, prompt):
        """
        Infer a suitable response to the given prompt.
        
        :param promp: The user's prompt
        :type number: string
    
        :return: The chatbot's response
        :rtype: string
        """
        response = self.model(prompt)
        return response
    
    

