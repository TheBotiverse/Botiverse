from botiverse.bots.WhizBot.WhizBot_GRU import WhizBot_GRU
from botiverse.bots.WhizBot.WhizBot_BERT import WhizBot_BERT

class WhizBot:
    '''
    A class that provides an interface for the WhizBot-BERT and WhizBot-GRU models.
    '''
    def __init__(self, repr='BERT'):
        """
        Initializes WhizBot and sets its representation type.
        :param repr: The representation type of the WhizBot model. Either "BERT" or "GRU".
        :type repr: str
        """
        if repr == 'BERT':
            self.bot = WhizBot_BERT()
        elif repr == 'GRU':
            self.bot = WhizBot_GRU()
        else:
            raise ValueError('Invalid representation type for WhizBot. Please choose either "BERT" or "GRU".')
    
    
    
    def read_data(self, file_path):
        """
        Reads and pre-processes the data, sets up the model based on the data and prepares the train-validation split.

        :param file_path: The path to the file that contains the dataset.
        :type file_path: str

        :returns: None
        """
        self.bot.read_data(file_path)

    def train(self, epochs=10, batch_size=32):
        """
        Trains the model using the training dataset.

        :param epochs: The number of training epochs.
        :type epochs: int

        :param batch_size: The number of training examples utilized used to make one paramenters updat.
        :type batch_size: int

        :returns: None
        """
        self.bot.train(epochs, batch_size)

    def validation(self, batch_size=32):
        """
        Tests the model performance using the validation dataset and calculates the accuracy.
        
        :param batch_size: The number of training examples utilized used to make one paramenters updat.
        :type batch_size: int

        :returns: None
        """
        self.bot.validation(batch_size)

    def infer(self, string):
        """
        Performs inference using the model.

        :param string: The input string to perform inference on.
        :type string: str

        :returns: A random response from the response list of the predicted label.
        """
        return self.bot.infer(string)

    def save(self, path):
        """
        Saves the model parameters to the given path.

        :param path: The path where the model parameters will be saved.
        :type path: str

        :returns: None
        """
        self.bot.save(path)

    def load(self, path):
        """
        Loads the model parameters from the given path.

        :param path: The path from where the model parameters will be loaded.
        :type path: str

        :returns: None
        """
        self.bot.load(path)