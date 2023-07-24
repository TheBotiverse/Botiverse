import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from botiverse.models.GRUClassifier.GRUClassifier import GRUTextClassifier
from botiverse.preprocessors.Special.WhizBot_GRU_Preprocessor.WhizBot_GRU_Preprocessor import WhizBot_GRU_Preprocessor
import random

class WhizBot_GRU:
    '''An interface for the WhizBot_GRU model which is based on simple GRU model'''
    def __init__(self):
        """
        Initializes WhizBot_GRU, and will prepare the GPU device based on CUDA availability.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def read_data(self, file_path):
        """
        Reads and pre-processes the data, sets up the model based on the data and prepares the train-validation split.

        :param file_path: The path to the file that contains the dataset.
        :type file_path: str

        :returns: None
        """
        # read data
        self.preprocessor = WhizBot_GRU_Preprocessor(file_path)
        # process data
        self.data = self.preprocessor.process()
        # prepare model
        vocab_size = len(self.preprocessor.tokenizer.get_vocab())
        num_labels = len(self.preprocessor.label_dict)
        self.model = GRUTextClassifier(vocab_size, 300, num_labels).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.05)
        # train validation split
        self.train_data = self.data.sample(frac=0.8, random_state=0)
        self.validation_data = self.data.drop(self.train_data.index)
        self.train_data = self.train_data.reset_index(drop=True)
        self.validation_data = self.validation_data.reset_index(drop=True)

    def train(self, epochs=50, batch_size=32):
        """
        Trains the model using the training dataset.

        :param epochs: The number of training epochs.
        :type epochs: int

        :param batch_size: The number of training examples utilized used to make one paramenters updat.
        :type batch_size: int

        :returns: None
        """
        self.model.train()
        for epoch in range(epochs):
            for i in tqdm(range(0, len(self.train_data), batch_size)):
                self.model.zero_grad()
                # get the batches
                batch_texts = torch.stack(self.train_data['text'][i:i+batch_size].tolist()).to(self.device)
                batch_labels = torch.cat(self.train_data['label'][i:i+batch_size].tolist()).to(self.device)
                output = self.model(batch_texts)
                loss = self.criterion(output, batch_labels)
                # backpropagate
                loss.backward()
                self.optimizer.step()
            print("Epoch: " + str(epoch) + " Loss: " + str(loss.item()))

    def validation(self, batch_size=32):
        """
        Tests the model performance using the validation dataset and calculates the accuracy.
        
        :param batch_size: The number of training examples utilized used to make one paramenters updat.
        :type batch_size: int

        :returns: None
        """
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(self.validation_data), batch_size)):
                # get the batches
                batch_texts = torch.stack(self.validation_data['text'][i:i+batch_size].tolist()).to(self.device)
                batch_labels = torch.cat(self.validation_data['label'][i:i+batch_size].tolist()).to(self.device)
                outputs = self.model(batch_texts)
                _, predicted = torch.max(outputs.data, 1)
                # calculate accuracy
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        print('Accuracy: %d %%' % (100 * correct / total))

    def infer(self, string):
        """
        Performs inference using the model.

        :param string: The input string to perform inference on.
        :type string: str

        :returns: A random response from the response list of the predicted label.
        """
        self.model.eval()
        with torch.no_grad():
            string = self.preprocessor.process_string(string)
            string = string.unsqueeze(0).to(self.device)
            output = self.model(string)
            _, predicted = torch.max(output.data, 1)
            # get the label
            for key, value in self.preprocessor.label_dict.items():
                if value == predicted.item():
                    label = key
                    break
        # return a random responce of the label
        return random.choice(self.preprocessor.responces[label])

    def save(self, path):
        """
        Saves the model parameters to the given path.

        :param path: The path where the model parameters will be saved.
        :type path: str

        :returns: None
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """
        Loads the model parameters from the given path.

        :param path: The path from where the model parameters will be loaded.
        :type path: str

        :returns: None
        """
        self.model.load_state_dict(torch.load(path))