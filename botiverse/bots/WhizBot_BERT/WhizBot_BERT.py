import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
from torch import nn,optim
from botiverse.models.LinearClassifier.LinearClassifier import LinearClassifier
from botiverse.preprocessors.WhizBot_BERT_Preprocessor.WhizBot_BERT_Preprocessor import WhizBot_BERT_Preprocessor
import random

class WhizBot_GRU:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def read_data(self, file_path):
        # read data
        self.preprocessor = WhizBot_BERT_Preprocessor(file_path)
        # process data
        self.data = self.preprocessor.process()
        num_labels = len(self.preprocessor.label_dict)
        # prepare model
        self.model = LinearClassifier(768, num_labels).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        # train test split
        self.train_data = self.data.sample(frac=0.8, random_state=42)
        self.test_data = self.data.drop(self.train_data.index)
        self.train_data = self.train_data.reset_index(drop=True)
        self.test_data = self.test_data.reset_index(drop=True)

    def train(self, epochs=10, batch_size=32):
        self.model.train()
        for epoch in range(epochs):
            for i in tqdm(range(0, len(self.train_data), batch_size)):
                self.model.zero_grad()
                batch_texts = torch.cat(self.train_data['text'][i:i+batch_size].tolist()).to(self.device)
                batch_labels = torch.cat(self.train_data['label'][i:i+batch_size].tolist()).to(self.device)
                output = self.model(batch_texts)
                loss = self.criterion(output, batch_labels)
                loss.backward()
                self.optimizer.step()
            print("Epoch: " + str(epoch) + " Loss: " + str(loss.item()))

    def test(self, batch_size=32):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(self.test_data), batch_size)):
                batch_texts = torch.cat(self.test_data['text'][i:i+batch_size].tolist()).to(self.device)
                batch_labels = torch.cat(self.test_data['label'][i:i+batch_size].tolist()).to(self.device)
                outputs = self.model(batch_texts)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        print('Accuracy: %d %%' % (100 * correct / total))

    def infer(self, string):
        self.model.eval()
        with torch.no_grad():
            string = self.preprocessor.process_string(string).to(self.device)
            output = self.model(string)
            _, predicted = torch.max(output.data, 1)
            for key, value in self.preprocessor.label_dict.items():
                if value == predicted.item():
                    label = key
                    break
        return random.choice(self.preprocessor.responces[label])

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))