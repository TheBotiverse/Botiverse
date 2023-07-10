import torch
from transformers import AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
import torch.optim as optim
from botiverse.preprocessors.ConverseBot_Preprocessor.ConverseBot_Preprocessor import ConverseBot_Preprocessor
from botiverse.models.T5Model.T5Model import T5Model

class ConverseBot:
    def __init__(self, dataset=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # create a model instance
        self.model = T5Model()
        # load the Backenf finetuning parameters
        self.model.load_state_dict(AutoModelForSeq2SeqLM.from_pretrained("MohamedSaad/T5_ConverseBot").state_dict())
        # move the model to the GPU if available
        self.model.to(self.device)
        # load the preprocessor
        self.preprocessor = ConverseBot_Preprocessor(dataset)
        # process the dataset if it exists
        if dataset is not None:
          # preprocess the dataset
          self.data = self.preprocessor.process()
          # train validation split
          self.train_data = self.data.sample(frac=0.99, random_state=0)
          self.validation_data = self.data.drop(self.train_data.index)
          self.train_data = self.train_data.reset_index(drop=True)
          self.validation_data = self.validation_data.reset_index(drop=True)

    def train(self, epochs=1, batch_size=32):
        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00005)
        for epoch in range(epochs):
            for i in tqdm(range(0, len(self.train_data), batch_size)):
                self.model.zero_grad()
                # prepare the training batches
                batch_text_input_ids = torch.concat(self.train_data['text_input_ids'][i:i+batch_size].tolist()).to(self.device)
                batch_text_attention_mask = torch.concat(self.train_data['text_attention_mask'][i:i+batch_size].tolist()).to(self.device)
                batch_labels = torch.concat(self.train_data['target'][i:i+batch_size].tolist()).to(self.device)
                loss = self.model(input_ids=batch_text_input_ids, attention_mask=batch_text_attention_mask, labels=batch_labels).loss
                loss.backward()
                self.optimizer.step()
            print("Epoch: " + str(epoch) + " Loss: " + str(loss.item()))

    def validation(self, batch_size=32):
        total = 0
        loss = 0
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(self.validation_data), batch_size)):
                # prepare the validation batches
                batch_text_input_ids = torch.concat(self.validation_data['text_input_ids'][i:i+batch_size].tolist()).to(self.device)
                batch_text_attention_mask = torch.concat(self.validation_data['text_attention_mask'][i:i+batch_size].tolist()).to(self.device)
                batch_labels = torch.concat(self.validation_data['target'][i:i+batch_size].tolist()).to(self.device)
                outputs = self.model(input_ids=batch_text_input_ids, attention_mask=batch_text_attention_mask, labels=batch_labels)
                loss += outputs.loss.item()
                total += batch_labels.size(0)
        print('Validation Loss: ', loss/total)

    def infer(self, string):
        self.model.eval()
        token_obj = self.preprocessor.process_string(string)
        input_ids= token_obj['input_ids'].to(self.device)
        attention_mask = token_obj['attention_mask'].to(self.device)
        output_tokens = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=250)
        return self.preprocessor.decode_tokens(output_tokens)

    # save a model locally
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    # load a model from a local path
    def load(self, path):
        self.model.load_state_dict(torch.load(path))