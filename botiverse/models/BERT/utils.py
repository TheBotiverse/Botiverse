"""
This module contains utility functions for the BERT model.
"""

from collections import OrderedDict
from transformers import BertModel
from botiverse.models.BERT.config import BERTConfig
from botiverse.models.BERT.BERT import Bert


# Load pre-trained weights from trnasformers library
def LoadPretrainedWeights(model):
    """
    Load pre-trained weights from the transformers library.

    This function loads the pre-trained weights from the transformers library
    and updates the model's state_dict accordingly.

    :param model: The BERT model.
    :type model: Bert
    """

    # Get pre-trained weights from transformers library
    pretrained_model = BertModel.from_pretrained('bert-base-uncased')
    state_dict = pretrained_model.state_dict()

    # Delete position_ids from the state_dict if available
    if 'embeddings.position_ids' in state_dict.keys():
        del state_dict['embeddings.position_ids']

    # Get the new weights keys from the model
    new_keys = list(model.state_dict().keys())

    # Get the weights from the state_dict
    old_keys = list(state_dict.keys())
    weights = list(state_dict.values())

    # Create a new state_dict with the new keys
    new_state_dict = OrderedDict()
    for i in range(len(new_keys)):
        new_state_dict[new_keys[i]] = weights[i]
        # print(old_keys[i], '->', new_keys[i])

    model.load_state_dict(new_state_dict)

# Example comparing the outputs of the from scratch model to the pre-trained model from transformers library
import torch
from transformers import BertModel, BertTokenizer

def Example():
    """
    Example comparing the outputs of the from scratch model to the pre-trained model from transformers library.
    """
    
    # Build a BERT model from scratch
    config = BERTConfig()
    model = Bert(config)
    LoadPretrainedWeights(model)

    # Load pre-trained weights from the Transformers library
    pretrained_weights = 'bert-base-uncased'
    pretrained_model = BertModel.from_pretrained(pretrained_weights)

    # Tokenize the input sequence
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    input_text = ["This is a sample input sequence.", "batyousef is awesome"]
    inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')

    # Set dropout to zero during inference
    model.eval()
    pretrained_model.eval()

    ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']

    # Pass the inputs through both models and compare the outputs
    with torch.no_grad():
        model_output1, model_output2 = model(ids, token_type_ids, attention_mask)
        pretrained_output1, pretrained_output2 = pretrained_model(ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids,
                                            return_dict=False
                                            )

    print(model_output1.size(), model_output1)
    print(pretrained_output1.size(), pretrained_output1)
    print()
    print()
    print(model_output2.size(), model_output2)
    print(pretrained_output2.size(), pretrained_output2)

    print(model)
    print(pretrained_model)