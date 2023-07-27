"""
This Module has base code and interfaces for TripPy Dialogue State Tracker.
"""

import torch
from collections import OrderedDict
import gdown
import os

from botiverse.models.TRIPPY.utils import normalize, mask_utterance
from botiverse.models.TRIPPY.data import fix_slot_list, prepare_data, Dataset
from botiverse.models.TRIPPY.run import run
from botiverse.models.TRIPPY.infer import infer
from botiverse.models.TRIPPY.evaluate import eval
from botiverse.models.TRIPPY.config import TRIPPYConfig
from botiverse.models.TRIPPY.TRIPPY import TRIPPY


class TRIPPYDST:
    """
    TRIPPYDST is a class that represents the TripPy Dialogue State Tracker.

    It provides methods for loading the model, training the model, updating the dialogue state,
    getting the current dialogue state, deleting slots, resetting the tracker, and displaying the tracker information.
    
    :param domains: The list of domains to consider.
    :type domains: list[str]
    :param slot_list: List of slot names.
    :type slot_list: list[str]
    :param label_maps: Dictionary of the variants of the slot-values that are mapped to the canonical slot-values.
    :type label_maps: dict[str, list]
    :param non_referable_slots: The list of non-referable slots.
    :type non_referable_slots: list[str]
    :param non_referable_pairs: The list of non-referable slot pairs.
    :type non_referable_pairs: list[tuple[str, str]]
    :param from_scratch: Whether to train the model from scratch.
    :type from_scratch: bool
    :param TRIPPY_config: The configuration for the TRIPPY model, defaults to TRIPPYConfig()
    :type TRIPPY_config: TRIPPYConfig, optional
    """

    def __init__(self, domains, slot_list, label_maps, non_referable_slots, non_referable_pairs, from_scratch, BERT_config, TRIPPY_config=TRIPPYConfig()):
        self.domains = domains
        self.non_referable_slots = non_referable_slots
        self.non_referable_pairs = non_referable_pairs
        self.from_scratch = from_scratch
        self.BERT_config = BERT_config
        self.TRIPPY_config = TRIPPY_config

        slot_list = fix_slot_list(slot_list, domains)
        self.slot_list = slot_list
        self.n_slots = len(slot_list)
        self.label_maps = label_maps

        self.state = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TRIPPY(len(slot_list), TRIPPY_config.hid_dim, TRIPPY_config.n_oper, TRIPPY_config.dropout, from_scratch, BERT_config, TRIPPY_config).to(self.device)
        self.history = []

    def save_model(self, model_path):
        """
        Save the trained model.

        :param model_path: The path to save the model.
        :type model_path: str
        """
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, model_path, pretrained, test_path):
      """
      Load the trained model if path is given, else load a pretrained model.

      :param model_path: The path to the saved model.
      :type model_path: str
      :param test_path: The path to the test data for evaluation.
      :type test_path: str
      """

      # If model_path is None will the load a pretrained model.
      if model_path is None and pretrained == 'sim-R':
        # Download DST weights trained on sim-R
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(curr_dir, 'sim-R.pt')
        
        f_id = '1POjBULmqxBrebvINfl989bskAstV3Zld'
        file_url = f'https://drive.google.com/uc?export=download&confirm=pbef&id={f_id}'
        if not os.path.exists(model_path):
            gdown.download(file_url, model_path)
            print('Model downloaded successfully.')
        else:
            print('Model already exists. Skipping download.')

      if self.from_scratch == True:
          # Get saved weights
          state_dict = torch.load(model_path, map_location=self.device)

          # Delete position_ids from the state_dict if available
          if 'bert.embeddings.position_ids' in state_dict.keys():
              del state_dict['bert.embeddings.position_ids']

          # Get the new weights keys from the model
          new_keys = list(self.model.state_dict().keys())

          # Get the weights from the state_dict
          old_keys = list(state_dict.keys())
          weights = list(state_dict.values())

          # Create a new state_dict with the new keys
          new_state_dict = OrderedDict()
          for i in range(len(new_keys)):
              new_state_dict[new_keys[i]] = weights[i]
              # print(old_keys[i], '->', new_keys[i])

          self.model.load_state_dict(new_state_dict)

      else:
          self.model.load_state_dict(torch.load(model_path, map_location=self.device))

      print('Model loaded successfully.')
      if test_path is not None:
        print('Preprocessing the data...')
        test_raw_data, test_data = prepare_data(test_path, self.slot_list, self.label_maps, self.TRIPPY_config.tokenizer, self.TRIPPY_config.max_len, self.domains, self.non_referable_slots, self.non_referable_pairs, self.TRIPPY_config.multiwoz)
        test_dataset = Dataset(test_data, self.n_slots, self.TRIPPY_config.oper2id, self.slot_list)
        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=self.TRIPPY_config.test_batch_size)
        print('Evaluating the model on the data...')
        joint_goal_acc, per_slot_acc, macro_f1_score, all_f1_score = eval(test_raw_data, test_data, self.model, self.device, self.n_slots, self.slot_list, self.label_maps, self.TRIPPY_config.oper2id, self.TRIPPY_config.multiwoz)
        print(f'Joint Goal Acc: {joint_goal_acc}')
        print(f'Per Slot Acc: {per_slot_acc}')
        print(f'Macro F1 Score: {macro_f1_score}')
        print(f'All f1 score = {all_f1_score}')


    def train(self, train_path, dev_path=None, test_path=None):
      """
      Train the model.

      :param train_path: The path to the training data.
      :type train_path: str
      :param dev_path: The path to the development data for evaluation during training.
      :type dev_path: str
      :param test_path: The path to the test data for evaluation after training.
      :type test_path: str
      """

      # Save the temprorary model in the current directory.
      curr_dir = os.path.dirname(os.path.abspath(__file__))
      model_path = os.path.join(curr_dir, 'model.pt')

      # Train the model
      run(self.model, self.domains, self.slot_list, self.label_maps, train_path, dev_path, test_path, self.device, self.non_referable_slots, self.non_referable_pairs, model_path, self.TRIPPY_config)


    def update_state(self, sys_utter, user_utter, inform_mem):
      """
      Update the dialogue state based on the system and user utterances.

      :param sys_utter: The system utterance.
      :type sys_utter: str
      :param user_utter: The user utterance.
      :type user_utter: str
      :param inform_mem: The inform memory containing previous slot-value pairs.
      :type inform_mem: dict[str, list[str]]
      :return: The updated dialogue state.
      :rtype: dict[str, str]
      """

      # normalize utterances
      user_utter = ' '.join(normalize(user_utter, self.TRIPPY_config.multiwoz))
      sys_utter = ' '.join(normalize(sys_utter, self.TRIPPY_config.multiwoz))
      # delex the system utterance
      sys_utter = ' '.join(mask_utterance(sys_utter, inform_mem, self.TRIPPY_config.multiwoz, '[UNK]'))

      self.state = infer(self.model, self.slot_list, self.state, self.history, sys_utter, user_utter, inform_mem, self.device, self.TRIPPY_config.oper2id, self.TRIPPY_config.tokenizer, self.TRIPPY_config.max_len)
      self.history = [user_utter, sys_utter] + self.history
      return self.state.copy()

    def get_dialogue_state(self):
      """
      Get a copy of the current dialogue state.

      :return: A copy of the dialogue state.
      :rtype: dict[str, str]
      """

      state = self.state.copy()

      # if §§ is in the state, then remove it
      for key in state:
        if '§§' in state[key]:
          state[key] = state[key].replace('§§', '')
        if '§§ ' in state[key]:
          state[key] = state[key].replace('§§ ', '')

      return state

    def is_all_slots_filled(self):
      """
      Check if all slots are filled.

      :return: True if all slots are filled, False otherwise.
      :rtype: bool
      """
      return all([slot in self.state.keys() for slot in self.slot_list])

    def delete_slots(self, domain, slot):
      """
      Delete slots from the dialogue state.

      If a domain is specified, all slots in that domain will be deleted.
      If a slot is specified, that specific slot will be deleted.
      If neither domain nor slot is specified, all slots will be deleted.

      :param domain: The domain to delete slots from.
      :type domain: str
      :param slot: The slot to delete.
      :type slot: str
      """
      keys = self.state.keys()
      if domain is not None:
        for key in keys:
          if domain in key:
            del self.state[key]
      elif slot is not None:
          if slot in keys:
            del self.state[slot]
      else:
        for key in keys:
          del self.state[key]

    def reset(self):
      """
      Reset the dialogue state.

      Remove all slots from the dialogue state and clear the history.
      """
      keys = list(self.state.keys())
      for key in keys:
        del self.state[key]

      self.history = []


    def __str__(self):
      """
      Return a string representation of the TRIPPYDST object.

      :return: A string representation of the object.
      :rtype: str
      """
      string = ''
      string = string + '\ndomains: ' + str(self.domains)
      string = string + '\nslot_list: ' + str(self.slot_list)
      string = string + '\nlabel_maps: ' + str(self.label_maps)
      string = string + '\nnon_referable_slots: ' + str(self.non_referable_slots)
      string = string + '\nnon_referable_pairs: ' + str(self.non_referable_pairs)
      string = string + '\nfrom_scratch: ' + str(self.from_scratch)
      string = string + '\nslot_list: ' + str(self.slot_list)
      string = string + '\nn_slots: ' + str(self.n_slots)
      string = string + '\nlabel_maps: ' + str(self.label_maps)
      string = string + '\nstate: ' + str(self.state)
      string = string + '\ndevice: ' + str(self.device)
      # string = string + '\nmodel: ' + str(self.model)
      string = string + '\nhistory: ' + str(self.history)
      return string