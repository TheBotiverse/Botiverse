import torch
from collections import OrderedDict

from botiverse.models.TRIPPY.utils import normalize, mask_utterance
from botiverse.models.TRIPPY.data import get_ontology_label_maps, prepare_data, Dataset
from botiverse.models.TRIPPY.run import run
from botiverse.models.TRIPPY.infer import infer
from botiverse.models.TRIPPY.config import TRIPPYConfig
from botiverse.models.TRIPPY.TRIPPY import TRIPPY


class TRIPPYDST:

    def __init__(self, domains, ontology_path, label_maps_path, non_referable_slots, non_referable_pairs, from_scratch, TRIPPY_config=TRIPPYConfig()):
        self.domains = domains
        self.ontology_path = ontology_path
        self.label_maps_path = label_maps_path
        self.non_referable_slots = non_referable_slots
        self.non_referable_pairs = non_referable_pairs
        self.from_scratch = from_scratch
        self.TRIPPY_config = TRIPPY_config

        slot_list, label_maps = get_ontology_label_maps(ontology_path, label_maps_path, domains)
        self.slot_list = slot_list
        self.n_slots = len(slot_list)
        self.label_maps = label_maps
        self.state = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TRIPPY(len(slot_list), TRIPPY_config.hid_dim, TRIPPY_config.n_oper, TRIPPY_config.dropout, from_scratch).to(self.device)
        self.history = []

    def load_model(self, model_path, test_path):

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
              print(old_keys[i], '->', new_keys[i])

          self.model.load_state_dict(new_state_dict)

      else:
          self.model.load_state_dict(torch.load(model_path, map_location=self.device))

      print('Model loaded successfully.')
      if test_path is not None:
        print('Preprocessing the data...')
        test_raw_data, test_data = prepare_data(test_path, self.slot_list, self.label_maps, self.TRIPPY_config.tokenizer, self.TRIPPY_config.max_len, self.domains, self.non_referable_slots, self.non_referable_pairs)
        test_dataset = Dataset(test_data, self.n_slots, self.TRIPPY_config.oper2id, self.slot_list)
        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=TEST_BATCH_SIZE)
        print('Evaluating the model on the data...')
        joint_goal_acc, per_slot_acc, macro_f1_score, all_f1_score = eval(test_raw_data, test_data, self.model, self.device, self.n_slots, self.slot_list, self.label_maps, self.TRIPPY_config.oper2id)
        print(f'Joint Goal Acc: {joint_goal_acc}')
        print(f'Per Slot Acc: {per_slot_acc}')
        print(f'Macro F1 Score: {macro_f1_score}')
        print(f'All f1 score = {all_f1_score}')


    def train(self, train_path, dev_path, test_path, model_path):
      run(self.model, self.domains, self.slot_list, self.label_maps, train_path, dev_path, test_path, self.device, self.non_referable_slots, self.non_referable_pairs, model_path)

    def update_state(self, sys_utter, user_utter, inform_mem):
      # normalize utterances
      user_utter = ' '.join(normalize(user_utter))
      sys_utter = ' '.join(normalize(sys_utter))
      # delex the system utterance
      sys_utter = ' '.join(mask_utterance(sys_utter, inform_mem, '[UNK]'))

      self.state = infer(self.model, self.slot_list, self.state, self.history, sys_utter, user_utter, inform_mem, self.device, self.TRIPPY_config.oper2id, self.TRIPPY_config.tokenizer, self.TRIPPY_config.max_len)
      self.history = [user_utter, sys_utter] + self.history
      return self.state.copy()

    def get_dialogue_state(self):
      return self.state.copy()

    def delete_slots(self, domain, slot):
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
      keys = list(self.state.keys())
      for key in keys:
        del self.state[key]

      self.history = []


    def __str__(self):
      string = ''
      string = string + '\ndomains: ' + str(self.domains)
      string = string + '\nontology_path: ' + str(self.ontology_path)
      string = string + '\nlabel_maps_path: ' + str(self.label_maps_path)
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