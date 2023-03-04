from botiverse.TODS.DNN_DST.model import DSTModel
from botiverse.TODS.DNN_DST.config import *
from sklearn.model_selection import train_test_split
import torch
from botiverse.TODS.DNN_DST.evaluate import eval_f1_jac, eval_joint
from botiverse.TODS.DNN_DST.infer import infer
from botiverse.TODS.DNN_DST.run import run
from botiverse.TODS.DNN_DST.data import prepare_data, Dataset

class DNNDST:

    def __init__(self, domains, slot_list, label_maps):
        self.domains = domains
        self.slot_list = slot_list
        self.n_slots = len(slot_list)
        self.label_maps = label_maps
        self.state = {slot:None for slot in slot_list}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DSTModel(len(slot_list), HID_DIM, N_OPER, DROPOUT).to(self.device)
        self.history = []

    def load_model(self, model_path, test_json=None):
      self.model.load_state_dict(torch.load(model_path, map_location=self.device))
      print('Model loaded successfully.')
      if test_json is not None:
        print('Testing the model...')
        # test
        print('Preprocessing test set...')
        test_raw_data, test_data = prepare_data(test_json, self.slot_list, self.label_maps, TOKENIZER, MAX_LEN, self.domains)
        test_dataset = Dataset(test_data, self.n_slots, OPER2ID)
        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=TEST_BATCH_SIZE)
        print('Evaluating the model on test set...')
        jaccard_score, macro_f1_score, all_f1_score = eval_f1_jac(test_data_loader, self.model, self.device, self.n_slots)
        joint_goal_acc, states, sentences, indices = eval_joint(test_raw_data, test_data, self.model, self.device, self.n_slots, self.slot_list, self.label_maps)
        print(f'Joint Goal Acc: {joint_goal_acc}, Jaccard Score: {jaccard_score}, Macro F1 Score: {macro_f1_score}')
        print(f'All f1 score = {all_f1_score}')

    def train(self, train_json, dev_json=None, test_json=None):
      run(self.model, self.domains, self.slot_list, self.label_maps, train_json, dev_json, test_json, self.device)

    def update_state(self, sys_utter, user_utter):
      self.state = infer(self.model, self.slot_list, self.state, self.history, sys_utter, user_utter, self.device)
      self.history = [user_utter, sys_utter] + self.history

    def get_dialogue_state(self):
      return self.state

    def reset(self, slots=None):      
      if slots is None:
        slots = self.slot_list
      for slot in self.slots:
        self.state[slot] = None
    