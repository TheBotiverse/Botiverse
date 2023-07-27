"""
This module contains the base code and interface of deep Taskbot .
"""

from botiverse.bots.TaskBot.utils import RandomDP, PriorityDP, TemplateBasedNLG
from botiverse.models.TRIPPY.config import TRIPPYConfig
from botiverse.models.TRIPPY.TRIPPY_DST import TRIPPYDST
from botiverse.models.BERT.config import BERTConfig

import random

class TaskBot:
  """
  Instantiate a Deep Task Oriented Dialogue System chat bot.
  It aims to assist the user in completing certain tasks in specific domains.
  The chat bot can use a Deep learning approach for training and inference.

  :param domains: List of domain names.
  :type domains: list[str]

  :param slot_list: List of slot names.
  :type slot_list: list[str]

  :param start: List of initial system utterances and corresponding system acts.
  :type start: list[dict]

  :param templates: The predefined templates for generating responses.
  :type templates: list[dict]

  :param label_maps: Dictionary of the variants of the slot-values that are mapped to the canonical slot-values.
  :type label_maps: dict[str, list]

  :param policy: The dialogue policy to be used ('Random' or 'Priority').
  :type policy: str

  :param non_referable_slots: List of non-referable slots, defaults to an empty list.
  :type non_referable_slots: list[str]

  :param non_referable_pairs: List of non-referable slot-value pairs, defaults to an empty list.
  :type non_referable_pairs: list[tuple]

  :param from_scratch: Indicates whether to use BERT model implemented from scratch in the library, defaults to False.
  :type from_scratch: bool

  :param BERT_config: The configuration of the BERT model, defaults to BERTConfig().
  :type BERT_config: BERTConfig

  :param TRIPPY_config: The configuration of the TRIPPY model, defaults to TRIPPYConfig().
  :type TRIPPY_config: TRIPPYConfig

  :param verbose: Indicates whether to print the chatbot's state after each inference, defaults to False.
  :type verbose: bool

  :param append_state: Indicates whether to append the dialogue state to the response when all slots are filled, defaults to False.
  :type append_state: bool
  """
  def __init__(self, domains=[], slot_list=[], start=[], templates=[], label_maps={}, policy='Priority', non_referable_slots=[], non_referable_pairs=[], from_scratch=True, BERT_config=BERTConfig(), TRIPPY_config=TRIPPYConfig(), verbose=False, append_state=False):
    self.domains = domains
    self.policy = policy
    self.start = start
    self.is_start = True
    self.dst = TRIPPYDST(domains, slot_list, label_maps, non_referable_slots, non_referable_pairs, from_scratch, BERT_config, TRIPPY_config)
    self.dpo = RandomDP() if policy == 'Random' else PriorityDP() if policy == 'Priority' else None
    self.nlg = TemplateBasedNLG(templates)
    self.sys_utter = ''
    self.inform_mem = {}
    self.verbose = verbose
    self.append_state = append_state

  def save_model(self, model_path):
    """
    Save the trained DST model to the given path.

    :param model_path: Path to save the trained DST model.
    :type model_path: str
    """
    self.dst.save_model(model_path)


  def read_data(self, train_path, dev_path=None, test_path=None):
    """
    Read the training, development and testing data and store them in the chatbot.

    :param train_path: Path to the training data file.
    :type train_path: str

    :param dev_path: Path to the development data file, defaults to None.
    :type dev_path: str

    :param test_path: Path to the testing data file, defaults to None.
    :type test_path: str
    """

    self.train_path = train_path
    self.dev_path = dev_path
    self.test_path = test_path

  def train(self):
    """
    Train the chatbot model with the given training data.

    """
    self.dst.train(self.train_path, self.dev_path, self.test_path)

  def load_dst_model(self, model_path=None, pretrained='sim-R', test_path=None):
    """
    Load a trained DST model from the given path.

    :param model_path: Path to the trained DST model.
    :type model_path: str

    :param pretrained: The pretrained model to be used defaults to 'sim-R'.
    :type pretrained: str

    :param test_path: Path to the testing data file, if applicable, defaults to None.
    :type test_path: str
    """
    self.dst.load_model(model_path, pretrained, test_path)

  def infer(self, user_utter):
    """
    Infer a suitable response to the user's utterance.

    :param user_utter: The user's input utterance.
    :type user_utter: str

    :return: The chatbot's response.
    :rtype: str
    """

    response = None

    if self.is_start and len(self.start) > 0:
      temp = self.start[random.randint(0, len(self.start)-1)]
      response, inform_mem = temp['utterance'], temp['system_act']
      self.sys_utter = response
      self.inform_mem = inform_mem
    else:  
      state = self.dst.update_state(self.sys_utter, user_utter, self.inform_mem)
      action = self.dpo.get_action(state, self.nlg.get_templates_slots())
      if action is not None:
        response, inform_mem = self.nlg.generate(action)
        self.sys_utter = response
        self.inform_mem = inform_mem

    self.is_start = False

    if self.verbose:
      print(f'State: {self.get_dialogue_state()}')

    if self.append_state and self.dst.is_all_slots_filled():
      if response is None:
        response = ''
      state = self.get_dialogue_state()
      response = response + '\n' + str(state)

    return response
  
  def infer_with_state(self, user_utter, line_breaker='<br/>'):
    """
    Infer a suitable response to the user's utterance and return the dialogue state.

    :param user_utter: The user's input utterance.
    :type user_utter: str

    :return: The chatbot's response and the dialogue state concatenated into a single string.
    :rtype: str
    """

    response = self.infer(user_utter)
    if response is None:
      response = ''
    state = self.get_dialogue_state()
    return response + line_breaker + str(state)


  def suggest(self, template):
    """
    Set the system utterance and system act to suggest a specific response.

    :param template: The template containing the suggested system utterance and system act.
    :type template: dict
    """
    self.sys_utter = template['utterance']
    self.inform_mem = template['system_act']

  def get_dialogue_state(self):
    """
    Get the dialogue state.

    :return: The dialogue state.
    :rtype: dict
    """
    return self.dst.get_dialogue_state()

  def delete_slots(self, domain=None, slot=None):
    """
    Delete slots from the dialogue state.

    Note that:
    if domain!=None will delete all slots in that domain.
    if slot!=None will delete that slot.
    if both are None will delete all slots in all domains.

    :param domain: The domain from which to delete slots, defaults to None.
    :type domain: str

    :param slot: The slot to delete, defaults to None.
    :type slot: str
    """
    self.dst.delete_slots(domain, slot)

  def reset(self):
    """
    Reset the chatbot's state.
    """
    self.dst.reset()
    self.sys_utter = ''
    self.inform_mem = {}
    self.domain = self.domains[0]
    self.is_start = True

  def __str__(self):
    """
    Return a string representation of the chatbot.

    :return: A string representation of the chatbot.
    :rtype: str
    """
    string = ''
    string = string + '\ndomains: ' + str(self.domains)
    string = string + '\npolicy: ' + str(self.policy)
    string = string + '\nstart: ' + str(self.start)
    string = string + '\nis_start: ' + str(self.is_start)
    string = string + '\n\ndst: ' + str(self.dst)
    string = string + '\n\ndpo: ' + str(self.dpo)
    string = string + '\n\nnlg: ' + str(self.nlg)
    string = string + '\n\nsys_utter: ' + str(self.sys_utter)
    string = string + '\ninform_mem: ' + str(self.inform_mem)
    return string