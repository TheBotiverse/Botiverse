"""
This module contains the base code and interface of deep TODS .
"""

from botiverse.bots.deep_TODS.utils import RandomDP, PriorityDP, TemplateBasedNLG
from botiverse.models.TRIPPY.config import TRIPPYConfig
from botiverse.models.TRIPPY.TRIPPY_DST import TRIPPYDST

import random

class DeepTODS:
  """
  Instantiate a Deep Task Oriented Dialogue System chat bot.
  It aims to assist the user in completing certain tasks in specific domains.
  The chat bot can use a Deep learning approach for training and inference.

  :param name: The chatbot's name.
  :type name: str

  :param domains: List of domain names.
  :type domains: list[str]

  :param ontology_path: Path to the ontology file.
  :type ontology_path: str

  :param label_maps_path: Path to the label maps file.
  :type label_maps_path: str

  :param policy: The dialogue policy to be used ('Random' or 'Priority').
  :type policy: str

  :param start: List of initial system utterances and corresponding system acts.
  :type start: list[dict]

  :param templates: The predefined templates for generating responses.
  :type templates: list[dict]

  :param non_referable_slots: List of non-referable slots, defaults to an empty list.
  :type non_referable_slots: list[str]

  :param non_referable_pairs: List of non-referable slot-value pairs, defaults to an empty list.
  :type non_referable_pairs: list[tuple]

  :param from_scratch: Indicates whether to use BERT model implemented from scratch in the library, defaults to False.
  :type from_scratch: bool
  """
  def __init__(self, name, domains, ontology_path, label_maps_path, policy, start, templates, non_referable_slots=[], non_referable_pairs=[], from_scratch=False):
    self.name = name
    self.domains = domains
    self.policy = policy
    self.start = start
    self.is_start = True
    self.dst = TRIPPYDST(domains, ontology_path, label_maps_path, non_referable_slots, non_referable_pairs, from_scratch)
    self.dpo = RandomDP() if policy == 'Random' else PriorityDP() if policy == 'Priority' else None
    self.nlg = TemplateBasedNLG(templates)
    self.sys_utter = ''
    self.inform_mem = {}

  def train(self, train_path, dev_path, test_path, model_path):
    """
    Train the chatbot model with the given training data.

    :param train_path: Path to the training data file.
    :type train_path: str

    :param dev_path: Path to the development data file.
    :type dev_path: str

    :param test_path: Path to the testing data file.
    :type test_path: str

    :param model_path: Path to save the trained model.
    :type model_path: str
    """
    self.dst.train(train_path, dev_path, test_path, model_path)

  def load_dst_model(self, model_path, test_path=None):
    """
    Load a trained DST model from the given path.

    :param model_path: Path to the trained DST model.
    :type model_path: str

    :param test_path: Path to the testing data file, if applicable, defaults to None.
    :type test_path: str
    """
    self.dst.load_model(model_path, test_path)

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
    return response

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
    string = string + '\nname: ' + str(self.name)
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