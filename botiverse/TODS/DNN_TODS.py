from botiverse.TODS.utils2 import RandomDP, PriorityDP, TemplateBasedNLG
from botiverse.TODS.DNN_DST.DNN_DST import DNNDST

import random

class DNNTODS:

    def __init__(self, name, domains, ontology_path, label_maps_path, policy, start, templates, non_referable_slots=[], non_referable_pairs=[], from_scratch=False):
        self.name = name
        self.domains = domains
        self.policy = policy
        self.start = start
        self.is_start = True
        self.dst = DNNDST(domains, ontology_path, label_maps_path, non_referable_slots, non_referable_pairs, from_scratch)
        self.dpo = RandomDP() if policy == 'Random' else PriorityDP() if policy == 'Priority' else None
        self.nlg = TemplateBasedNLG(templates)
        self.sys_utter = ''
        self.inform_mem = {}

    def train(self, train_path, dev_path, test_path, model_path):
        self.dst.train(train_path, dev_path, test_path, model_path)

    def load_dst_model(self, model_path, test_path=None):
        self.dst.load_model(model_path, test_path)

    def infer(self, user_utter):
      
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
      self.sys_utter = template['utterance']
      self.inform_mem = template['system_act']

    def get_dialogue_state(self):
      return self.dst.get_dialogue_state()

    def delete_slots(self, domain=None, slot=None):
      self.dst.delete_slots(domain, slot)

    def reset(self):
        self.dst.reset()
        self.sys_utter = ''
        self.inform_mem = {}
        self.domain = self.domains[0]
        self.is_start = True

    def __str__(self):
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