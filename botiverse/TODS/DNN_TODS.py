from botiverse.TODS.utils2 import RandomDP, TemplateBasedNLG
from botiverse.TODS.DNN_DST.DNN_DST import DNNDST

class DNNTODS:

    def __init__(self, name, domains, slot_list, label_maps, templates):
        self.name = name
        self.dst = DNNDST(domains, slot_list, label_maps)
        self.dpo = RandomDP()
        self.nlg = TemplateBasedNLG(templates)
        self.sys_utter = ''
    
    def train(self, train_json, dev_json=None, test_json=None):
        self.dst.train(train_json, dev_json, test_json)
    
    def infer(self, prompt):
        self.dst.update_state(self.sys_utter, prompt)

        action = self.dpo.get_action(self.dst.get_dialogue_state())

        if action == "":
            return True, ""

        response = self.nlg.generate(action)

        return False, response
        
    def get_dialogue_state(self):
        return self.dst.get_dialogue_state()
    
    def reset(self, domain=None, slot=None):
        self.dst.reset(domain, slot)