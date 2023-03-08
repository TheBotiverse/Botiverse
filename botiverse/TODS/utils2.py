import re
import random

class RandomDP:
    def get_action(self, state):

        domains = set()
        for slot, value in state.items():
            if value is not None:
                domains.add(slot.split('-')[0])

        unfilled = []
        for slot, value in state.items():
            if slot.split('-')[0] in domains and value is None:
                unfilled.append(slot)

        max_len = len(unfilled)

        if max_len == 0:
            return ""

        return unfilled[random.randint(0, max_len-1)]

class TemplateBasedNLG:
    def __init__(self, templates):
        self.templates = templates

    def generate(self, slot):
        max_len = len(self.templates[slot])
        return self.templates[slot][random.randint(0, max_len-1)]
