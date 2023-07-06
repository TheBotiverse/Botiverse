import random

class PriorityDP:
    def get_action(self, state, templates_slots):

        filled = []
        for slot, value in state.items():
          if value is not None:
            filled.append(slot)

        filled = tuple(sorted(filled))

        top_idx = -1
        for idx, slots in enumerate(templates_slots):
          if all(element not in slots for element in filled):
            top_idx = idx
            break

        action = None
        if top_idx != -1:
          action = top_idx

        return action

    def __str__(self):
      string = ''
      string = string + '\PriorityDP'
      return string


class RandomDP:
    def get_action(self, state, templates_slots):

        filled = []
        for slot, value in state.items():
          if value is not None:
            filled.append(slot)

        filled = tuple(sorted(filled))

        candidates = []
        for idx, slots in enumerate(templates_slots):
          if all(element not in slots for element in filled):
            candidates.append(idx)

        # print('candidates', candidates)

        max_len = len(candidates)

        action = None

        if max_len > 0:
          action = candidates[random.randint(0, max_len-1)]

        return action

    def __str__(self):
      string = ''
      string = string + '\nRandomDP'
      return string


class TemplateBasedNLG:
    def __init__(self, templates):
      self.templates = templates
      self.templates_slots = []

      for template in templates:
        self.templates_slots.append(tuple(sorted(template['slots'])))

    def get_templates(self):
      return self.templates

    def get_templates_slots(self):
      return self.templates_slots

    def generate(self, idx):
      return self.templates[idx]['utterance'], self.templates[idx]['system_act']

    def __str__(self):
      string = ''
      string = string + '\ntemplates: ' + str(self.templates)
      string = string + '\ntemplates_slots: ' + str(self.templates_slots)
      return string