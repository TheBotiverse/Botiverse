"""
This module ontains utility code used by the deep Taskbot module
such as Natural Language Understanding (NLU), Dialogue State Tracker (DST) ... etc.
"""

import random

class PriorityDP:
    """
    Dialogue Policy Optimizer that selects the next action based on priority.

    This policy selects the action with the highest priority that does not conflict with
    the slots already filled in the dialogue state.

    """
    def get_action(self, state, templates_slots):
        """
        Get the next action based on the given dialogue state and available action templates.

        :param state: The current dialogue state.
        :type state: dict

        :param templates_slots: The available different combinations of slots that can be filled by templates.
        :type templates_slots: list[tuple]

        :return: The index of the selected action template, or None if no action is available.
        :rtype: int or None
        """
        
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
      """
      Return a string representation of the PriorityDP policy.

      :return: A string representation of the PriorityDP policy.
      :rtype: str
      """
      string = ''
      string = string + '\nPriorityDP'
      return string


class RandomDP:
    """
    Dialogue Policy Optimizer that selects the next action randomly.

    This policy selects the action randomly from the available action templates
    that do not conflict with the slots already filled in the dialogue state.

    """
    def get_action(self, state, templates_slots):
        """
        Get the next action based on the given dialogue state and available action templates.

        :param state: The current dialogue state.
        :type state: dict

        :param templates_slots: The available different combinations of slots that can be filled by templates.
        :type templates_slots: list[tuple]

        :return: The index of the selected action template, or None if no action is available.
        :rtype: int or None
        """
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
      """
      Return a string representation of the RandomDP policy.

      :return: A string representation of the RandomDP policy.
      :rtype: str
      """
      string = ''
      string = string + '\nRandomDP'
      return string


class TemplateBasedNLG:
    """
    Natural Language Generation module that generates responses based on predefined templates.

    This module uses a set of predefined templates containing utterances and corresponding system acts.
    Given an index, it generates the corresponding system utterance and system act.

    :param templates: The predefined templates for generating responses.
    :type templates: list[dict]
    """

    def __init__(self, templates):
      self.templates = templates
      self.templates_slots = []

      for template in templates:
        self.templates_slots.append(tuple(sorted(template['slots'])))

    def get_templates(self):
      """
      Get the predefined templates.

      :return: The predefined templates.
      :rtype: list[dict]
      """
      return self.templates

    def get_templates_slots(self):
      """
      Get the slots associated with the predefined templates.

      :return: The slots associated with the predefined templates.
      :rtype: list[tuple]
      """
      return self.templates_slots

    def generate(self, idx):
      """
      Generate a response based on the given index.

      :param idx: The index of the template to generate a response from.
      :type idx: int

      :return: The generated system utterance and system act.
      :rtype: tuple[str, list] or None
      """

      if idx < 0 or idx >= len(self.templates):
        return None, None

      return self.templates[idx]['utterance'], self.templates[idx]['system_act']

    def __str__(self):
      """
      Return a string representation of the TemplateBasedNLG module.

      :return: A string representation of the TemplateBasedNLG module.
      :rtype: str
      """
      string = ''
      string = string + '\ntemplates: ' + str(self.templates)
      string = string + '\ntemplates_slots: ' + str(self.templates_slots)
      return string