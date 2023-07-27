"""
This module ontains utility code used by the basic Taskbot module
such as Natural Language Understanding (NLU), Dialogue State Tracker (DST) ... etc.
"""
import re
import random

class RuleBasedNLU:
    """
    Instantiate a Rule-Based Natural Language Understanding module.
    It uses Regex to extract domain and slot-fillers from the user's utterance.

    :param domains_pattern: Dictionary where the key is a name of a domain
        and the value is a Regex that is used to capture that domain.
    :type domains_pattern: dict[str, str]

    :param slots_pattern: Dictionary where the key is a name of a domain
        and the value is another dictionary where the key is a name of a slot
        inside the domain and the value is a Regex that is used to capture that slot.
    :type slots_pattern: dict[str, dict[str, str]]
    """

    def __init__(self, domains_pattern, slots_pattern):
        self.domains_pattern = domains_pattern
        self.slots_pattern = slots_pattern

    def get_domain(self, prev_domain, utterance):
        """
        Takes previously extracted domain and the user's utterance and extract the domain from it,
        if no domain can be extracted it returns prev_domain.

        :param prev_domain: The domain from previous user's utterance.
        :type prev_domain: str

        :param utterance: The user's utterance.
        :type utterance: str

        :return: The success of the domain extraction & The extracted domain.
        :rtype: tuple[bool, str]
        """
        detected = []
        for domain, pattern in self.domains_pattern.items():
            if re.search(pattern, utterance):
                detected.append(domain)

        if len(detected) > 1:
            return False, ""

        if len(detected) == 0:
            return True, prev_domain

        return True, detected[0]

    def get_slot_fillers(self, current_domain, utterance):
        """
        Takes the current extracted domain and the user's utterance
        and extract the slot-fillers.

        :param current_domain: The domain extracted from the current
            user's utterance.
        :type utterance: str

        :param utterance: The user's utterance.
        :type utterance: str

        :return: The success of the slot-fillers extraction, the extracted slots
            & the extracted slot-values.
        :rtype: tuple[bool, list[str], list[str]]
        """
        slots, values = [], []
        for slot, pattern in self.slots_pattern[current_domain].items():
            matches = set()
            for match in re.finditer(pattern, utterance):
                matches.add(" ".join(match.groups()))

            if len(matches) > 1:
                return False, [], []

            if len(matches) == 1:
                slots.append(slot)
                values.append(list(matches)[0])

        return True, slots, values

class MostRecentDST:
    """
    Instantiate a Dialogue State Tracker module.
    It maintains the state of the dialogue which is mainly represented in
    the slots and the slots-values that are extracted from the user's utterances.
    It is a most recent tracker which means it only keeps track of the recent value in case
    if multiple values appear for the same slot.

    :param domains_slots: The slots of each domain in the system.
    :type domains_slots: dict[str, list[str]]
    """
    def __init__(self, domains_slots):
        self.state = {}
        for domain, slots in domains_slots.items():
            self.state[domain] = {}
            for slot in slots:
                self.state[domain][slot] = None

    def update_state(self, domain, slots, values):
        """
        Update the slots values of a certain domain.

        :param domain: The domain of the slots to be updated.
        :type domain: str

        :param slots: List the of the slots to be updated.
        :type slots: list[str]

        :param values: List of the values of the slots.
        :type values: list[str]
        """
        for i, slot in enumerate(slots):
            self.state[domain][slot] = values[i]

    def get_dialogue_state(self):
        """
        Gets the dialogue state.

        :return: The dialogue state, where state["domain"]["slot"] indicates
            value of slot "slot" in domain "domain".
        :rtype: dict[str, dict[str, str]]
        """
        return self.state
    
    def is_all_slots_filled(self, domain):
        """
        Checks if all slots in a certain domain are filled.

        :param domain: The domain to be checked.
        :type domain: str

        :return: The success of the check.
        :rtype: bool
        """
        for slot, value in self.state[domain].items():
            if value is None:
                return False
        return True

    def reset(self, domain=None, slot=None):
        """
        Reset the dialogue state.
        Note: if the domain or the slot is equal to None all domains
        or slots will be reset respectively.

        :param domain: The domain to be reset, defaults to None.
        :type domain: str

        :param slot: The slot to be reset, defaults to None.
        :type slot: str

        """
        if domain is None and slot is None:
            for cur_domain, slot_filler in self.state.items():
                for cur_slot in slot_filler:
                    self.state[cur_domain][cur_slot] = None
        elif domain is not None and slot is None:
            for cur_slot in self.state[domain]:
                self.state[domain][cur_slot] = None
        elif domain is None and slot is not None:
            for cur_domain, slot_filler in self.state.items():
                if slot in slot_filler:
                    self.state[cur_domain][slot] = None
        else:
            self.state[domain][slot] = None

class RandomDP:
    """
    Instantiate a Random Dialogue Policy module.
    Randomly determines the next action which is the
    next empty slot to ask user about.
    """
    def get_action(self, current_domain, state):
        """
        Takes the current domain and dialogue state and return
        the next action which is the next empty slot to ask user about.

        :param current_domain: The current extracted domain.
        :type current_domain: str

        :param state: The current state of the dialogue.
        :type state: dict[str, dict[str, str]]

        :return: The next empty slot to ask the user about.
        :rtype: str
        """

        unfilled = []
        for slot, value in state[current_domain].items():
            if value is None:
                unfilled.append(slot)

        max_len = len(unfilled)

        if max_len == 0:
            return "ALL-FILLED"

        return unfilled[random.randint(0, max_len-1)]

class TemplateBasedNLG:
    """
    Instantiate a Template Based Natural Language Generation module.
    It uses a predefined templates to generate a response to the user
    and ask about empty slots.

    :param templates: The predefined templates, where templates["domain"]["slot"] is
        a list of questions asking about slot "slot" in domain "domain".
    :type templates: dict[str, dict[str, list[str]]]
    """
    def __init__(self, templates):
        self.templates = templates

    def generate(self, domain, slot):
        """
        Takes a slot and the domain of the slot and generate a question randomly
        from the templates about the slot.

        :param domain: The domain of the slost.
        :type domain: str

        :param slot: The slot.
        :type slot: str

        :return: The generated question.
        :rtype: str
        """
        
        response = ''
        if slot == "ALL-FILLED":
            if "ALL-FILLED" in self.templates[domain]:
                max_len = len(self.templates[domain]["ALL-FILLED"])
                response = self.templates[domain]["ALL-FILLED"][random.randint(0, max_len-1)]
        else:
            max_len = len(self.templates[domain][slot])
            response = self.templates[domain][slot][random.randint(0, max_len-1)]

        return response
