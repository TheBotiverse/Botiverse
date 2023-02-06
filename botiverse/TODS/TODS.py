"""
This module contains the base code and interface of TODS, .
"""
from botiverse.TODS.utils import RuleBasedNLU, MostRecentDST, RandomDP, TemplateBasedNLG


class TODS:
    """
    Instantiate a Task Oriented Dialogue System chat bot.
    It aims to assist the user in completing certain tasks in specific domains.
    The chat bot can use a fully Rule-Based approach which required no data for
    training and can also include Deep learning and data can be then used to train
    the chatbot model.

    :param name: The chatbot's name.
    :type name: str

    :param domains_slots: The slots of each domain in the system.
    :type domains_slots: dict[str, list[str]]

    :param templates: The predefined templates, where templates["domain"]["slot"] is
        a list of questions asking about slot "slot" in domain "domain".
    :type templates: dict[str, dict[str, list[str]]]

    :param domains_pattern: Dictionary where the key is a name of a domain
        and the value is a Regex that is used to capture that domain.
    :type domains_pattern: dict[str, str]

    :param slots_pattern: Dictionary where the key is a name of a domain
        and the value is another dictionary where the key is a name of a slot
        inside the domain and the value is a Regex that is used to capture that slot.
    :type slots_pattern: dict[str, dict[str, str]]

    :param is_classical: Indicates whether to use a fully classical approach, defaults to False
    :type is_classial: bool
    """

    def __init__(self, name, domains_slots, templates,
    domains_pattern, slots_pattern, is_classical=False):
        self.name = name
        self.nlu = RuleBasedNLU(domains_pattern, slots_pattern) if is_classical is False else None
        self.dst = MostRecentDST(domains_slots)
        self.dpo = RandomDP()
        self.nlg = TemplateBasedNLG(templates)
        self.current_domain = None

    # def train(self, data):
    #     """
    #     Train the chatbot model with the given JSON data.

    #     :param data: A stringfied JSON object containing the training data
    #     :type number: string

    #     :return: None
    #     :rtype: NoneType
    #     """
    #     rubbish = data

    def infer(self, prompt):
        """
        Infer a suitable response to the given prompt.

        :param promp: The user's prompt
        :type number: str

        :return: When all slots in the current domain are filled it returns True, &
            the domain name, otherwise it returns False, & the chatbot's response.
        :rtype: tuple[bool, str]
        """

        status, self.current_domain = self.nlu.get_domain(self.current_domain, prompt)
        if status is False or self.current_domain is None:
            return False, "Sorry I don't know what domain are you talking about!"

        status, slots, values = self.nlu.get_slot_fillers(self.current_domain, prompt)
        if status is False:
            return False, "Sorry I Couldn't catch that can you say it in a different way!"

        self.dst.update_state(self.current_domain, slots, values)

        action = self.dpo.get_action(self.current_domain, self.dst.get_dialogue_state())
        if action == "":
            return True, self.current_domain

        response = self.nlg.generate(self.current_domain, action)

        return False, response

    def get_dialogue_state(self):
        """
        Gets the dialogue state.

        :return: The dialogue state, where state["domain"]["slot"] indicates
            value of slot "slot" in domain "domain".
        :rtype: dict[str, dict[str, str]]
        """
        return self.dst.get_dialogue_state()

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
        self.dst.reset(domain, slot)
