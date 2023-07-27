"""
This module contains the base code and interface of basic Taskbot.
"""
from botiverse.bots.BasicTaskBot.utils import RuleBasedNLU, MostRecentDST, RandomDP, TemplateBasedNLG


class BasicTaskBot:
    """
    Instantiate a Task Oriented Dialogue System chat bot.
    It aims to assist the user in completing certain tasks in specific domains.
    The chat bot can use a fully Rule-Based approach which required no data for
    training and can also include Deep learning and data can be then used to train
    the chatbot model.

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

    :param verbose: Whether to print the internal state of the chatbot.
    :type verbose: bool

    :param append_state: Whether to append the internal state of the chatbot to the response.
    :type append_state: bool

    """

    def __init__(self, domains_slots, templates, domains_pattern, slots_pattern, verbose=False, append_state=False):
        self.nlu = RuleBasedNLU(domains_pattern, slots_pattern)
        self.dst = MostRecentDST(domains_slots)
        self.dpo = RandomDP()
        self.nlg = TemplateBasedNLG(templates)
        self.verbose = verbose
        self.current_domain = None
        self.append_state = append_state

    def read_data(self, data):
        """
        Read the chatbot training data, redundant here as we are using a fully classical approach.

        :param data: any
        :type number: any

        :return: None
        :rtype: NoneType
        """
        pass

    def train(self):
        """
        Train the chatbot model with the given data in read_data, redundant here as we are using a fully classical approach.

        :param data: any
        :type number: any

        :return: None
        :rtype: NoneType
        """
        pass
    
    def infer(self, prompt):
        """
        Infer a suitable response to the given prompt.

        :param promp: The user's prompt
        :type number: str

        :return: When all slots in the current domain are filled it returns empty string, otherwise it returns the response.
        :rtype: str
        """

        response = ""
        status, self.current_domain = self.nlu.get_domain(self.current_domain, prompt)
        # if no domain is detected or the domain is not supported return a default response
        if status is False or self.current_domain is None:
            response = "Sorry I don't know what domain are you talking about!"
        else:
            # get the slots and their values
            status, slots, values = self.nlu.get_slot_fillers(self.current_domain, prompt)
            # if no slots are detected or the slots are not supported return a default response
            if status is False:
                response = "Sorry I Couldn't catch that can you say it in a different way!"
            else:
                # update the dialogue state
                self.dst.update_state(self.current_domain, slots, values)
                # get the action to be taken
                action = self.dpo.get_action(self.current_domain, self.dst.get_dialogue_state())
                # if no action is detected return empty response
                response = self.nlg.generate(self.current_domain, action)

        if self.verbose == True:
            print(f'State: {self.dst.get_dialogue_state()}')
        
        if self.append_state and self.dst.is_all_slots_filled(self.current_domain):
            response += str(self.dst.get_dialogue_state()[self.current_domain])

        return response

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
