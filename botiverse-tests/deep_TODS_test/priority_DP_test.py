import unittest
from botiverse.bots.TaskBot.utils import PriorityDP

class PriorityDPTests(unittest.TestCase):

    def setUp(self):
        self.policy = PriorityDP()

    def test_get_action_with_no_filled_slots(self):
        state = {}
        templates_slots = [(1, 2), (3, 4), (5, 6)]
        expected_action = 0
        action = self.policy.get_action(state, templates_slots)
        self.assertEqual(action, expected_action)

    def test_get_action_with_filled_slots(self):
        state = {'slot1': 'value1', 'slot2': 'value2'}
        templates_slots = [('slot1', 'value1'), ('slot3', 'value3'), ('slot4', 'value4')]
        expected_action = 1
        action = self.policy.get_action(state, templates_slots)
        self.assertEqual(action, expected_action)

    def test_get_action_with_no_available_action(self):
        state = {'slot1': 'value1', 'slot2': 'value2'}
        templates_slots = [('slot1', 'value1'), ('slot2', 'value2')]
        expected_action = None
        action = self.policy.get_action(state, templates_slots)
        self.assertEqual(action, expected_action)

    def test_string_representation(self):
        expected_string = '\nPriorityDP'
        string = str(self.policy)
        self.assertEqual(string, expected_string)

if __name__ == '__main__':
    unittest.main()
