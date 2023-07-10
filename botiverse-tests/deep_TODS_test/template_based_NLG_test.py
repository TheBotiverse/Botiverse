import unittest
from botiverse.bots.deep_TODS.utils import TemplateBasedNLG

class TemplateBasedNLGTests(unittest.TestCase):

    def setUp(self):
        templates = [
            {
                'utterance': 'Hello!',
                'slots': [],
                'system_act': {}
            },
            {
                'utterance': 'The egyptain food is delicious.',
                'slots': ['restaurant-food_type'],
                'system_act': {'restaurant-food_type': ['egyptian']}
            },
            {
                'utterance': 'Goodbye!',
                'slots': [],
                'system_act': {}
            }
        ]
        self.nlg = TemplateBasedNLG(templates)

    def test_get_templates(self):
        expected_templates = [
            {
                'utterance': 'Hello!',
                'slots': [],
                'system_act': {}
            },
            {
                'utterance': 'The egyptain food is delicious.',
                'slots': ['restaurant-food_type'],
                'system_act': {'restaurant-food_type': ['egyptian']}
            },
            {
                'utterance': 'Goodbye!',
                'slots': [],
                'system_act': {}
            }
        ]
        templates = self.nlg.get_templates()
        self.assertEqual(templates, expected_templates)

    def test_get_templates_slots(self):
        expected_slots = [
            (),
            ('restaurant-food_type',),
            ()
        ]
        slots = self.nlg.get_templates_slots()
        self.assertEqual(slots, expected_slots)

    def test_generate_with_valid_index(self):
        idx = 1
        expected_utterance = 'The egyptain food is delicious.'
        expected_system_act = {'restaurant-food_type': ['egyptian']}
        utterance, system_act = self.nlg.generate(idx)
        self.assertEqual(utterance, expected_utterance)
        self.assertEqual(system_act, expected_system_act)

    def test_generate_with_invalid_index(self):
        idx = 10
        expected_utterance = None
        expected_system_act = None
        utterance, system_act = self.nlg.generate(idx)
        self.assertEqual(utterance, expected_utterance)
        self.assertEqual(system_act, expected_system_act)

if __name__ == '__main__':
    unittest.main()
