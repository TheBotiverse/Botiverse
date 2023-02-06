import unittest
from botiverse import TODS


class TODSTest(unittest.TestCase):

    def setUp(self):

        name = "boti"
        domains_slots = {"book-flight": ["source", "destination", "time"]}
        templates = {
                        "book-flight":
                        {
                            "source": ["Where do you want to fly from?",
                                    "From where will you take the flight?"],
                            "destination": ["What is your destination?",
                                            "Where do you want to go?"],
                            "time": ["What time do you want to leave?"]
                        }
                    }
        domains_pattern = {"book-flight": r"(i|I) want to (book|reserve) a? flights?"}
        slots_pattern = {
                            "book-flight":
                            {
                                "source": r"from(?: city)? (cairo|giza)",
                                "destination": r"to(?: city)? (cairo|giza)",
                                "time": r"(saturday|sunday|monday|tuesday|wednesday|thursday|friday)"
                            }
                        }

        self.model = TODS(name, domains_slots, templates, domains_pattern, slots_pattern)

    def test_return_type_infer(self):
        """Test that infer returns a tuple of bool and stringd"""

        # Make a prompt and provide it to the model then check response type
        prompt = "Can you tell me a joke?"
        status, response = self.model.infer(prompt)

        self.assertIsInstance(status, bool)
        self.assertIsInstance(response, str)

    # def test_return_type_train(self):
    #     """Test that train returns None"""
    #     # Make dummy data and provide it to the model then check response type
    #     data = "abcdefgh"
    #     nothing = self.model.train(data)
    #     self.assertIsInstance(nothing, type(None))




if __name__ == '__main__':
    unittest.main()
    
    
# python -m unittest -v botiverse-tests/TODS_test/TODS_test.py
