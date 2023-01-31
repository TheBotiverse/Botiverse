import unittest
import math
from botiverse import basic_chatbot


class BasicChatBotTestCase(unittest.TestCase):

    def setUp(self):
        self.model = basic_chatbot("Max")

    def test_return_type_infer(self):
        """Test that infer returns a string"""

        # Make a prompt and provide it to the model then check response type
        prompt = "Can you tell me a joke?"
        response = self.model.infer(prompt)
        
        self.assertIsInstance(response, str)

    def test_return_type_train(self):
        """Test that train returns None"""
        # Make dummy data and provide it to the model then check response type
        data = "abcdefgh"
        nothing = self.model.train(data)
        self.assertIsInstance(nothing, type(None))




if __name__ == '__main__':
    unittest.main()
    

# whenever you need to test the BasicChatBot class, run the following:
# python -m unittest -v botiverse-tests/basic_chatbot_test/basic_chatbot_test.py
