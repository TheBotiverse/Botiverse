"""
An example of using the botiverse package.
"""
from botiverse import basic_chatbot

Max = basic_chatbot("Max")
Max.train("abcdefgh")
response = Max.infer("Can you tell me a joke?")
print(response)