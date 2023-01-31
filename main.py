"""
An example of using the botiverse package.
"""
from botiverse import basic_chatbot
from botiverse import TODS


Max = basic_chatbot("Max")
Max.train("abcdefgh")
response = Max.infer("Can you tell me a joke?")
print(response)


Jax = TODS("Jax")
Jax.train("hijklmno")
response = Jax.infer("How old are you?")
print(response)
