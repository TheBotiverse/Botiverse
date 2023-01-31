from botiverse import basic_chatbot

Max = basic_chatbot("Max")
Max.train("abcdefgh")
response = Max.infer("Can you tell me a joke?")
print(response)

from botiverse import TODS

Jax = TODS("Jax")
Jax.train("hijklmno")
response = Jax.infer("How old are you?")
print(response)