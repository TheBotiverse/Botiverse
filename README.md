# Botiverse
A library that imports chatbots from other galaxies

### Installation
```
pip install botiverse
```

### Get started
Try to import and playaround with the Basic Chat Bot:

```Python
from botiverse import basic_chatbot

# Make a new chatbot and give it a name
Max = basic_chatbot("Max")

# Train the chatbot
Max.train("abcdefgh")

# Ask the chatbot a question
response = Max.infer("Can you tell me a joke?")

print(response)
```
