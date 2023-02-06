"""
An example of using the botiverse package.
"""
from botiverse import basic_chatbot
from botiverse import TODS

Max = basic_chatbot("Max")
Max.train("abcdefgh")
response = Max.infer("Can you tell me a joke?")
print(response)

NAME = "boti"
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

chatbot = TODS(NAME, domains_slots, templates, domains_pattern, slots_pattern)

print(chatbot.get_dialogue_state())
print(chatbot.infer("I want to book a flight"))
print(chatbot.infer("I want to go to giza from cairo"))
print(chatbot.infer("on monday"))
print(chatbot.get_dialogue_state())
print(chatbot.reset())
print(chatbot.get_dialogue_state())