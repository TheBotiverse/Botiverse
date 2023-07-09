"""
Example on using Rule-Based TODS.
"""
from botiverse.bots.basic_TODS.basic_TODS import BasicTODS

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

chatbot = BasicTODS(NAME, domains_slots, templates, domains_pattern, slots_pattern)

print(chatbot.get_dialogue_state())
print(chatbot.infer("I want to book a flight"))
print(chatbot.infer("I want to go to giza from cairo"))
print(chatbot.infer("on monday"))
print(chatbot.get_dialogue_state())
print(chatbot.reset())
print(chatbot.get_dialogue_state())
