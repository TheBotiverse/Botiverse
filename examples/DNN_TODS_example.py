
from botiverse.bots import DNNTODS

name = "boti"
domains = ['restaurant']
slot_list = ['restaurant-food']
label_maps = {}
templates = {'restaurant-food': ["what is the type of the food?", "what type of food do you want?"]}

chatbot = DNNTODS(name, domains, slot_list, label_maps, templates)

train_json = open('train_data.json').read()

print(train_json)

chatbot.train(train_json, train_json, train_json)
