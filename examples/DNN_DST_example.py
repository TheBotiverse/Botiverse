
from botiverse.bots import DNNDST

domains = ['restaurant']
slot_list = ['restaurant-food']
label_maps = {}


train_json = open('train_data.json').read()

print(train_json)

dst = DNNDST(domains, slot_list, label_maps)
dst.train(train_json, train_json, train_json)

