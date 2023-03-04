import json
from botiverse.TODS.DNN_DST.utils import RawDataInstance, DataInstance
import torch
import numpy as np
from tqdm import tqdm


def get_ontology_label_maps(ontology_path, label_maps_path, domains):
  # read ontology
  file = open(ontology_path)
  slot_list = json.load(file)

  # delete slots not in domains 
  del_slot = []
  for slot in slot_list:
    found = False
    for domain in domains:
      if domain in slot:
        found = True
    if found == False:
      del_slot.append(slot)
  for slot in del_slot:
    del slot_list[slot_list.index(slot)]

  # read label_maps
  file = open(label_maps_path)
  label_maps = json.load(file)

  return sorted(slot_list), label_maps

def read_raw_data(data_json, slot_list, max_len, domains):
  # read json
  parsed_data = json.loads(data_json)

  raw_data = []
  # loop over dialogues
  for dial_info in parsed_data:
    dial_idx = dial_info['dialogue_idx']

    history = []
    last_state = {}
    # loop over dialogue turns
    for turn in dial_info['dialogue']:

      turn_idx = turn['turn_idx']
      turn_domain = turn["domain"]
      user_utter = turn['transcript']
      sys_utter = turn['system_transcript']

      # ->>>>>>>>>> DELETE: MULtIWOZ ONLY
      if turn_domain not in domains:
        continue

      # create dialogue state
      cur_state = {}
      for slot_info in turn['belief_state']:
        cur_state[slot_info['slots'][0]] = slot_info['slots'][1]

      # append current instance
      raw_data.append(RawDataInstance(dial_idx,
                                      turn_idx,
                                      turn_domain,
                                      user_utter,
                                      sys_utter,
                                      history,
                                      last_state,
                                      cur_state))
      
      # update history & last state for next turn
      history = [user_utter, sys_utter] + history
      last_state = cur_state

  return raw_data

def create_target_labels(last_state, cur_state, slot_list):

  # initialize all operations to carryover & all targets to NULL
  opers = ['carryover'] * len(slot_list)
  target_values = ['[NULL]'] * len(slot_list)

  # remove from the current state any slot that has 'none' value
  # or is not in the ontology
  cur_slots = list(cur_state.keys())
  for slot in cur_slots:
    if cur_state[slot] == 'none' or slot not in slot_list:
      cur_state.pop(slot)

  # for any slot in the current state if its value has changed
  # from the last state then its operation is either 'dontcare'
  # or 'update'
  for slot in cur_state:
    if slot not in last_state or cur_state[slot] != last_state[slot]:
      slot_idx = slot_list.index(slot)
      if cur_state[slot] == 'dontcare':
        opers[slot_idx] = 'dontcare'
      elif cur_state[slot] == 'yes':
          opers[slot_idx] = 'yes'
      elif cur_state[slot] == 'no':
          opers[slot_idx] = 'no'
      else:
        opers[slot_idx] = 'update'
        target_values[slot_idx] = cur_state[slot]
        # target_values[slot_idx]  = " ".join(target_values[slot_idx].split())

  # for any slot in the last state but not in the current state
  # its operation is 'delete'
  for slot in last_state:
    if slot not in cur_state:
      slot_idx = slot_list.index(slot)
      opers[slot_idx] = 'delete'

  return opers, target_values

def create_slot_span(input, target_value, tok_input_offsets, padding_len, label_maps):

  # # remove any extra spaces in the target slot value
  # target_values = " ".join(target_values.split())

  # get all possible variants of the slot value
  label_variants = [target_value]
  if target_value in label_maps:
    label_variants = label_variants + label_maps[target_value]
  
  # find the first variant of the target slot that appears
  # in the input string 
  start, end = -1, -1
  found = False
  for label in label_variants:
    if found == True:
      break
    for idx in (j for j, e in enumerate(input) if e == label[0]):
      if input[idx:idx + len(label)] == label:
        start, end = idx, idx + len(label) - 1
        found = True
        break
  
  # mark the target span in the input string
  char_target = [0] * len(input)
  if start != -1 and end != -1:
    for j in range(start, end + 1):
      if input[j] != " ":
        char_target[j] = 1

  # mark the target span after tokenization
  span = [0] * len(tok_input_offsets)
  for j, (offset1, offset2) in enumerate(tok_input_offsets):
    if sum(char_target[offset1:offset2]) > 0:
      span[j] = 1

  # update the target as tok_input_offsets doesn not include
  # [CLS] & [SEP] in the start & end of input string 
  span = [0] + span + [0]

  # get the start & end index of the span if any 
  # otherwise 0
  span_start = 0
  span_end = 0
  non_zero = np.nonzero(span)[0]
  if len(non_zero) > 0:
    span_start = non_zero[0]
    span_end = non_zero[-1]

  # pad the target span 
  span = span + [0] * padding_len

  return span, span_start, span_end

def create_inputs(history, user_utter, sys_utter, tokenizer, max_len):

  # create input string
  history = " ".join(history)
  current_utter = user_utter + ' [SEP] ' + sys_utter + ' [SEP] '
  input = current_utter + history
  input = " ".join(input.split())

  # tokenize and truncate input
  tok_input = tokenizer.encode(input)
  tok_input_tokens = tok_input.tokens[:max_len]
  tok_input_ids = tok_input.ids[:max_len]
  tok_input_offsets = tok_input.offsets[1:max_len-1]
  if tok_input_tokens[-1] != '[SEP]':
    tok_input_tokens[-1] = '[SEP]'
    tok_input_ids[-1] = 102

  # create mask & input type id
  mask = [1] * len(tok_input_ids)
  token_type_ids = []
  cnt = 0
  for i, token in enumerate(tok_input_tokens):
    token_type_ids.append(1 if cnt >= 2 else 0)
    cnt += 1 if token == '[SEP]' else 0
  
  # pad the inputs
  padding_len = max_len - len(tok_input_ids)
  ids = tok_input_ids + [0] * padding_len
  mask = mask + [0] * padding_len
  token_type_ids = token_type_ids + [0] * padding_len
  
  return input, ids, mask, token_type_ids, tok_input_offsets, tok_input_tokens, padding_len

def create_data(raw_data, slot_list, label_maps, tokenizer, max_len):

  data = []

  # loop over raw data
  for turn in tqdm(raw_data):
    
    # create turn operations and span targets
    opers, target_values = create_target_labels(turn.last_state,
                                                turn.cur_state,
                                                slot_list)
    
    # create model inputs
    input, ids, mask, token_type_ids, tok_input_offsets, input_tokens, padding_len = create_inputs(turn.history,
                                                                                                   turn.user_utter,
                                                                                                   turn.sys_utter,
                                                                                                   tokenizer,
                                                                                                   max_len)

    spans = []
    spans_start = []
    spans_end = []
    # create the target span for each slot
    for slot in range(len(slot_list)):
      span, span_start, span_end = create_slot_span(input,
                                                    target_values[slot], 
                                                    tok_input_offsets, 
                                                    padding_len,
                                                    label_maps)
      spans.append(span)
      spans_start.append(span_start)
      spans_end.append(span_end)

    data.append(DataInstance(ids,
                             mask,
                             token_type_ids,
                             spans,
                             spans_start,
                             spans_end,
                             padding_len,
                             input_tokens,
                             input,
                             opers,
                             target_values))
  return data

def prepare_data(data_json, slot_list, label_maps, tokenizer, max_len, domains):

  # create raw data
  raw_data = read_raw_data(data_json, slot_list, max_len, domains)

  # create data
  data = create_data(raw_data, slot_list, label_maps, tokenizer, max_len)

  return raw_data, data

class Dataset(torch.utils.data.Dataset):

  def __init__(self, data, n_slots, oper2id):

    # for k in inputs:
    #   inputs[k] = inputs[k][:32]

    self.ids = [turn.ids for turn in data]
    self.mask = [turn.mask for turn in data]
    self.token_type_ids = [turn.token_type_ids for turn in data]
    self.spans_start = [turn.spans_start for turn in data]
    self.spans_end = [turn.spans_end for turn in data]
    self.padding_len = [turn.padding_len for turn in data]
    self.input_tokens = [' '.join(turn.input_tokens) for turn in data]
    self.target_values = ['[VALUESEP]'.join(turn.target_values) for turn in data]
    self.opers = [[oper2id[oper] for oper in turn.opers] for turn in data]

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, idx):
    return {
        'ids': torch.tensor(self.ids[idx], dtype=torch.long),
        'mask': torch.tensor(self.mask[idx], dtype=torch.long),
        'token_type_ids': torch.tensor(self.token_type_ids[idx], dtype=torch.long),
        'spans_start': torch.tensor(self.spans_start[idx], dtype=torch.long),
        'spans_end': torch.tensor(self.spans_end[idx], dtype=torch.long),
        'padding_len': torch.tensor(self.padding_len[idx], dtype=torch.long),
        'input_tokens': self.input_tokens[idx],
        'target_values': self.target_values[idx],
        'opers': torch.tensor(self.opers[idx], dtype=torch.long)
    }

