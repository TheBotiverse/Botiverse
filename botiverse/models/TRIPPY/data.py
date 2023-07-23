"""
This Module contains the data processing functions for TRIPPY.
"""

import json
import torch
import numpy as np
import re
from tqdm import tqdm

from botiverse.models.TRIPPY.utils import RawDataInstance, DataInstance, normalize, is_included, included_with_label_maps, match_with_label_maps, mask_utterance

def fix_slot_list(slot_list, domains):
  """
  Fix slot list by filtering slots based on domains.

  :param slot_list: List of slot names.
  :type slot_list: list[str]

  :param domains: The list of domains to filter the slots.
  :type domains: list[str]

  :return: The new fixed and sorted slot list.
  :rtype: list[str]
  """

  # delete slots not in the domains
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

  return sorted(slot_list)


def read_raw_data(data_path, slot_list, max_len, domains, multiwoz):
  """
  Read raw data from the JSON file and preprocess it.

  :param data_path: The path to the JSON data file.
  :type data_path: str

  :param slot_list: The list of slots.
  :type slot_list: list[str]

  :param max_len: The maximum length of the input sequence.
  :type max_len: int

  :param domains: The list of domains.
  :type domains: list[str]

  :return: The list of raw data instances.
  :rtype: list[RawDataInstance]
  """

  # read data
  file = open(data_path)
  parsed_data = json.load(file)

  raw_data = []
  # loop over dialogues
  for dial_info in parsed_data:
    dial_idx = dial_info['dialogue_idx']

    history = []
    # loop over dialogue turns
    for turn in dial_info['dialogue']:

      # turn id
      turn_idx = turn['turn_idx']

      # turn utterances
      user_utter = turn['user_utterance']
      sys_utter = turn['system_utterance']

      # normalize utterances
      user_utter = ' '.join(normalize(user_utter, multiwoz))
      sys_utter = ' '.join(normalize(sys_utter, multiwoz))

      # get the changed slots in this turn
      turn_slots = turn['turn_slots']

      # Get system actions which will be used as the inform memory
      inform_mem = turn['system_act']

      # mask the system utterance by removing labels appeared in system acts
      sys_utter = ' '.join(mask_utterance(sys_utter, inform_mem, multiwoz, '[UNK]'))

      # append current instance
      raw_data.append(RawDataInstance(dial_idx,
                                      turn_idx,
                                      user_utter,
                                      sys_utter,
                                      history,
                                      turn_slots,
                                      inform_mem))

      # update history & last state for next turn
      history = [user_utter, sys_utter] + history

  return raw_data


def create_slot_span(input, target_value, tok_input_offsets, padding_len, label_maps):
  """
  Create a slot span given the input, target value, and token input offsets, 
  by matching the target value as tokens with the input sequence. 

  :param input: The input string.
  :type input: str

  :param target_value: The target value.
  :type target_value: str

  :param tok_input_offsets: The token input offsets.
  :type tok_input_offsets: list[tuple[int, int]]

  :param padding_len: The padding length.
  :type padding_len: int

  :param label_maps: The label maps.
  :type label_maps: dict

  :return: The slot span, span start index, and span end index.
  :rtype: tuple[list[int], int, int]
  """

  # get all possible variants of the slot value
  label_variants = [target_value]
  if target_value in label_maps:
    label_variants = label_variants + label_maps[target_value]

  # match the target value as tokens
  start, end = -1, -1
  found = False
  input_list = input.split()
  first_idx = input_list.index('[SEP]')
  max_idx = first_idx
  for label in label_variants:
    label_list = [item for item in map(str.strip, re.split("(\W+)", label)) if len(item) > 0]
    if found == True:
      break
    for idx in (j for j, e in enumerate(input_list) if(e == label_list[0] and j < max_idx)):
      if input_list[idx:idx + len(label_list)] == label_list:
        start, end = idx, idx + len(label_list) - 1
        found = True

  # mark the selected part as characters in the input
  input = " ".join(input_list)
  ch_start, ch_end = -1, -1
  acc_len = 0
  for idx, tok in enumerate(input_list):
    if start == idx:
      ch_start = acc_len + idx
    acc_len += len(tok)
    if end == idx:
      ch_end = acc_len + idx - 1

  # mark the target span in the input string
  char_target = [0] * len(input)
  if ch_start != -1 and ch_end != -1:
    for j in range(ch_start, ch_end + 1):
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
  """
  Create inputs for BERT using the history, user utterance, system utterance,
  by creating and tokenizing the input seqence and creating the masks.

  :param history: The history of utterances.
  :type history: list[str]

  :param user_utter: The user's utterance.
  :type user_utter: str

  :param sys_utter: The system's utterance.
  :type sys_utter: str

  :param tokenizer: The tokenizer to tokenize the input.
  :type tokenizer: transformers.PreTrainedTokenizer

  :param max_len: The maximum length of the input.
  :type max_len: int

  :return: The input string, token IDs, attention mask, token type IDs,
            token input offsets, tokenized input tokens, and padding length.
  :rtype: tuple[str, list[int], list[int], list[int], list[tuple[int, int]],
                list[str], int]
  """


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


def is_informed(value, target, label_maps, multiwoz):
  """
  Check if a value is informed by the system given a target value and label maps.

  :param value: The value to check.
  :type value: str

  :param target: The target value.
  :type target: str

  :param label_maps: The label maps.
  :type label_maps: dict

  :return: A tuple indicating if the value is informed and the informed value.
  :rtype: tuple[bool, str]
  """

  informed = False
  informed_value = 'none'

  target = ' '.join(normalize(target, multiwoz))

  if value == target or is_included(value, target) or is_included(target, value):
    informed = True
  if value in label_maps:
    informed = included_with_label_maps(target, value, label_maps)
  elif target in label_maps:
    informed = included_with_label_maps(value, target, label_maps)
  if informed: informed_value = value

  return informed, informed_value


def get_refered_slot(target_value, slot, last_state, non_referable_slots, non_referable_pairs, label_maps={}):
    """
    Get the referred slot if the user refers to another slot in the dialogue state given a target value, slot, last state, 
    non-referable slots, non-referable pairs, and label maps.

    :param target_value: The target value.
    :type target_value: str

    :param slot: The slot to check.
    :type slot: str

    :param last_state: The last state.
    :type last_state: dict

    :param non_referable_slots: The list of non-referable slots.
    :type non_referable_slots: list[str]

    :param non_referable_pairs: The list of non-referable slot pairs.
    :type non_referable_pairs: list[tuple[str, str]]

    :param label_maps: The label maps.
    :type label_maps: dict, optional

    :return: The referred slot.
    :rtype: str
    """

    referred_slot = 'none'

    if slot in non_referable_slots:
        return referred_slot

    if slot in last_state and last_state[slot] == target_value:
      return referred_slot

    for s in last_state:

        if s in non_referable_slots:
            continue

        if ((slot, s) in non_referable_pairs) or ((s, slot) in non_referable_pairs):
          continue

        if slot == s:
          continue

        if match_with_label_maps(last_state[s], target_value, label_maps):
            referred_slot = s
            break

    return referred_slot


def create_labels(target_value, slot, last_state, input, tok_input_offsets, inform_mem, label_maps, padding_len, max_len, non_referable_slots, non_referable_pairs, multiwoz):
  """
  Create the target operation and the span labels for a slot.

  :param target_value: The target value.
  :type target_value: str

  :param slot: The slot.
  :type slot: str

  :param last_state: The last state.
  :type last_state: dict

  :param input: The input string.
  :type input: str

  :param tok_input_offsets: The token input offsets.
  :type tok_input_offsets: list[tuple[int, int]]

  :param inform_mem: The inform memory.
  :type inform_mem: dict

  :param label_maps: The label maps.
  :type label_maps: dict

  :param padding_len: The padding length.
  :type padding_len: int

  :param max_len: The maximum length of the input.
  :type max_len: int

  :param non_referable_slots: The list of non-referable slots (slots that can not use refering).
  :type non_referable_slots: list[str]

  :param non_referable_pairs: The list of non-referable slot pairs (slots pairs that can not refer to each other).
  :type non_referable_pairs: list[tuple[str, str]]

  :return: The operation, span, span start index, span end index, referred slot, and informed value.
  :rtype: tuple[str, list[int], int, int, str, str]
  """

  oper = 'carryover'
  span = [0] * max_len
  span_start = 0
  span_end = 0
  refered_slot = 'none'
  informed_value = 'none'

  # assert target_value != 'none', 'target value can not be none'

  if target_value in ['[NULL]', 'none']:
    oper = 'carryover'
  elif target_value in ['dontcare', 'yes', 'no']:
    oper = target_value
  else:
    span, span_start, span_end = create_slot_span(input,
                                                  target_value,
                                                  tok_input_offsets,
                                                  padding_len,
                                                  label_maps)

    informed = False
    if slot in inform_mem:
      assert len(inform_mem[slot]) == 1, 'greater than 1'
      informed, informed_value = is_informed(inform_mem[slot][0], target_value, label_maps, multiwoz)

    refered_slot = get_refered_slot(target_value, slot, last_state, non_referable_slots, non_referable_pairs, label_maps)

    if sum(span) != 0:
      oper = 'update'
    elif informed == True:
      oper = 'inform'
    elif refered_slot != 'none':
      oper = 'refer'
    else:
      oper = 'unpointable'

  return oper, span, span_start, span_end, refered_slot, informed_value


def create_data(raw_data, slot_list, label_maps, tokenizer, max_len, non_referable_slots, non_referable_pairs, multiwoz):
  """
  Create the data instances for training or evaluation.

  :param raw_data: The list of raw data instances.
  :type raw_data: list[RawDataInstance]

  :param slot_list: The list of slots.
  :type slot_list: list[str]

  :param label_maps: The label maps.
  :type label_maps: dict

  :param tokenizer: The tokenizer to tokenize the input.
  :type tokenizer: transformers.PreTrainedTokenizer

  :param max_len: The maximum length of the input.
  :type max_len: int

  :param non_referable_slots: The list of non-referable slots.
  :type non_referable_slots: list[str]

  :param non_referable_pairs: The list of non-referable slot pairs.
  :type non_referable_pairs: list[tuple[str, str]]

  :return: The list of data instances.
  :rtype: list[DataInstance]
  """

  data = []

  last_state = {}
  cur_state = {}
  prev_dial_idx = -1
  # loop over raw data
  for turn in tqdm(raw_data):

    # if new dialogue reset the state
    if turn.dial_idx != prev_dial_idx or turn.turn_idx == 0:
      cur_state = {}
      last_state = {}

    # update previous dialogue index
    prev_dial_idx = turn.dial_idx

    # create model inputs
    input, ids, mask, token_type_ids, tok_input_offsets, input_tokens, padding_len = create_inputs(turn.history,
                                                                                                   turn.user_utter,
                                                                                                   turn.sys_utter,
                                                                                                   tokenizer,
                                                                                                   max_len)


    target_values = []
    opers = []
    spans = []
    spans_start = []
    spans_end = []
    refer = ['none'] * len(slot_list)
    inform_aux_features = [0] * len(slot_list)
    ds_aux_features = [0] * len(slot_list)

    # for each slot determine its values
    for slot_idx, slot in enumerate(slot_list):

      # get the slot target value
      target_value = '[NULL]'
      if slot in turn.turn_slots:
        target_value = turn.turn_slots[slot]
      elif slot in cur_state:
        target_value = cur_state[slot]


      # get slot labels
      (oper,
       span,
       span_start,
       span_end,
       refered_slot,
       informed_value) = create_labels(target_value,
                                      slot,
                                      last_state,
                                      input,
                                      tok_input_offsets,
                                      turn.inform_mem,
                                      label_maps,
                                      padding_len,
                                      max_len,
                                      non_referable_slots,
                                      non_referable_pairs,
                                      multiwoz)

      if slot in cur_state and target_value == cur_state[slot] and oper in ['dontcare', 'yes', 'no', 'refer']:
        oper = 'carryover'


      # create auxiliary features
      # mark each informed slot as 1
      if slot in turn.inform_mem:
        inform_aux_features[slot_idx] = 1
      # mark each filled slot as 1
      if slot in cur_state:
        ds_aux_features[slot_idx] = 1

      # update the state
      if oper != 'carryover':
        cur_state[slot] = target_value
        if oper == 'unpointable':
          oper = 'carryover'

#       if turn.dial_idx == 'MUL2491.json' and turn.turn_idx == 8 and slot == 'restaurant-name':
#         print(oper)
#         print(span)
#         print(refered_slot)
#         print(informed_value)
#         print(last_state)
#         print(cur_state)

      target_values.append(target_value)
      opers.append(oper)
      spans.append(span)
      spans_start.append(span_start)
      spans_end.append(span_end)
      refer[slot_idx] = refered_slot

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
                             target_values,
                             last_state.copy(),
                             cur_state.copy(),
                             refer,
                             inform_aux_features,
                             ds_aux_features))

    # update last state
    last_state = cur_state.copy()


  return data


def prepare_data(data_path, slot_list, label_maps, tokenizer, max_len, domains, non_referable_slots, non_referable_pairs, multiwoz):
  """
  Prepare the data for training or evaluation, this usually the function you want to call to preprocess the data for
  TripPy model, it encapsulates the whole process of preprcessing the data by calling the other functions in this
  module.

  :param data_path: The path to the JSON data file.
  :type data_path: str

  :param slot_list: The list of slots.
  :type slot_list: list[str]

  :param label_maps: The label maps.
  :type label_maps: dict

  :param tokenizer: The tokenizer to tokenize the input.
  :type tokenizer: transformers.PreTrainedTokenizer

  :param max_len: The maximum length of the input.
  :type max_len: int

  :param domains: The list of domains.
  :type domains: list[str]

  :param non_referable_slots: The list of non-referable slots.
  :type non_referable_slots: list[str]

  :param non_referable_pairs: The list of non-referable slot pairs.
  :type non_referable_pairs: list[tuple[str, str]]

  :return: The raw data and prepared data.
  :rtype: tuple[list[RawDataInstance], list[DataInstance]]
  """


  # create raw data
  raw_data = read_raw_data(data_path, slot_list, max_len, domains, multiwoz)

  # create data
  data = create_data(raw_data, slot_list, label_maps, tokenizer, max_len, non_referable_slots, non_referable_pairs, multiwoz)

  return raw_data, data


class Dataset(torch.utils.data.Dataset):
  """
  PyTorch Dataset for the TRIPPY model.
  
  :param data: The list of data instances.
  :type data: list[DataInstance]

  :param n_slots: The number of slots.
  :type n_slots: int

  :param oper2id: The mapping of operations to IDs.
  :type oper2id: dict[str, int]

  :param slot_list: The list of slots.
  :type slot_list: list[str]
  """

  def __init__(self, data, n_slots, oper2id, slot_list):

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
    # get the index of the refered slot, in case the slot is not present in the slot_list then that means "none"
    # index of "none" is n_slots
    self.refer = [[(slot_list.index(r) if r in slot_list else n_slots) for r in turn.refer] for turn in data]
    self.inform_aux_features = [turn.inform_aux_features for turn in data]
    self.ds_aux_features = [turn.ds_aux_features for turn in data]


  def __len__(self):
    """
    Get the length of the dataset.

    :return: The length of the dataset.
    :rtype: int
    """
    return len(self.ids)

  def __getitem__(self, idx):
    """
    Get an item from the dataset at the given index.

    :param idx: The index of the item.
    :type idx: int

    :return: The item at the given index.
    :rtype: dict[str, torch.Tensor or str]
    """
    return {
        'ids': torch.tensor(self.ids[idx], dtype=torch.long),
        'mask': torch.tensor(self.mask[idx], dtype=torch.long),
        'token_type_ids': torch.tensor(self.token_type_ids[idx], dtype=torch.long),
        'spans_start': torch.tensor(self.spans_start[idx], dtype=torch.long),
        'spans_end': torch.tensor(self.spans_end[idx], dtype=torch.long),
        'padding_len': torch.tensor(self.padding_len[idx], dtype=torch.long),
        'input_tokens': self.input_tokens[idx],
        'target_values': self.target_values[idx],
        'opers': torch.tensor(self.opers[idx], dtype=torch.long),
        'refer': torch.tensor(self.refer[idx], dtype=torch.long),
        'inform_aux_features': torch.tensor(self.inform_aux_features[idx], dtype=torch.float),
        'ds_aux_features': torch.tensor(self.ds_aux_features[idx], dtype=torch.float)
    }