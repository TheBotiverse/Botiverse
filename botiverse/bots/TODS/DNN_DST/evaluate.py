import torch
import numpy as np
from sklearn.metrics import f1_score
import copy
from tqdm import tqdm

from botiverse.TODS.DNN_DST.utils import normalize, is_included, included_with_label_maps, create_span_output
from botiverse.TODS.DNN_DST.config import *


def get_informed_value(value, target, label_maps):
  informed = False
  informed_value = value

  value = ' '.join(normalize(value))
  target = ' '.join(normalize(target))

  if value == target or is_included(value, target) or is_included(target, value):
    informed = True
  if value in label_maps:
    informed = included_with_label_maps(target, value, label_maps)
  elif target in label_maps:
    informed = included_with_label_maps(value, target, label_maps)
  if informed: informed_value = target

  return informed_value


def eval(raw_data, data, model, device, n_slots, slot_list, label_maps):

  model.eval()

  # normalize label_maps
  label_maps_tmp = {}
  for v in label_maps:
      label_maps_tmp[' '.join(normalize(v))] = [' '.join(normalize(nv)) for nv in label_maps[v]]
  label_maps = label_maps_tmp


  # metrics
  Y_true, Y_pred = [], []
  per_slot_acc = {slot:[] for slot in slot_list}
  joint_goal_acc = []

  # state
  pred_last_state = {}
  pre_dialogue_idx = -1

  # debugging
  states = []
  sentences = []
  indices = []
  prev_idx = -1

  with torch.no_grad():

    for raw_turn, turn in tqdm(zip(raw_data, data), total=len(raw_data)):

      ids = torch.tensor(turn.ids, dtype=torch.long).unsqueeze(0).to(device)
      mask = torch.tensor(turn.mask, dtype=torch.long).unsqueeze(0).to(device)
      token_type_ids = torch.tensor(turn.token_type_ids, dtype=torch.long).unsqueeze(0).to(device)
      inform_aux_features = torch.tensor(turn.inform_aux_features, dtype=torch.float).unsqueeze(0).to(device)
      # ds_aux_features = torch.tensor(turn.ds_aux_features, dtype=torch.float).unsqueeze(0).to(device)
      input_tokens = ' '.join(turn.input_tokens)
      padding_len = turn.padding_len
      turn_idx = raw_turn.turn_idx
      dialogue_idx = raw_turn.dial_idx
      current_state = turn.cur_state
      inform_mem = raw_turn.inform_mem
      opers = turn.opers

      # new dialogue reset the state and the state auxiliary features
      if turn_idx == 0 or dialogue_idx != pre_dialogue_idx:
        pred_last_state = {}
        ds_aux_features = torch.zeros((1, n_slots)).to(device)

      # get model outputs
      slots_start_logits, slots_end_logits, slots_oper_logits, slots_refer_logits = model(ids=ids,
                                                                                          mask=mask,
                                                                                          token_type_ids=token_type_ids,
                                                                                          inform_aux_features=inform_aux_features,
                                                                                          ds_aux_features=ds_aux_features)

      # update the predicted state of each slot
      pred_current_state = pred_last_state.copy()
      for slot_idx, slot in enumerate(slot_list):

        # get the predicted operation
        pred_oper = slots_oper_logits[slot_idx][0].argmax(dim=-1).item()

        # keep track of operations for f1 score
        Y_pred.append(pred_oper)
        Y_true.append(OPER2ID[opers[slot_idx]])

        # update the slot based on the operation
        if pred_oper == OPER2ID['carryover']: # carryover
          continue
        elif pred_oper == OPER2ID['dontcare']: # dontcare
          pred_current_state[slot] = 'dontcare'
        elif pred_oper == OPER2ID['update']: # update
          pred_current_state[slot] = create_span_output(slots_start_logits[slot_idx][0].cpu().detach().numpy(),
                                                        slots_end_logits[slot_idx][0].cpu().detach().numpy(),
                                                        padding_len,
                                                        input_tokens)
        elif pred_oper == OPER2ID['refer']: # refer
          refered_slot = slots_refer_logits[slot_idx][0].argmax(dim=-1).item()
          if refered_slot != n_slots and slot_list[refered_slot] in pred_last_state:
            pred_current_state[slot] = pred_last_state[slot_list[refered_slot]]
        elif pred_oper == OPER2ID['yes']: # yes
          pred_current_state[slot] = 'yes'
        elif pred_oper == OPER2ID['no']: # no
          pred_current_state[slot] = 'no'
        elif pred_oper == OPER2ID['inform']: # inform
          if slot in inform_mem:
            pred_current_state[slot] = '§§' + inform_mem[slot][0]

      # update the state auxiliary features
      for slot_idx, slot in enumerate(slot_list):
          ds_aux_features[0, slot_idx] = 1 if slot in pred_current_state else 0

      # calculate accuracy
      joint = True
      for slot_idx, slot in enumerate(slot_list):

        # if not present in both
        if slot not in current_state and slot not in pred_current_state:
          per_slot_acc[slot].append(1.0)
          continue

        # if slot only in one of them then mark as 0
        if (slot in current_state and slot not in pred_current_state) or (slot not in current_state and slot in pred_current_state):
          joint = False
          per_slot_acc[slot].append(0.0)
          continue

        # get values
        val = current_state[slot]
        pred_val = pred_current_state[slot]

        # normalize values
        val = ' '.join(normalize(val))
        pred_val = ' '.join(normalize(pred_val))

        # handle inform
        if pred_val[0:3] == "§§ ":
          if pred_val[3:] != 'none':
              pred_val = get_informed_value(pred_val[3:], val, label_maps)
        elif pred_val[0:2] == "§§":
            if pred_val[2:] != 'none':
                pred_val = get_informed_value(pred_val[2:], val, label_maps)

        # match
        if pred_val == val:
          per_slot_acc[slot].append(1.0)
        elif val != 'none' and val != 'dontcare' and val != 'true' and val != 'false' and val in label_maps:
          no_match = True
          for variant in label_maps[val]:
              if variant == pred_val:
                  no_match = False
                  break
          if no_match:
              per_slot_acc[slot].append(0.0)
              joint = False
          else:
              per_slot_acc[slot].append(1.0)
        else:
            per_slot_acc[slot].append(0.0)
            joint = False

      # append joint
      joint_goal_acc.append(1.0 if joint else 0.0)


      # update vars for next turn
      pred_last_state = pred_current_state.copy()
      pre_dialogue_idx = dialogue_idx

#       # debugging
#       if per_slot_acc['attraction-name'][-1] < 0.99 and prev_idx != dialogue_idx:
#         print('dialogue_idx', dialogue_idx)
#         print('turn_idx', turn_idx)
#         print('pred_state', dict(sorted(pred_current_state.items())))
#         print('cur_state', dict(sorted(current_state.items())))
#         print('input tok', input_tokens)
#         print('inform_mem', inform_mem)
#         print('inform aux', inform_aux_features)
#         print('oper', opers[slot_list.index('attraction-name')])
#         prev_idx = dialogue_idx

      # debugging
      if joint == False:
        states.append((copy.deepcopy(pred_current_state), current_state))
        sentences.append(input_tokens)
        indices.append((dialogue_idx, turn_idx))

  # debugging
  # prev = ""
  # for i in range(len(states)):
  #   if prev == indices[i][0] or len(states[i][0]) != len(states[i][1]):
  #     continue
  #   if 'attraction-name' not in states[i][1]:# and 'restaurant-name' not in states[i][1]:
  #     continue
  #   prev = indices[i][0]
  #   print(dict(sorted(states[i][0].items())))
  #   print(dict(sorted(states[i][1].items())))
  #   print(sentences[i])
  #   print(indices[i])
  #   print("\n")

  # calculate per slot accuracy
  per_slot_acc = {slot: np.mean(acc) for slot, acc in per_slot_acc.items()}

  # calculate f1 scores
  macro_f1_score = f1_score(Y_true, Y_pred, average='macro')
  all_f1_score = f1_score(Y_true, Y_pred, average=None)

  return np.mean(joint_goal_acc), per_slot_acc, macro_f1_score, all_f1_score