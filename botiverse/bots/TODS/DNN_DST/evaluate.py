import torch
import numpy as np
import string
from sklearn.metrics import f1_score
from botiverse.TODS.DNN_DST.utils import normalize_text
import copy
from tqdm import tqdm

def jaccard(str1, str2): 
  a = set(str1.lower().split()) 
  b = set(str2.lower().split()) 
  c = a.intersection(b) 
  return float(len(c)) / (len(a) + len(b) - len(c))

def create_span_output(output_start, output_end, padding_len, input_tokens):

  mask = [0] * (len(output_start) - padding_len)

  if padding_len > 0:
    idx_start = np.argmax(output_start[1:-padding_len]) + 1
    idx_end = np.argmax(output_end[1:-padding_len]) + 1
  else:
    idx_start = np.argmax(output_start[1:]) + 1
    idx_end = np.argmax(output_end[1:]) + 1

  for mj in range(idx_start, idx_end + 1):
    mask[mj] = 1

  output_tokens = [x for p, x in enumerate(input_tokens.split()) if mask[p] == 1]
  output_tokens = [x for x in output_tokens if x not in ('[CLS]', '[SEP]')]

  final_output = ''
  for ot in output_tokens:
    if ot.startswith('##'):
      final_output = final_output + ot[2:]
    elif len(ot) == 1 and ot in string.punctuation:
      final_output = final_output + ot
    elif len(final_output) > 0 and final_output[-1] in string.punctuation:
      final_output = final_output + ot
    else:
      final_output = final_output + " " + ot

  final_output = final_output.strip()
  
  return final_output

def eval_f1_jac(data_loader, model, device, n_slots):
  
  model.eval()

  jaccards = []
  y_true, y_pred = [], []

  with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(data_loader)):
      
      ids = batch['ids'].to(device)
      mask = batch['mask'].to(device)
      token_type_ids = batch['token_type_ids'].to(device)
      input_tokens = batch['input_tokens']
      padding_len = batch['padding_len']
      target_values = batch['target_values']
      spans_start = batch['spans_start']
      spans_end = batch['spans_end']
      opers = batch['opers']

      slots_start_logits, slots_end_logits, slots_oper_logits = model(ids=ids,
                                                                      mask=mask,
                                                                      token_type_ids=token_type_ids)
      
      for b in range(len(ids)):
        for slot in range(n_slots):

          final_output = create_span_output(slots_start_logits[slot][b].cpu().detach().numpy(),
                                            slots_end_logits[slot][b].cpu().detach().numpy(),
                                            padding_len[b],
                                            input_tokens[b])

          target_value = target_values[b].split('[VALUESEP]')[slot].strip()
          jac = jaccard(target_value, final_output.strip())
          if spans_start[b][slot] != -100 and spans_start[b][slot] != 0:
            jaccards.append(jac)

      for slot in range(n_slots):
        pred = slots_oper_logits[slot].argmax(dim=-1)
        y_pred.extend(pred.tolist())
        y_true.extend(opers[:,slot].tolist())
  
  mean_jac = np.mean(jaccards)
  macro_f1_score = f1_score(y_true, y_pred, average='macro')
  all_f1_score = f1_score(y_true, y_pred, average=None)

  return mean_jac, macro_f1_score, all_f1_score

def right_state(pred_state, true_state, label_maps):
  
  if len(pred_state) != len(true_state):
    return False

  for slot, value in true_state.items():
    variant_labels = [value]
    if value in label_maps:
      variant_labels += label_maps[value]
    #
    for i in range(len(variant_labels)):
      variant_labels[i] = normalize_text(variant_labels[i])
    if slot in pred_state:
      pred_state[slot] = normalize_text(pred_state[slot])
    #
    if slot not in pred_state or pred_state[slot] not in variant_labels:
      return False

  return True

def eval_joint(raw_data, data, model, device, n_slots, slot_list, label_maps):
  
  model.eval()

  pred_last_state = {}
  joint_goal_acc = 0
  states = []
  sentences = []
  indices = []
  pre_dialogue_idx = -1
  with torch.no_grad():

    for raw_turn, turn in tqdm(zip(raw_data, data), total=len(raw_data)):

      ids = torch.tensor(turn.ids, dtype=torch.long).unsqueeze(0).to(device)
      mask = torch.tensor(turn.mask, dtype=torch.long).unsqueeze(0).to(device)
      token_type_ids = torch.tensor(turn.token_type_ids, dtype=torch.long).unsqueeze(0).to(device)
      input_tokens = ' '.join(turn.input_tokens)
      padding_len = turn.padding_len
      turn_idx = raw_turn.turn_idx
      dialogue_idx = raw_turn.dial_idx
      current_state = raw_turn.cur_state

      # if dialogue_idx == 'PMUL2075.json' and turn_idx == 0:
      #   print(ids)
      #   print(mask)
      #   print(token_type_ids)
      #   print(input_tokens)
      #   print(padding_len)
      #   print(turn_idx)
      #   print(dialogue_idx)
      #   print(current_state)
      #   print(pred_last_state)


      if turn_idx == 0 or dialogue_idx != pre_dialogue_idx:
        pred_last_state = {}


      # if dialogue_idx == 'PMUL2075.json' and turn_idx == 0:
      #   print(pred_last_state)


      slots_start_logits, slots_end_logits, slots_oper_logits = model(ids=ids,
                                                                      mask=mask,
                                                                      token_type_ids=token_type_ids)

      pred_current_state = pred_last_state
      for slot in range(n_slots):

        pred_oper = slots_oper_logits[slot][0].argmax(dim=-1).item()

        if pred_oper == 0: # carryover
          continue
        elif pred_oper == 1: # dontcare
          pred_current_state[slot_list[slot]] = 'dontcare'
        elif pred_oper == 2: # update
          pred_current_state[slot_list[slot]] = create_span_output(slots_start_logits[slot][0].cpu().detach().numpy(),
                                                                   slots_end_logits[slot][0].cpu().detach().numpy(),
                                                                   padding_len,
                                                                   input_tokens)
        elif pred_oper == 3: # delete
          if slot_list[slot] in pred_current_state:
            pred_current_state.pop(slot_list[slot])
        elif pred_oper == 4: # yes
          pred_current_state[slot_list[slot]] = 'yes'
        elif pred_oper == 5: # no
          pred_current_state[slot_list[slot]] = 'no'

      if right_state(pred_current_state, current_state, label_maps) == True:
        joint_goal_acc += 1
      
      # if dialogue_idx == 'PMUL2075.json' and turn_idx == 0:
      #   print(pred_current_state)
      #   print(right_state(pred_current_state, current_state, label_maps))

      if right_state(pred_current_state, current_state, label_maps) == False:
        states.append((copy.deepcopy(pred_current_state), current_state))
        sentences.append(input_tokens)
        indices.append((dialogue_idx, turn_idx))

      pred_last_state = pred_current_state
      pre_dialogue_idx = dialogue_idx

  return joint_goal_acc / len(data), states, sentences, indices