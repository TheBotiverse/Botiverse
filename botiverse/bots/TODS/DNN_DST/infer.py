import torch

from botiverse.TODS.DNN_DST.data import create_inputs
from botiverse.TODS.DNN_DST.utils import create_span_output
from botiverse.TODS.DNN_DST.config import *


def infer(model, slot_list, current_state, history, sys_utter, user_utter, inform_mem, device):

  model.eval()

  # turn data to inputs
  input, ids, mask, token_type_ids, tok_input_offsets, input_tokens, padding_len = create_inputs(history,
                                                                                                 user_utter,
                                                                                                 sys_utter,
                                                                                                 TOKENIZER,
                                                                                                 MAX_LEN)


  # print(input, ids, mask, token_type_ids, tok_input_offsets, input_tokens, padding_len)


  with torch.no_grad():
    n_slots = len(slot_list)
    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0).to(device)
    inform_aux_features = torch.zeros((1, n_slots)).to(device)
    ds_aux_features = torch.zeros((1, n_slots)).to(device)
    input_tokens = ' '.join(input_tokens)
    padding_len = padding_len

    for slot_idx, slot in enumerate(slot_list):
      if slot in inform_mem:
        inform_aux_features[0, slot_idx] = 1
      if slot in current_state:
        ds_aux_features[0, slot_idx] = 1

    # print(slot_list)
    # print(inform_aux_features)
    # print(inform_mem)


    # get model outputs
    slots_start_logits, slots_end_logits, slots_oper_logits, slots_refer_logits = model(ids=ids,
                                                                                        mask=mask,
                                                                                        token_type_ids=token_type_ids,
                                                                                        inform_aux_features=inform_aux_features,
                                                                                        ds_aux_features=ds_aux_features)


    # update the predicted state of each slot
    pred_state = current_state.copy()
    for slot_idx, slot in enumerate(slot_list):

      # get the predicted operation
      pred_oper = slots_oper_logits[slot_idx][0].argmax(dim=-1).item()
      # print(slot, torch.softmax(slots_oper_logits[slot_idx][0], dim=-1))

      # update the slot based on the operation
      if pred_oper == OPER2ID['carryover']: # carryover
        continue
      elif pred_oper == OPER2ID['dontcare']: # dontcare
        pred_state[slot] = 'dontcare'
      elif pred_oper == OPER2ID['update']: # update
        pred_state[slot] = create_span_output(slots_start_logits[slot_idx][0].cpu().detach().numpy(),
                                              slots_end_logits[slot_idx][0].cpu().detach().numpy(),
                                              padding_len,
                                              input_tokens)
      elif pred_oper == OPER2ID['refer']: # refer
        refered_slot = slots_refer_logits[slot_idx][0].argmax(dim=-1).item()
        if refered_slot != n_slots and slot_list[refered_slot] in current_state:
          pred_state[slot] = current_state[slot_list[refered_slot]]
      elif pred_oper == OPER2ID['yes']: # yes
        pred_state[slot] = 'yes'
      elif pred_oper == OPER2ID['no']: # no
        pred_state[slot] = 'no'
      elif pred_oper == OPER2ID['inform']: # inform
        if slot in inform_mem:
          pred_state[slot] = '§§' + inform_mem[slot][0]


  return pred_state