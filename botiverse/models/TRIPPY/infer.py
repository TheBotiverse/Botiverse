"""
This Module has the inference functions for TRIPPY.
"""


import torch

from botiverse.models.TRIPPY.data import create_inputs
from botiverse.models.TRIPPY.utils import create_span_output


def infer(model, slot_list, current_state, history, sys_utter, user_utter, inform_mem, device, oper2id, tokenizer, max_len):
  """
  Infer the dialogue state using the TRIPPY model.

  :param model: The TRIPPY model for inference.
  :type model: TRIPPY
  :param slot_list: The list of slots.
  :type slot_list: list
  :param current_state: The current dialogue state.
  :type current_state: dict
  :param history: The dialogue history.
  :type history: list
  :param sys_utter: The system's utterance.
  :type sys_utter: str
  :param user_utter: The user's utterance.
  :type user_utter: str
  :param inform_mem: The inform memory.
  :type inform_mem: dict
  :param device: The device to run the inference on.
  :type device: torch.device
  :param oper2id: The mapping of operations to IDs.
  :type oper2id: dict
  :param tokenizer: The tokenizer to tokenize the input.
  :type tokenizer: transformers.PreTrainedTokenizer
  :param max_len: The maximum length of the input sequence.
  :type max_len: int
  :return: The predicted dialogue state.
  :rtype: dict
  """

  model.eval()

  # turn data to inputs
  input, ids, mask, token_type_ids, tok_input_offsets, input_tokens, padding_len = create_inputs(history,
                                                                                                 user_utter,
                                                                                                 sys_utter,
                                                                                                 tokenizer,
                                                                                                 max_len)


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
      if pred_oper == oper2id['carryover']: # carryover
        continue
      elif pred_oper == oper2id['dontcare']: # dontcare
        pred_state[slot] = 'dontcare'
      elif pred_oper == oper2id['update']: # update
        pred_state[slot] = create_span_output(slots_start_logits[slot_idx][0].cpu().detach().numpy(),
                                              slots_end_logits[slot_idx][0].cpu().detach().numpy(),
                                              padding_len,
                                              input_tokens)
      elif pred_oper == oper2id['refer']: # refer
        refered_slot = slots_refer_logits[slot_idx][0].argmax(dim=-1).item()
        if refered_slot != n_slots and slot_list[refered_slot] in current_state:
          pred_state[slot] = current_state[slot_list[refered_slot]]
      elif pred_oper == oper2id['yes']: # yes
        pred_state[slot] = 'yes'
      elif pred_oper == oper2id['no']: # no
        pred_state[slot] = 'no'
      elif pred_oper == oper2id['inform']: # inform
        if slot in inform_mem:
          pred_state[slot] = '§§' + inform_mem[slot][0]


  return pred_state