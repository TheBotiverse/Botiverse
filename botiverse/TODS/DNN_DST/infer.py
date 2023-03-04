from botiverse.TODS.DNN_DST.config import *
import torch
from botiverse.TODS.DNN_DST.data import create_inputs
from botiverse.TODS.DNN_DST.evaluate import create_span_output

def infer(model, slot_list, current_state, history, sys_utter, user_utter, device):
  
  model.eval()

  # turn data to inputs
  input, ids, mask, token_type_ids, tok_input_offsets, input_tokens, padding_len = create_inputs(history,
                                                                                                 user_utter,
                                                                                                 sys_utter,
                                                                                                 TOKENIZER,
                                                                                                 MAX_LEN)
  # print(input, ids, mask, token_type_ids, tok_input_offsets, input_tokens, padding_len)

  n_slots = len(slot_list)

  with torch.no_grad():
    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0).to(device)
    input_tokens = ' '.join(input_tokens)
    padding_len = padding_len

    slots_start_logits, slots_end_logits, slots_oper_logits = model(ids=ids,
                                                                    mask=mask,
                                                                    token_type_ids=token_type_ids)

    for slot in range(n_slots):

      pred_oper = slots_oper_logits[slot][0].argmax(dim=-1).item()

      if pred_oper == 0: # carryover
        continue
      elif pred_oper == 1: # dontcare
        current_state[slot_list[slot]] = 'dontcare'
      elif pred_oper == 2: # update
        current_state[slot_list[slot]] = create_span_output(slots_start_logits[slot][0].cpu().detach().numpy(),
                                                            slots_end_logits[slot][0].cpu().detach().numpy(),
                                                            padding_len,
                                                            input_tokens)
      elif pred_oper == 3: # delete
        if slot_list[slot] in current_state:
          current_state.pop(slot_list[slot])
      elif pred_oper == 4: # yes
        current_state[slot_list[slot]] = 'yes'
      elif pred_oper == 5: # no
        current_state[slot_list[slot]] = 'no'

  return current_state