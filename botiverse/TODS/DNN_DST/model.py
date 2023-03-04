import torch.nn as nn
from transformers import BertModel


class DSTModel(nn.Module):

  def __init__(self, n_slots, hid_dim, n_oper, dropout):
    super(DSTModel, self).__init__()

    self.hid_dim = hid_dim
    self.n_oper = n_oper
    self.n_slots = n_slots

    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.oper_layers = nn.ModuleList([nn.Linear(hid_dim, n_oper) for _ in range(n_slots)])
    self.span_layers = nn.ModuleList([nn.Linear(hid_dim, 2) for _ in range(n_slots)])
    self.dropout = nn.Dropout(dropout)

  def forward(self, ids, mask, token_type_ids):

    #ids = [batch size, seq len]
    #mask = [batch size, seq len]
    #token_type_ids = [batch size, seq len]

    #sequence_output = [batch size, seq len, hid_dim]
    #pooled_output = [batch size, seq len, 1]
    sequence_output, pooled_output = self.bert(ids, 
                                               attention_mask=mask,
                                               token_type_ids=token_type_ids,
                                               return_dict=False
                                               )
    
    sequence_output = self.dropout(sequence_output)
    pooled_output = self.dropout(pooled_output)

    slots_start_logits = []
    slots_end_logits = []
    slots_oper_logits = []
    for slot in range(self.n_slots):
      
      # oper_logits = [batch size, seq len, n_oper]
      oper_logits = self.oper_layers[slot](pooled_output)
      
      # span_logits = [batch size, seq len, 2]
      span_logits = self.span_layers[slot](sequence_output)

      # start_logits = [batch size, seq len, 1]
      # end_logits = [batch size, seq len, 1]
      start_logits, end_logits = span_logits.split(1, dim=-1)

      # start_logits = [batch size, seq len]
      # end_logits = [batch size, seq len]
      start_logits = start_logits.squeeze(-1)
      end_logits = end_logits.squeeze(-1)

      slots_start_logits.append(start_logits)
      slots_end_logits.append(end_logits)
      slots_oper_logits.append(oper_logits)

    return slots_start_logits, slots_end_logits, slots_oper_logits