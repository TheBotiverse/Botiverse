import torch
import torch.nn as nn
from transformers import BertModel

from botiverse.models.BERT.BERT import Bert
from botiverse.models.BERT.config import BERTConfig
from botiverse.models.BERT.utils import LoadPretrainedWeights

class TRIPPY(nn.Module):

  def __init__(self, n_slots, hid_dim, n_oper, dropout, from_scratch, BERT_config=BERTConfig()):
    super(TRIPPY, self).__init__()

    self.hid_dim = hid_dim
    self.n_oper = n_oper
    self.n_slots = n_slots

    if from_scratch == True:
        # Build a BERT model from scratch
        self.bert = Bert(BERT_config)
        LoadPretrainedWeights(self.bert)
    else:
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    aux_dim = 2 * n_slots
    self.oper_layers = nn.ModuleList([nn.Linear(hid_dim + aux_dim, n_oper) for _ in range(n_slots)])
    self.span_layers = nn.ModuleList([nn.Linear(hid_dim, 2) for _ in range(n_slots)])
    self.refer_layers = nn.ModuleList([nn.Linear(hid_dim + aux_dim, n_slots + 1) for _ in range(n_slots)])
    self.dropout = nn.Dropout(dropout)

    # auxiliary features layers
    self.inform_aux_layer = nn.Linear(n_slots, n_slots)
    self.ds_aux_layer = nn.Linear(n_slots, n_slots)

  def forward(self, ids, mask, token_type_ids, inform_aux_features, ds_aux_features):

    # ids = [batch size, seq len]
    # mask = [batch size, seq len]
    # token_type_ids = [batch size, seq len]

    # sequence_output = [batch size, seq len, hid_dim]
    # pooled_output = [batch size, hid_dim]
    sequence_output, pooled_output = self.bert(ids,
                                               attention_mask=mask,
                                               token_type_ids=token_type_ids,
                                               return_dict=False
                                               )

    sequence_output = self.dropout(sequence_output)
    pooled_output = self.dropout(pooled_output)

    # concatenate the auxiliary features
    # pooled_output = [batch size, hid_dim + 2 * n_slots]
    # print(pooled_output.shape)
    # print(inform_aux_features.shape)
    # print(ds_aux_features.shape)
    pooled_output = torch.cat((pooled_output, self.inform_aux_layer(inform_aux_features), self.ds_aux_layer(ds_aux_features)), 1)
    # print(pooled_output.shape)
    # print("\n\n\n\n")

    slots_start_logits = []
    slots_end_logits = []
    slots_oper_logits = []
    slots_refer_logits = []
    for slot in range(self.n_slots):

      # oper_logits = [batch size, n_oper]
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

      # refer_logits = [batch size, n_slots + 1]
      refer_logits = self.refer_layers[slot](pooled_output)

      slots_start_logits.append(start_logits)
      slots_end_logits.append(end_logits)
      slots_oper_logits.append(oper_logits)
      slots_refer_logits.append(refer_logits)

    return slots_start_logits, slots_end_logits, slots_oper_logits, slots_refer_logits