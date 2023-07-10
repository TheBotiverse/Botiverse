"""
This Module has the TRIPPY model.
"""

import torch
import torch.nn as nn
from transformers import BertModel

from botiverse.models.BERT.BERT import Bert
from botiverse.models.BERT.config import BERTConfig
from botiverse.models.TRIPPY.config import TRIPPYConfig
from botiverse.models.BERT.utils import LoadPretrainedWeights

class TRIPPY(nn.Module):
  """
  TRIPPY (Task-oriented Reasoning and Inference for Pre-trained models with Pre-trained Ypesystem) model.

  This class implements the TRIPPY model for task-oriented dialogue understanding and slot filling.

  :param n_slots: The number of slots, corresponding to the number of dialogue slots to be filled.
  :type n_slots: int
  :param hid_dim: The hidden dimension size.
  :type hid_dim: int
  :param n_oper: The number of operations.
  :type n_oper: int
  :param dropout: The dropout rate.
  :type dropout: float
  :param from_scratch: Whether to build the BERT model from scratch or load pre-trained weights, defaults to False.
  :type from_scratch: bool
  :param BERT_config: The configuration for the BERT model, defaults to BERTConfig().
  :type BERT_config: BERTConfig
  """

  def __init__(self, n_slots, hid_dim, n_oper, dropout, from_scratch, BERT_config=BERTConfig(), TRIPPY_config=TRIPPYConfig()):
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
    """
    Forward pass of the TRIPPY model.

    :param ids: The input token IDs.
    :type ids: torch.Tensor, shape [batch size, seq len]
    :param mask: The attention mask indicating which tokens are valid.
    :type mask: torch.Tensor, shape [batch size, seq len]
    :param token_type_ids: The token type IDs.
    :type token_type_ids: torch.Tensor, shape [batch size, seq len]
    :param inform_aux_features: The auxiliary features for informing slots.
    :type inform_aux_features: torch.Tensor, shape [batch size, n_slots]
    :param ds_aux_features: The auxiliary features for dialogue state tracking.
    :type ds_aux_features: torch.Tensor, shape [batch size, n_slots]
    :return: Tuple containing the logits for slot start positions, slot end positions, slot operations, and slot references.
    :rtype: tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
    """
    
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