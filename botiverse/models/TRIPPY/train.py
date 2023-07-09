"""
This Module has the training functions for TRIPPY.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from botiverse.models.TRIPPY.utils import AverageMeter


def span_loss_fn(start_logits, end_logits, targets_start, targets_end, ignore_idx):
  """
  Compute the span loss.

  :param start_logits: The start logits.
  :type start_logits: torch.Tensor
  :param end_logits: The end logits.
  :type end_logits: torch.Tensor
  :param targets_start: The start targets.
  :type targets_start: torch.Tensor
  :param targets_end: The end targets.
  :type targets_end: torch.Tensor
  :param ignore_idx: The index to ignore in the loss calculation.
  :type ignore_idx: int
  :return: The span loss.
  :rtype: torch.Tensor
  """
  l1 = nn.CrossEntropyLoss(ignore_index=ignore_idx, reduction='none')(start_logits, targets_start)
  l2 = nn.CrossEntropyLoss(ignore_index=ignore_idx, reduction='none')(end_logits, targets_end)
  return (l1 + l2) / 2.0

def oper_loss_fn(oper_logits, oper_labels, ignore_idx):
  """
  Compute the operation loss.

  :param oper_logits: The operation logits.
  :type oper_logits: torch.Tensor
  :param oper_labels: The operation labels.
  :type oper_labels: torch.Tensor
  :param ignore_idx: The index to ignore in the loss calculation.
  :type ignore_idx: int
  :return: The operation loss.
  :rtype: torch.Tensor
  """
  l = nn.CrossEntropyLoss(ignore_index=ignore_idx, reduction='none')(oper_logits, oper_labels)
  return l

def refer_loss_fn(refer_logits, refer_labels, ignore_idx):
  """
  Compute the refer loss.

  :param refer_logits: The refer logits.
  :type refer_logits: torch.Tensor
  :param refer_labels: The refer labels.
  :type refer_labels: torch.Tensor
  :param ignore_idx: The index to ignore in the loss calculation.
  :type ignore_idx: int
  :return: The refer loss.
  :rtype: torch.Tensor
  """
  l = nn.CrossEntropyLoss(ignore_index=ignore_idx, reduction='none')(refer_logits, refer_labels)
  return l

def train(data_loader, model, optimizer, device, scheduler, n_slots, ignore_idx, oper2id):
  """
  Perform the training loop for a model on the given data.

  :param data_loader: The data loader providing the training batches.
  :type data_loader: DataLoader

  :param model: The model to be trained.
  :type model: nn.Module

  :param optimizer: The optimizer used to update the model's parameters.
  :type optimizer: Optimizer

  :param device: The device (e.g., CPU or GPU) on which the training will be performed.
  :type device: torch.device

  :param scheduler: The scheduler for adjusting the learning rate during training.
  :type scheduler: _LRScheduler

  :param n_slots: The number of slots in the task.
  :type n_slots: int

  :param ignore_idx: The index to ignore during loss computation.
  :type ignore_idx: int

  :param oper2id: A dictionary mapping operation names to their corresponding IDs.
  :type oper2id: dict
  """

  model.train()

  losses = AverageMeter()

  tk0 = tqdm(data_loader)
  for i, batch in enumerate(tk0):

    ids = batch['ids'].to(device)
    mask = batch['mask'].to(device)
    token_type_ids = batch['token_type_ids'].to(device)
    spans_start = batch['spans_start'].to(device)
    spans_end = batch['spans_end'].to(device)
    opers = batch['opers'].to(device)
    refer = batch['refer'].to(device)
    inform_aux_features = batch['inform_aux_features'].to(device)
    ds_aux_features = batch['ds_aux_features'].to(device)

    optimizer.zero_grad()
    slots_start_logits, slots_end_logits, slots_oper_logits, slots_refer_logits = model(ids=ids,
                                                                                        mask=mask,
                                                                                        token_type_ids=token_type_ids,
                                                                                        inform_aux_features=inform_aux_features,
                                                                                        ds_aux_features=ds_aux_features)

    batch_loss = 0.0

    for slot in range(n_slots):

      oper_loss = oper_loss_fn(slots_oper_logits[slot], opers[:,slot], ignore_idx)

      span_loss = span_loss_fn(slots_start_logits[slot],
                               slots_end_logits[slot],
                               spans_start[:,slot],
                               spans_end[:,slot],
                               ignore_idx)
      token_is_pointable = (spans_start[:,slot] > 0).float()
      span_loss *= token_is_pointable

      refer_loss = refer_loss_fn(slots_refer_logits[slot], refer[:,slot], ignore_idx)
      token_is_referrable = (opers[:,slot] == oper2id['refer']).float()
      refer_loss *= token_is_referrable

      total_loss = 0.8 * oper_loss + 0.1 * span_loss + 0.1 * refer_loss

      batch_loss += total_loss.sum()

    # batch_loss /= n_slots
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    losses.update(batch_loss.item(), ids.size(0))

    tk0.set_postfix(loss=losses.avg)