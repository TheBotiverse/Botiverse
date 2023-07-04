import torch
import torch.nn as nn
from botiverse.TODS.DNN_DST.utils import AverageMeter
from tqdm import tqdm

def span_loss_fn(start_logits, end_logits, targets_start, targets_end, ignore_idx):
  l1 = nn.CrossEntropyLoss(ignore_index=ignore_idx, reduction='none')(start_logits, targets_start)
  l2 = nn.CrossEntropyLoss(ignore_index=ignore_idx, reduction='none')(end_logits, targets_end)
  return (l1 + l2) / 2.0

def oper_loss_fn(oper_logits, oper_labels, ignore_idx):
  l = nn.CrossEntropyLoss(ignore_index=ignore_idx, reduction='none')(oper_logits, oper_labels)
  return l

def train(data_loader, model, optimizer, device, scheduler, n_slots, ignore_idx):
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

    optimizer.zero_grad()
    slots_start_logits, slots_end_logits, slots_oper_logits = model(ids=ids,
                                                                    mask=mask,
                                                                    token_type_ids=token_type_ids)
    
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

      total_loss = 0.8 * oper_loss + 0.2 * span_loss 

      batch_loss += total_loss.sum()
    
    # batch_loss /= n_slots
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    losses.update(batch_loss.item(), ids.size(0))

    tk0.set_postfix(loss=losses.avg)