import re
import string
import numpy as np

from botiverse.models.TRIPPY.config import MULTIWOZ


class RawDataInstance():
  def __init__(self,
               dial_idx,
               turn_idx,
               user_utter,
               sys_utter,
               history,
               turn_slots,
               inform_mem):
    self.dial_idx = dial_idx
    self.turn_idx = turn_idx
    self.user_utter = user_utter
    self.sys_utter = sys_utter
    self.history = history
    self.turn_slots = turn_slots
    self.inform_mem = inform_mem

  def __str__(self):
    string = ''
    string = string + '\ndial_idx: ' + str(self.dial_idx)
    string = string + '\nturn_idx: ' + str(self.turn_idx)
    string = string + '\nuser_utter: ' + str(self.user_utter)
    string = string + '\nsys_utter: ' + str(self.sys_utter)
    string = string + '\nhistory: ' + str(self.history)
    string = string + '\nturn_slots: ' + str(self.turn_slots)
    string = string + '\ninform_mem: ' + str(self.inform_mem)
    return string

class DataInstance():
  def __init__(self,
               ids,
               mask,
               token_type_ids,
               spans,
               spans_start,
               spans_end,
               padding_len,
               input_tokens,
               input,
               opers,
               target_values,
               last_state,
               cur_state,
               refer,
               inform_aux_features,
               ds_aux_features):
    self.ids = ids
    self.mask = mask
    self.token_type_ids = token_type_ids
    self.spans = spans
    self.spans_start = spans_start
    self.spans_end = spans_end
    self.padding_len = padding_len
    self.input_tokens = input_tokens
    self.input = input
    self.opers = opers
    self.target_values = target_values
    self.last_state = last_state
    self.cur_state = cur_state
    self.refer = refer
    self.inform_aux_features = inform_aux_features
    self.ds_aux_features = ds_aux_features

  def __str__(self):
    string = ''
    string = string + '\nids: ' + str(self.ids)
    string = string + '\nmask: ' + str(self.mask)
    string = string + '\ntoken_type_ids: ' + str(self.token_type_ids)
    string = string + '\nspans: ' + str(self.spans)
    string = string + '\nspans_start: ' + str(self.spans_start)
    string = string + '\nspans_end: ' + str(self.spans_end)
    string = string + '\npadding_len: ' + str(self.padding_len)
    string = string + '\ninput_tokens: ' + str(self.input_tokens)
    string = string + '\ninput: ' + str(self.input)
    string = string + '\nopers: ' + str(self.opers)
    string = string + '\ntarget_values: ' + str(self.target_values)
    string = string + '\nlast_state: ' + str(self.last_state)
    string = string + '\ncur_state: ' + str(self.cur_state)
    string = string + '\nrefer: ' + str(self.refer)
    string = string + '\ninform_aux_features: ' + str(self.inform_aux_features)
    string = string + '\nds_aux_features: ' + str(self.ds_aux_features)

    return string

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def normalize(text):
  text_lower = text.lower()
  if MULTIWOZ == True:
    # TODO: Delete MULTIWOZ
    text_norm = normalize_text(text_lower) # for mutliwoz only
  else:
    text_norm = text_lower
  text_tok = [tok for tok in map(lambda x: re.sub(" ", "", x), re.split("(\W+)", text_norm)) if len(tok) > 0]
  return text_tok


def is_included(value, target):

  included = False

  value = [item for item in map(str.strip, re.split("(\W+)", value)) if len(item) > 0]
  target = [item for item in map(str.strip, re.split("(\W+)", target)) if len(item) > 0]

  for i in range(len(value)):
    if value[i:i + len(target)] == target:
      included = True

  return included


def included_with_label_maps(value, target, label_maps):
  included = False

  variants = [target]
  if target in label_maps:
    variants += label_maps[target]

  for variant in variants:
    if value == variant or is_included(value, variant) or is_included(variant, value):
      included = True

  return included


def match_with_label_maps(value, target, label_maps={}):
    equal = False
    if value == target:
      equal = True
    elif target in label_maps:
      for variant in label_maps[target]:
        if value == variant:
          equal = True

    return equal


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


def mask_utterance(utter, inform_mem, replace_with='[UNK]'):
  utter = normalize(utter)
  for slot, informed_values in inform_mem.items():
    for informed_value in informed_values:
      informed_tok = normalize(informed_value)
      for i in range(len(utter)):
        if utter[i:i + len(informed_tok)] == informed_tok:
          utter[i:i + len(informed_tok)] = [replace_with] * len(informed_tok)
  return utter

#___preprocessing temp to be deleted________________________________________________________________________________________________________________


def normalize_time(text):

    # This code is only related to MultiWoz Dataset

    text = re.sub("(\d{1})(a\.?m\.?|p\.?m\.?)", r"\1 \2", text) # am/pm without space
    text = re.sub("(^| )(\d{1,2}) (a\.?m\.?|p\.?m\.?)", r"\1\2:00 \3", text) # am/pm short to long form
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2}) ?(\d{2})([^0-9]|$)", r"\1\2 \3:\4\5", text) # Missing separator
    text = re.sub("(^| )(\d{2})[;.,](\d{2})", r"\1\2:\3", text) # Wrong separator
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2})([;., ]|$)", r"\1\2 \3:00\4", text) # normalize simple full hour time
    text = re.sub("(^| )(\d{1}:\d{2})", r"\g<1>0\2", text) # Add missing leading 0
    # Map 12 hour times to 24 hour times
    text = re.sub("(\d{2})(:\d{2}) ?p\.?m\.?", lambda x: str(int(x.groups()[0]) + 12 if int(x.groups()[0]) < 12 else int(x.groups()[0])) + x.groups()[1], text)
    text = re.sub("(^| )24:(\d{2})", r"\g<1>00:\2", text) # Correct times that use 24 as hour
    return text


def normalize_text(text):

    # This code is only related to MultiWoz Dataset

    text = normalize_time(text)
    text = re.sub("n't", " not", text)
    text = re.sub("(^| )zero(-| )star([s.,? ]|$)", r"\g<1>0 star\3", text)
    text = re.sub("(^| )one(-| )star([s.,? ]|$)", r"\g<1>1 star\3", text)
    text = re.sub("(^| )two(-| )star([s.,? ]|$)", r"\g<1>2 star\3", text)
    text = re.sub("(^| )three(-| )star([s.,? ]|$)", r"\g<1>3 star\3", text)
    text = re.sub("(^| )four(-| )star([s.,? ]|$)", r"\g<1>4 star\3", text)
    text = re.sub("(^| )five(-| )star([s.,? ]|$)", r"\g<1>5 star\3", text)
    text = re.sub("archaelogy", "archaeology", text) # Systematic typo
    text = re.sub("guesthouse", "guest house", text) # Normalization
    text = re.sub("(^| )b ?& ?b([.,? ]|$)", r"\1bed and breakfast\2", text) # Normalization
    text = re.sub("bed & breakfast", "bed and breakfast", text) # Normalization
    text = re.sub("\t", " ", text) # Error
    text = re.sub("\n", " ", text) # Error
    return text
