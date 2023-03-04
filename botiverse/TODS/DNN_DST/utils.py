import re

class RawDataInstance():
  def __init__(self,
               dial_idx,
               turn_idx,
               turn_domain,
               user_utter,
               sys_utter,
               history,
               last_state,
               cur_state):       
    self.dial_idx = dial_idx
    self.turn_idx = turn_idx
    self.turn_domain = turn_domain
    self.user_utter = user_utter
    self.sys_utter = sys_utter
    self.history = history
    self.last_state = last_state
    self.cur_state = cur_state

  def __str__(self):
    string = ''
    string = string + '\ndial_idx: ' + str(self.dial_idx)
    string = string + '\nturn_idx: ' + str(self.turn_idx)
    string = string + '\nturn_domain: ' + str(self.turn_domain)
    string = string + '\nuser_utter: ' + str(self.user_utter)
    string = string + '\nsys_utter: ' + str(self.sys_utter)
    string = string + '\nhistory: ' + str(self.history)
    string = string + '\nlast_state: ' + str(self.last_state)
    string = string + '\ncur_state: ' + str(self.cur_state)
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
               target_values):
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
    return string

def normalize_time(text):

    # This code is only related to MultiWoz Dataset
    # This code is from https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public

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
    # This code is from https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public

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