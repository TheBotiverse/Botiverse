import torch
import torch.nn as nn
import math

class AttentionModule(nn.Module):
  def __init__(self,
               is_decoder=False,
               num_positional_encoding_buckets=32,
               positional_encoding_max_distance=128,
               d_model=768,
               num_heads=12,
               dropout_rate=0.1,
               has_positional_encoding=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.has_positional_encoding = has_positional_encoding
        self.num_positional_encoding_buckets = num_positional_encoding_buckets
        self.positional_encoding_max_distance = positional_encoding_max_distance
        self.d_model = d_model
        self.per_head_dim = d_model // num_heads
        self.n_heads = num_heads
        self.dropout = dropout_rate

        self.q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o = nn.Linear(self.d_model, self.d_model, bias=False)

        if self.has_positional_encoding:
            self.relative_attention_bias = nn.Embedding(self.num_positional_encoding_buckets, self.n_heads)

  def relative_positional_encoding(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
      relative_buckets = 0
      if bidirectional: # self attention in encoder
          num_buckets //= 2
          relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
          relative_position = torch.abs(relative_position)
      else: # self attention in decoder (you can't attend to what is yet to come) for efficient utilization of buckets
          relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
      
      # now all is in the positive positions realm (mapping to buckets is much straightforward)
      max_exact = num_buckets // 2
      is_small = relative_position < max_exact

      relative_position_if_large = max_exact + (
          torch.log(relative_position.float() / max_exact)
          / math.log(max_distance / max_exact)
          * (num_buckets - max_exact)
      ).to(torch.long)
      relative_position_if_large = torch.min(
          relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
      )

      relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
      return relative_buckets

  def compute_bias(self, query_length, key_length, device=None):
      if device is None:
          device = self.relative_attention_bias.weight.device
      context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
      memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
      relative_position = memory_position - context_position
      
      relative_position_bucket = self.relative_positional_encoding(
          relative_position,
          bidirectional=(not self.is_decoder),
          num_buckets=self.num_positional_encoding_buckets,
          max_distance=self.positional_encoding_max_distance,
      )
      positional_embeddings = self.relative_attention_bias(relative_position_bucket) # mapping bucketes to their corresponding embeddings
      positional_embeddings = positional_embeddings.permute([2, 0, 1]).unsqueeze(0)
      return positional_embeddings

  def forward(
      self,
      hidden_states,
      mask=None,
      key_value_states=None,
      position_bias=None):
      batch_size, seq_length = hidden_states.shape[:2]

      real_seq_length = seq_length

      key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

      def shape(states):
          return states.view(batch_size, -1, self.n_heads, self.per_head_dim).transpose(1, 2)

      def unshape(states):
          return states.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

      query_states = shape(self.q(hidden_states))

      if key_value_states is not None:
        key_states= shape(self.k(key_value_states))
        value_states = shape(self.v(key_value_states))
      else:
        key_states= shape(self.k(hidden_states))
        value_states = shape(self.v(hidden_states))

      scores = torch.matmul( # getting the weightes over the whole batch over all the heads and all at once
          query_states, key_states.transpose(3, 2)
      )

      if position_bias is None:
          if not self.has_positional_encoding:
              position_bias = torch.zeros(
                  (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
              )
          else:
              position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

          if mask is not None:
              position_bias = position_bias + mask # mask here is not 0 and 1 but -inf and 0

      position_bias_masked = position_bias

      scores += position_bias_masked
      attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
          scores
      )
      attn_weights = nn.functional.dropout(
          attn_weights, p=self.dropout, training=self.training
      )

      attn_output = unshape(torch.matmul(attn_weights, value_states))
      attn_output = self.o(attn_output)

      return attn_output, position_bias

class NewGELUActivation(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class DenseGatedActDenseModule(nn.Module):
  def __init__(self,
               d_model=768,
               d_ff=2048,
               dropout_rate=0.1):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = NewGELUActivation()

  def forward(self, hidden_states):
      hidden_gelu = self.act(self.wi_0(hidden_states))
      hidden_linear = self.wi_1(hidden_states)
      hidden_states = hidden_gelu * hidden_linear
      hidden_states = self.dropout(hidden_states)
      hidden_states = self.wo(hidden_states)
      return hidden_states

class LayerNormModule(nn.Module):
  def __init__(self, layer_size = 768, eps=1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(layer_size))
    self.epsilon = eps
  def forward(self, hidden_states):
    mean_square = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(mean_square + self.epsilon)
    return self.weight * hidden_states

class FFModule(nn.Module):
  def __init__(self,
                dropout_rate=0.1):
        super().__init__()
        self.DenseReluDense = DenseGatedActDenseModule()
        self.layer_norm = LayerNormModule()
        self.dropout = nn.Dropout(dropout_rate)

  def forward(self, hidden_states):
      forwarded_states = self.layer_norm(hidden_states)
      forwarded_states = self.DenseReluDense(forwarded_states)
      hidden_states = hidden_states + self.dropout(forwarded_states)
      return hidden_states

class SelfAttentionModule(nn.Module):
  def __init__(self,
               is_decoder,
               dropout_rate=0.1,
               has_positional_encoding=False):
    super().__init__()
    self.SelfAttention = AttentionModule(is_decoder, has_positional_encoding=has_positional_encoding)
    self.layer_norm = LayerNormModule()
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, hidden_states, attention_mask=None, position_bias=None):
      normed_hidden_states = self.layer_norm(hidden_states)
      attention_output = self.SelfAttention(
          normed_hidden_states,
          mask=attention_mask,
          position_bias=position_bias
      )
      hidden_states = hidden_states + self.dropout(attention_output[0])
      position_bias = attention_output[1]
      return hidden_states, position_bias

class EncoderBlock(nn.Module):
  def __init__(self, has_positional_encoding=False):
    super().__init__()
    self.layer = nn.ModuleList()
    self.layer.append(SelfAttentionModule(is_decoder=False, has_positional_encoding=has_positional_encoding))
    self.layer.append(FFModule())

  def forward(self, hidden_states, attention_mask=None, position_bias=None):
    self_attention_outputs = self.layer[0](hidden_states, attention_mask=attention_mask, position_bias=position_bias)
    hidden_states = self_attention_outputs[0]
    position_bias = self_attention_outputs[1]
    hidden_states = self.layer[-1](hidden_states)
    return hidden_states, position_bias

class EncoderModule(nn.Module):
  def __init__(self, embed_tokens, num_layers= 12, dropout_rate=0.1):
    super().__init__()
    self.embed_tokens = embed_tokens
    self.block = nn.ModuleList(
            [EncoderBlock(has_positional_encoding=bool(i == 0)) for i in range(num_layers)]
        )
    self.final_layer_norm = LayerNormModule()
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, input_ids=None, attention_mask=None):
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])

    inputs_embeds = self.embed_tokens(input_ids)
    
    if attention_mask is None:
      attention_mask = torch.ones(input_shape)
    final_attention_mask = attention_mask.to(dtype=torch.float32)
    final_attention_mask = (1.0 - final_attention_mask) * torch.finfo(torch.float32).min

    hidden_states = self.dropout(inputs_embeds)
    position_bias = None
    for block in self.block:
      layer_outputs = block(
        hidden_states,
        attention_mask = final_attention_mask,
        position_bias = position_bias
        )
      hidden_states = layer_outputs[0]
      position_bias = layer_outputs[1]

    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)
    return hidden_states

class CrossAttentionModule(nn.Module):
  def __init__(self, dropout_rate=0.1):
    super().__init__()
    self.EncDecAttention = AttentionModule(is_decoder=True, has_positional_encoding=False)
    self.layer_norm = LayerNormModule()
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, hidden_states, key_value_states, encoder_attention_mask=None):
      normed_hidden_states = self.layer_norm(hidden_states)
      attention_output = self.EncDecAttention(
          normed_hidden_states,
          mask=encoder_attention_mask,
          key_value_states=key_value_states
      )
      hidden_states = hidden_states + self.dropout(attention_output[0])
      position_bias = attention_output[1]
      return hidden_states, position_bias

class DecoderBlock(nn.Module):
  def __init__(self, has_positional_encoding=False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(SelfAttentionModule(is_decoder=True, has_positional_encoding=has_positional_encoding))
        self.layer.append(CrossAttentionModule())
        self.layer.append(FFModule())

  def forward(
      self,
      hidden_states,
      attention_mask=None,
      position_bias=None,
      encoder_hidden_states=None,
      encoder_attention_mask=None
  ):
      self_attention_outputs = self.layer[0](
          hidden_states,
          attention_mask=attention_mask,
          position_bias=position_bias
      )
      hidden_states = self_attention_outputs[0]
      position_bias = self_attention_outputs[1]

      cross_attention_outputs = self.layer[1](
          hidden_states,
          key_value_states=encoder_hidden_states,
          encoder_attention_mask=encoder_attention_mask
      )
      hidden_states = cross_attention_outputs[0]

      hidden_states = self.layer[-1](hidden_states)
      return hidden_states, position_bias

class DecoderModule(nn.Module):
  def __init__(self, embed_tokens, num_layers= 12, dropout_rate=0.1):
    super().__init__()

    self.embed_tokens = embed_tokens

    self.block = nn.ModuleList(
        [DecoderBlock(has_positional_encoding=bool(i == 0)) for i in range(num_layers)]
    )
    self.final_layer_norm = LayerNormModule()
    self.dropout = nn.Dropout(dropout_rate)

  def forward(
              self,
              input_ids=None,
              attention_mask=None,
              encoder_hidden_states=None,
              encoder_attention_mask=None
  ):
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    inputs_embeds = self.embed_tokens(input_ids)

    if attention_mask is None:
      attention_mask = torch.ones(input_shape)
    seq_length = input_shape[1]
    lower_triangular_mask = torch.tril(torch.ones((seq_length, seq_length))).view(1, 1, seq_length, seq_length)
    final_attention_mask = lower_triangular_mask * attention_mask
    final_attention_mask = (1.0 - final_attention_mask) * torch.finfo(torch.float32).min
    
    if encoder_attention_mask is None:
      encoder_attention_mask = torch.ones((encoder_hidden_states.size(0), encoder_hidden_states.size(1)))
    encoder_attention_mask = encoder_attention_mask.to(dtype=torch.float32)
    encoder_attention_mask = (1.0 - encoder_attention_mask) * torch.finfo(torch.float32).min

    hidden_states = self.dropout(inputs_embeds)
    position_bias = None
    for block in self.block:
      layer_outputs = block(
          hidden_states,
          attention_mask=final_attention_mask,
          position_bias=position_bias,
          encoder_hidden_states=encoder_hidden_states,
          encoder_attention_mask=encoder_attention_mask
      )

      hidden_states = layer_outputs[0]
      position_bias = layer_outputs[1]
    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)
    return hidden_states

class T5Model(nn.Module):
    def __init__(self, vocab_size=32128, d_model=768):
        super().__init__()
        self.shared = nn.Embedding(vocab_size, d_model)
        self.encoder = EncoderModule(self.shared)
        self.decoder = DecoderModule(self.shared)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None):
        # encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # decoder
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
        )
        # lm_head
        lm_logits = self.lm_head(decoder_outputs)
        return lm_logits
    
    def generate(
      self,
      input_ids=None,
      attention_mask=None,
      max_length=10):
      # put model in evaluation mode (stop dropout and backwrad propagation calculation)
      self.eval()
      encoder_outputs = self.encoder(
          input_ids=input_ids,
          attention_mask=attention_mask
      )
      # autoregressive generation
      generated_ids = [torch.tensor([0])]# start_token
      cur_ids = torch.zeros((input_ids.size(0), 1)).long()# start_token
      # genrate until max_length or eos_token
      for _ in range(max_length):
        decoder_outputs = self.decoder(
            input_ids=cur_ids,
            encoder_hidden_states=encoder_outputs
        )
        lm_logits = self.lm_head(decoder_outputs)
        next_token_logits = lm_logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        generated_ids.append(next_token_id)
        cur_ids = torch.cat([cur_ids, next_token_id.unsqueeze(-1)], dim=-1)
        if next_token_id == 1: # eos_token
          break
      return torch.cat(generated_ids, dim=-1)