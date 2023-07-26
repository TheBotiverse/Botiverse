#!/usr/bin/env python
# coding: utf-8

# # <font color="cyan">FastSpeech 1.0 </font> Implementation
# 
# In this notebook, we shall demonstrate implementing FastSpeech 1.0 from scratch for inference purposes. 

# In[1]:


'''
FastSpeech 1.0 interface and implementation from scratch in PyTorch for inference.
'''
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 
# ![Image](https://i.imgur.com/ZDR7wqr.png)
# 

# #### Clearly, we have three main components in the model:
# 
# <div align='center'>
# <table>
#   <tr>
#     <th colspan="1"><font color="yellow">Component</font></th>
#     <th colspan="3"><font color="yellow">Encoder</font></th>
#     <th colspan="2"><font color="yellow">Length Regulator</font></th>
#     <th colspan="3"><font color="yellow">Decoder</font></th>
# 
#   </tr>
#   <tr>
#     <th colspan="1"><font color="white">Subcomponenets</font></th>
#     <th colspan="1"><font color="white">Phoneme Embedding</font></th>
#     <th colspan="1"><font color="white">Positional Encoding</font></th>
#     <th colspan="1"><font color="white">FFT Block</font></th>
#     <th colspan="1"><font color="white">Duration Predictor</font></th>
#     <th colspan="1"><font color="white">LR Logic</font></th>
#     <th colspan="1"><font color="white">Positional Encoding</font></th>
#     <th colspan="1"><font color="white">FFT Block</font></th>
#     <th colspan="1"><font color="white">Linear Layer</font></th>
#   </tr>
# 
#   <tr>
#   <th colspan="1">Takes</th>
#   <td colspan="3"> <font color='cyan'>[batch_size, seq_len]</font></td>
#   <td colspan="2"> <font color='cyan'>[batch_size, seq_len, emb_dim]</font> </td>
#   <td colspan="3"> <font color='cyan'>[batch_size, new_seq_len, emb_dim]</font> </td>
#   </tr>
# 
#   <tr>
#   <th colspan="1">Yields</th>
#   <td colspan="3"> <font color='cyan'>[batch_size, seq_len, emb_dim]  </td>
#   <td colspan="2"> <font color='cyan'>[batch_size, new_seq_len, emb_dim]</font> </td>
#   <td colspan="3"> <font color='cyan'>[batch_size, new_seq_len, mel_num]</font> </td>
#   </tr>
# 
#   <tr>
#   <th colspan="1">Purpose</th>
#   <td colspan="3">Learn a representation for phonemes. The three layers within guarantee that underlying words, how they are ordered and other words all take part in the representation. </td>
#   <td colspan="2">Predict the duration of each phoneme and repeat accordingly</td>
#   <td colspan="3">Given time-aligned phoneme representations learn to transform them into a mel spectogram</td>
#   </tr>
# </table>
# </div>
# 
# Once we have the spectogram, all that's needed it a vocoder to transform to audio by performing an approximate inverse mel-transform such as Griffin-Lim algorithm or using a pretrained model that does the task more accurately such as WaveGlow; hence, we will go with the latter but we won't implement WaveGlow from scratch.
# 

# #### We will start by implementing 
# 
# <div align='left'>
# <table>
#   <tr>
#     <th colspan="2"><font color="cyan">FFT Block</th>
#   </tr>
#   <tr>
#     <th colspan="1">Multi-head Attention</th>
#     <th colspan="1">Conv1DNet</th>
#   </tr>
# </table>
# </div>
# 
# #### Then we have that
# 
# <div align='left'>
# <table>
#   <tr>
#     <th colspan="3"><font color="deepskyblue">Encoder</font></th>
#     <th colspan="3"><font color="deepskyblue">Decoder</font></th>
#   </tr>
#   <tr>
#     <th colspan="1">Phoneme Emb.</th>
#     <th colspan="1">Positional Enc.</th>
#     <th colspan="1">FFT Block</th>
#     <th colspan="1">Positional Encoding</th>
#     <th colspan="1">FFT Block</th>
#     <th colspan="1">Linear Layer</th>
#   </tr>
# </table>
# </div>
# 
# #### So we follow with
# 
# <div align='left'>
# <table>
#   <tr>
#     <th colspan="2"><font color="deepskyblue">Length Regulator</font></th>
#   </tr>
#   <tr>
#     <th colspan="1">LR Logic</th>
#     <th colspan="1">Duration Predictor</th>
#   </tr>
# 
# </table>
# </div>
# 
# #### And we finally have that
# 
# <div align='left'>
# <table>
#   <tr>
#     <th colspan="3"><font color="deepskyblue">Model</font></th>
#   </tr>
#   <tr>
#     <th colspan="1">Encoder</th>
#     <th colspan="1">Length Regulator</th>
#     <th colspan="1">Decoder</th>
#   </tr>
# </table>
# </div>

# ### <font color="cyan">1. Feedforward-Transformer Block </font>

# ![img](https://i.imgur.com/Kb88BwT.png)

# #### <font color="white"> Multi-head Self-Attention </font>

# <div align='center'>
# 
# $MultiHead(Q, K, V) = \text{Concat}(head_1, \ldots, head_h) \cdot W^O$
# 
# where
# 
#   $head = \text{Attention}(qW_q, kW_k, vW_v) =\text{Attention}(Q, K, V)$
# 
#   such that
# 
#   $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \cdot V$
# 
#   
#   </div>

# In[2]:


import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
    '''
    Multi-Head Attention module with a residual connection and layer normalization. Used as self-attention in the FFT block of the encoder and decoder.

    :param num_head: Number of attention heads.
    :type num_head: int
    :param emb_dim: Input encoder/decoder embeddings dimensions.
    :type emb_dim: int
    :param h_dim: Hidden dimension (output dimension of the linear layers Wq, Wk, Wv).
    :type h_dim: int
    :param dropout: Dropout probability. Default is 0.1.
    :type dropout: float, optional
    '''
    def __init__(self, num_head, emb_dim, h_dim, dropout=0.1):
        super().__init__()

        self.num_head = num_head
        self.h_dim = h_dim                            # dimensionality of the final output
        self.head_dim = h_dim // num_head             # dimensionality of each head

        self.Wq = nn.Linear(emb_dim, h_dim)         # Equivalent to using head_dim, num_head times
        self.Wk = nn.Linear(emb_dim, h_dim)
        self.Wv = nn.Linear(emb_dim, h_dim)

        self.softmax = nn.Softmax(dim=2)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(h_dim, emb_dim)
        
       
    def forward(self, q, k, v, mask):
        '''
        Pass given query, key, and value through the Multi-Head Attention module.

        :param q: Query tensor of shape [batch_size, seq_len, emb_dim].
        :type q: torch.Tensor
        :param k: Key tensor of shape [batch_size, seq_len, emb_dim].
        :type k: torch.Tensor
        :param v: Value tensor of shape [batch_size, seq_len, emb_dim].
        :type v: torch.Tensor
        :param mask: Mask to apply to the attention so that padding tokens do not attend and are not attended to.
        :type mask: torch.Tensor
        
        :returns: Output tensor of shape [batch_size, seq_len, emb_dim].
        :rtype: torch.Tensor
        '''
        residual = q
        batch_size, num_head, seq_len, head_dim = q.size(0), self.num_head, q.size(1), self.head_dim
        Q, K, V = self.Wq(q), self.Wk(k), self.Wv(v)              # [batch_size, seq_len, emb_dim -> h_dim]

        review = lambda X: X.view(batch_size, seq_len, num_head, head_dim).transpose(1, 2).contiguous()
        Q, K, V = review(Q), review(K), review(V)                  # [batch_size, num_head, seq_len, head_dim]
        
        reshape = lambda X: X.view(batch_size * num_head, seq_len, head_dim)
        Q, K, V = reshape(Q), reshape(K), reshape(V)               # [batch_size * num_head, seq_len, head_dim]

        mask = mask.repeat(num_head, 1, 1)                         # [batch_size * num_head, seq_len, seq_len]
        
        a = torch.bmm(Q, K.transpose(1, 2))/self.scale             # [batch_size * num_head, seq_len, seq_len]
        
        a = a.masked_fill(mask, -np.inf)

        a = self.softmax(a)
        
        a = self.dropout(a)
        
        output = torch.bmm(a, V)                                  # [batch_size * num_head, seq_len, head_dim]   
        
        output = output.view(batch_size, num_head, seq_len, head_dim).transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, num_head * head_dim)
        # [batch_size, seq_len, num_head * head_dim]

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


# #### <font color="white"> Conv1D Network </font>

# $Embeddings ⇒ Conv1D ⇒ ReuLU ⇒ Conv1D ⇒  Dropout ⇒ AddNorm$

# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DNet(nn.Module):
    '''
    1D convolutional network with residual connection and layer normalization. Used in the FFT block of the encoder and decoder.

    :param inp_dim: Input dimension.
    :type inp_dim: int
    :param inner_dim: Inner dimension of the convolutional layers (output dimension of the first convolutional layer).
    :type inner_dim: int
    :param dropout: Dropout probability. Default is 0.1.
    :type dropout: float, optional
    '''
    def __init__(self, inp_dim, inner_dim, dropout=0.1):
        super().__init__()
        # Both convolutions have SAME padding so they only change the number of 1D channels
        self.conv1 = nn.Conv1d(inp_dim, inner_dim, kernel_size=9, padding=4) # from n to n-9+2*4+1 = n
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(inner_dim, inp_dim, kernel_size=1, padding=0) # from n to n-1+2*0+1 = n
        self.layer_norm = nn.LayerNorm(inp_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):   
        '''
        Pass given input through the 1D convolutional network. The input comes from the Multi-Head Attention module.

        :param x: Input tensor of shape [batch_size, seq_len, d_in].
        :type x: torch.Tensor
        :returns: Output tensor of shape [batch_size, seq_len, d_in].
        :rtype: torch.Tensor
        '''
        residual = x                                     # [batch_size, seq_len, d_in]
        x = x.transpose(1, 2)                            # [batch_size, d_in, seq_len]
        x = F.relu(self.conv1(x))                        # [batch_size, d_out, seq_len]
        x = self.conv2(x)                                # [batch_size, din, seq_len]
        x = x.transpose(1, 2)                            # [batch_size, seq_len, din]
        
        output = self.dropout(x)
        output = self.layer_norm(output + residual)

        return output


# #### <font color="white"> FFT Block </font>
# 
# Here we just put the two pieces together

# In[4]:


import torch.nn as nn

class FFTBlock(nn.Module):
    '''
    FFT block used in the encoder and decoder. It consists of a Multi-Head Attention module and a 1D convolutional network.

    :param emb_dim: Input encoder/decoder embeddings dimensions.
    :type emb_dim: int
    :param num_head: Number of heads for the Multi-Head Attention module.
    :type num_head: int
    :param h_dim: Hidden dimension (output dimension of the linear layers Wq, Wk, Wv).
    :type h_dim: int
    :param inner_dim: Inner dimension of the convolutional layers (output dimension of the first convolutional layer).
    :type inner_dim: int
    :param dropout: Dropout probability. Default is 0.1.
    :type dropout: float, optional
    '''
    def __init__(self, emb_dim, num_head, h_dim, inner_dim, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.attn_out = MultiHeadAttention(num_head, emb_dim, h_dim, dropout=dropout)
        self.conv_out = Conv1DNet(emb_dim, inner_dim, dropout=dropout)

    def forward(self, input, non_pad_mask, attn_mask):
        '''
        Pass given encoder/decoder embeddings through the FFT block. The input comes from the previous FFT block or the input embeddings.

        :param input: Input tensor of shape [batch_size, seq_len, emb_dim].
        :type input: torch.Tensor
        :param non_pad_mask: Mask to nullify outputs due to padding tokens.
        :type non_pad_mask: torch.Tensor
        :param attn_mask: Mask to apply to the attention so that future tokens do not attend and are not attended to.
        :type attn_mask: torch.Tensor
        :returns: Output tensor of shape [batch_size, seq_len, emb_dim] and attention weights tensor of shape [batch_size * num_head, seq_len, seq_len].
        :rtype: tuple(torch.Tensor, torch.Tensor)
        '''
        non_pad_mask = non_pad_mask.float()
        # From [batch_size, seq_len, emb_dim] to [batch_size, seq_len, h_dim->emb_dim]
        output = self.attn_out(input, input, input, attn_mask) * non_pad_mask

        # From [batch_size, seq_len, emb_dim] to [batch_size, seq_len, emb_dim]
        output = self.conv_out(output) * non_pad_mask

        return output


# ### <font color="cyan">2. Encoder</font> & <font color="cyan">Decoder</font> 

# ![](https://i.imgur.com/HKBCPO7.png)

# It's obvious that they share most of the structure except for the first/last layer. Hence, we will implement both in one module. But before we do we need positional encoding.

# #### <font color="white"> Positional Encoding </font>

# 
# $$PE_{(pos, 2i)} = \sin(\frac{{pos}}{{10000^{2i/d_{\text{inp}}}}})$$
# 
# 
# $$PE_{(pos, 2i+1)} = \cos(\frac{{pos}}{{10000^{2i/d_{\text{inp}}}}})$$
# 

# The purpose of this is to add some notion of order to the input sequence without distorting it by adding a positional depending signal

# In[5]:


import numpy as np
import torch

class SinusoidEncodingTable:
    '''
    Sinusoid encoding table used in the encoder and decoder. It is used to add positional information to the input embeddings.

    :param max_seq_len: Maximum sequence length.
    :type max_seq_len: int
    :param inp_dim: Input encoder/decoder embeddings dimensions.
    :type inp_dim: int
    :param padding_idx: Index of the padding token. Default is None.
    :type padding_idx: int, optional
    '''
    def __init__(self, max_seq_len, inp_dim, padding_idx=None):
        self.max_seq_len = max_seq_len + 1
        self.inp_dim = inp_dim
        self.padding_idx = padding_idx
        self.sinusoid_table = self.build_table()

    def build_table(self):
        '''
        Build the sinusoid encoding table.

        :returns: Sinusoid encoding table of shape [max_seq_len, inp_dim] which is indexed by the position of the input embeddings and the index of the input embeddings value.
        :rtype: torch.Tensor
        '''
        # Compute the denominator
        inds = np.arange(self.inp_dim) // 2
        div_term = np.power(10000, 2 * inds / self.inp_dim)

        # Compute the numerator
        positions = np.arange(self.max_seq_len)

        # Compute the table of shape [max_seq_len, inp_dim]
        sinusoid_table = np.outer(positions, 1/div_term)

        # Apply sin to even indices in the array; 2i and cos to odd indices; 2i+1
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) 
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  

        if self.padding_idx is not None:
            sinusoid_table[self.padding_idx] = 0.

        return torch.Tensor(sinusoid_table)


# #### <font color="white"> Dencoder </font>
# 
# This is both the encoder and decoder in one module

# In[6]:


import torch.nn as nn

class Dencoder(nn.Module):
    '''
    A single module that implements the encoder and decoder of the FastSpeech 1.0 model. 

    :param mode: 'Encoder' or 'Decoder'.
    :type mode: str
    :param vocab_dim: Vocabulary dimension. None if mode is 'Decoder'.
    :type vocab_dim: int, optional
    :param max_seq_len: Maximum sequence length as needed by the sinusoid encoding table.
    :type max_seq_len: int
    :param emb_dim: Encoder/decoder embeddings dimensions.
    :type emb_dim: int
    :param num_layer: Number of FFT blocks.
    :type num_layer: int
    :param num_head: Number of heads for the Multi-Head Attention module.
    :type num_head: int
    :param h_dim: Hidden dimension (output dimension of the linear layers Wq, Wk, Wv).
    :type h_dim: int
    :param d_inner: Inner dimension of the convolutional layers (output dimension of the first convolutional layer).
    :type d_inner: int
    :param mel_num: Number of mel spectrogram bins (to map the final decoder embeddings). None if mode is 'Encoder'.
    :type mel_num: int, optional
    :param dropout: Dropout probability.
    :type dropout: float
    '''
    def __init__(self, mode, vocab_dim, max_seq_len, emb_dim, num_layer, num_head, h_dim, d_inner, mel_num, dropout):
        super(Dencoder, self).__init__()
        self.mode = mode

        self.embedding = nn.Embedding(vocab_dim, emb_dim, padding_idx=0) if self.mode == 'Encoder' else lambda x: x

        table = SinusoidEncodingTable(max_seq_len, emb_dim, padding_idx=0).sinusoid_table
        self.position_encoding = nn.Embedding.from_pretrained(table, freeze=True)

        self.layer_stack = nn.ModuleList([FFTBlock(emb_dim, num_head, h_dim, d_inner, dropout=dropout) for _ in range(num_layer)])

        self.linear = nn.Linear(emb_dim, mel_num) if self.mode == 'Decoder' else lambda x: x
            
    def forward(self, inp_seq, inp_seq_pos):                                            
        '''
        Pass given input sequence through the encoder/decoder.

        :param inp_seq: Input sequence of shape [batch_size, seq_len] for Encoder or [batch_size, seq_len, emb_dim] for Decoder.
        :type inp_seq: torch.Tensor
        :param inp_seq_pos: Input sequence positions of shape [batch_size, seq_len].
        :type inp_seq_pos: torch.Tensor
        :returns: Output tensor of shape [batch_size, seq_len, emb_dim] for Decoder or [batch_size, seq_len, mel_num] for Encoder.
        :rtype: torch.Tensor
        '''
        # Prepare non-pad and attention masks
        inp_attn = inp_seq if self.mode == 'Encoder' else inp_seq_pos              # [batch_size, seq_len]
        
        non_pad_mask = (inp_attn.unsqueeze(2) != 0)                                # [batch_size, seq_len, 1]
        attn_mask = (inp_attn.unsqueeze(1) == 0).repeat(1, inp_attn.size(1), 1)    # [batch_size, seq_len, seq_len]

        # Forward
        output = self.embedding(inp_seq) + self.position_encoding(inp_seq_pos)       # [batch_size, seq_len, emb_dim]
        for layer in self.layer_stack:
            output = layer(output, non_pad_mask=non_pad_mask, attn_mask=attn_mask)
        
        output = self.linear(output)
        
        return output


# ### <font color="cyan">3. Length Regulator </font>

# ![](https://i.imgur.com/xUvPwh6.png)

# #### <font color="white"> Duration Predictor </font>
# 

# $PhonemeEmb ⇒ Conv1D ⇒ ReuLU ⇒ LayerNorm ⇒  Dropout ⇒ Conv1D ⇒ ReuLU ⇒ LayerNorm ⇒  Dropout ⇒ Relu ⇒ Linear$

# In[7]:


import torch.nn as nn

class DurationPredictor(nn.Module):
    '''
    Duration predictor module. It predicts the duration of each phoneme in the input sequence of encoder phoneme embeddings.

    :param inp_dim: Input dimension.
    :type inp_dim: int
    :param inner_dim: Inner dimension of the convolutional layers (output dimension of the first convolutional layer).
    :type inner_dim: int
    :param kernel_size: Kernel size of the convolutional layers.
    :type kernel_size: int
    :param padding_size: Padding size of the convolutional layers.
    :type padding_size: int
    :param dropout: Dropout probability. Default is 0.1.
    :type dropout: float, optional
    '''
    def __init__(self, inp_dim, inner_dim, kernel_size, padding_size, dropout=0.1):
        super(DurationPredictor, self).__init__()

        self.conv1d_1 = nn.Conv1d(inp_dim, inner_dim, kernel_size, padding=padding_size)
        self.layer_norm_1 = nn.LayerNorm(inner_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv1d_2 = nn.Conv1d(inner_dim, inner_dim, kernel_size, padding=padding_size)
        self.layer_norm_2 = nn.LayerNorm(inner_dim)
        self.linear_layer = nn.Linear(inner_dim, 1)     
        # Predicts a scalar mel_duration per phoneme

    def forward(self, encoder_output):
        '''
        Pass given encoder output through the duration predictor module. The input comes from the encoder. 

        :param encoder_output: Encoder output of shape [batch_size, seq_len, emb_dim].
        :type encoder_output: torch.Tensor
        :returns: Predicted duration of each phoneme in the input sequence of encoder phoneme embeddings of shape [batch_size, seq_len].
        :rtype: torch.Tensor
        '''
        x = encoder_output.contiguous().transpose(1, 2)
        x = self.conv1d_1(x)
        x = x.contiguous().transpose(1, 2)
        x = self.relu(self.layer_norm_1(x))
        x = self.dropout(x)
        x = x.contiguous().transpose(1, 2)
        x = self.conv1d_2(x)
        x = x.contiguous().transpose(1, 2)
        x = self.relu(self.layer_norm_2(x))
        x = self.dropout(x)
        
        out = self.relu(self.linear_layer(x))

        out = out.squeeze()
        out = out.unsqueeze(0)      # Leading dimension should not be removed in inference.

        return out


# #### <font color="white">Length Regulation Logic </font>
# 

# Given $H=[h_1, h_2, ..., h_n]$ as phone embeddings from the encoder predict the duration of each phoneme $d=[d_1, d_2, ..., d_n]$ and repeat accordingly.

# In[8]:


import torch
import torch.nn as nn

class LengthRegulator(nn.Module):
    '''
    Length regulator module. It repeats the encoder outputs according to the predicted duration of each phoneme in the input sequence of encoder phoneme embeddings.

    :param inp_dim: Input dimension.
    :type inp_dim: int
    :param inner_dim: Inner dimension of the convolutional layers (output dimension of the first convolutional layer).
    :type inner_dim: int
    :param kernel_size: Kernel size of the convolutional layers.
    :type kernel_size: int
    :param padding_size: Padding size of the convolutional layers.
    :type padding_size: int
    :param dropout: Dropout probability. Default is 0.1.
    :type dropout: float, optional
    '''
    def __init__(self, inp_dim, inner_dim, kernel_size, padding_size, dropout=0.1):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = DurationPredictor(inp_dim, inner_dim, kernel_size, padding_size, dropout)

    def forward(self, enc_output):
        '''
        Pass given encoder output through the length regulator module. The input comes from the encoder.

        :param enc_output: Encoder output of shape [batch_size, seq_len, emb_dim].
        :type enc_output: torch.Tensor
        :returns: Modified encoder output of shape [batch_size, new_seq_len, emb_dim] and new positional encoding for the modified encoder output of shape [batch_size, new_seq_len].
        :rtype: tuple(torch.Tensor, torch.Tensor)
        '''
        tiny_slowdown = 0.5                                  # to prevent rounding from dropping phonemes 
        duration_predictions = (self.duration_predictor(enc_output) + tiny_slowdown).int()
        # [batch_size, seq_len, enc_dim] to [batch_size, new_seq_len] (scalar mel_duration per phoneme)
        
        # Repeat each phoneme in the encoder output according to its predicted duration
        new_seq_lens_per_batch = torch.sum(duration_predictions, -1)
        new_seq_len = torch.max(new_seq_lens_per_batch).item()
        
        batch_size, seq_len, enc_dim = enc_output.size()
        mod_enc_output = torch.zeros((batch_size, new_seq_len, enc_dim), device=enc_output.device)
        for sequence in range(batch_size):
            count = 0
            for phoneme in range(seq_len):
                hidden = enc_output[sequence][phoneme]
                reps = duration_predictions[sequence][phoneme]
                for _ in range(reps):
                    mod_enc_output[sequence][count] = hidden
                    count += 1
        
        # Form a new positional encoding for the modified encoder output
        output_pos = torch.LongTensor([i + 1 for i in range(mod_enc_output.size(1))]).unsqueeze(0).to(enc_output.device)
        # [batch_size, seq_len, enc_dim] -> [batch_size, new_seq_len, enc_dim] and [batch_size, new_seq_len]
        return mod_enc_output, output_pos


# ### <font color="cyan">4. Model </font>

# ![](https://i.imgur.com/azh7cwn.png)

# #### Hyperparameters

# In[9]:



# Encoder
VOCAB_SIZE = 300
ENC_EMB_DIM = 256
ENC_NUM_LAYER = 4
ENC_NUM_HEAD = 2
ENC_1D_FILTER_SIZE = 1024
DROPOUT_PROB = 0.1

# Length Regulator
INNER_DIM = 256
DP_KERNEL_SIZE = 3
DROPOUT_PROB = 0.1
DP_PADDING = 1

# Decoder
MAX_SEQ_LEN = 3000
DEC_EMB_DIM = 256
DEC_NUM_LAYER = 4
DEC_NUM_HEAD = 2
DEC_1D_FILTER_SIZE = 1024
MEL_NUM = 80


# #### <font color="white">FastSpeech</font>
# 

# In[10]:


import torch.nn as nn

class FastSpeech(nn.Module):
    '''
    FastSpeech 1.0 model. It consists of an encoder, a duration predictor, a length regulator, and a decoder.
    '''
    def __init__(self):
        '''
        Initialize the structure of the FastSpeech 1.0 model.
        '''
        super(FastSpeech, self).__init__()

        self.encoder = Dencoder(mode='Encoder', vocab_dim=VOCAB_SIZE, max_seq_len=VOCAB_SIZE, emb_dim=ENC_EMB_DIM,
                                num_layer=ENC_NUM_LAYER, num_head=ENC_NUM_HEAD, h_dim=ENC_EMB_DIM,
                                d_inner=ENC_1D_FILTER_SIZE, mel_num=None, dropout=DROPOUT_PROB)
        
        self.length_regulator = LengthRegulator(inp_dim=ENC_EMB_DIM, inner_dim=INNER_DIM, kernel_size=DP_KERNEL_SIZE, 
                                                padding_size=DP_PADDING, dropout=DROPOUT_PROB)
        
        self.decoder = Dencoder(mode='Decoder', vocab_dim=None, max_seq_len=MAX_SEQ_LEN, emb_dim=DEC_EMB_DIM, 
                                num_layer=DEC_NUM_LAYER, num_head=DEC_NUM_HEAD, h_dim=DEC_EMB_DIM,
                                d_inner=DEC_1D_FILTER_SIZE, mel_num=MEL_NUM, dropout=DROPOUT_PROB)


    def forward(self, text_seq, src_pos):
        '''
        Pass given input sequence through the FastSpeech 1.0 model.

        :param text_seq: Input sequence of shape [batch_size, seq_len] and assigns a unique id to each character in it.
        :type text_seq: torch.Tensor
        :param src_pos: Input sequence positions (indices) of shape [batch_size, seq_len].
        :type src_pos: torch.Tensor
        :returns: Predicted mel spectrogram of shape [batch_size, mel_num, new_seq_len].
        :rtype: torch.Tensor
        '''
        enc_output = self.encoder(text_seq, src_pos)
        
        length_regulator_output, decoder_pos = self.length_regulator(enc_output)
        
        spectogram = self.decoder(length_regulator_output, decoder_pos)

        return spectogram


# ### <font color="cyan">5. Waveglow

# ![](https://i.imgur.com/Xl9HEgm.png)

#  </font>

# ## <font color="yellow"> Inference </font>

# In[11]:


import warnings; warnings.filterwarnings("ignore")
from scipy.io.wavfile import write
try:
    from playsound import playsound
    import waveglow
except:
    pass
import os
import gdown


# In[12]:


class TTS():
    '''
    Text-to-Speech (TTS) class that implements the FastSpeech 1.0 model and the WaveGlow model for speech synthesis.

    :param force_download_wg: Whether to force download the WaveGlow weights if they already seem to exist. Default is False.
    :type force_download_wg: bool, optional
    :param force_download_fs: Whether to force download the FastSpeech 1.0 weights if they already seem to exist. Default is False.
    :type force_download_fs: bool, optional
    '''
    def __init__(self, force_download_wg=False, force_download_fs=False):
        # see if there is a WaveGlow folder in the current directory
        # print file and folder names in the current directory
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(curr_dir + '/weights.pth') or force_download_fs:
            print("Weights not found. Downloading FastSpeech 1.0 weights...")
            # if not, download the WaveGlow folder
            f_id = '1G630THkg1CaAZiYAK-rcg7hLNIxu2oO5' 
            gdown.download(f'https://drive.google.com/uc?export=download&confirm=pbef&id={f_id}', curr_dir + '/weights.pth', quiet=False)
            print("Done.")              
        self.symbols = list('_-!\'(),.:;? ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.model = nn.DataParallel(FastSpeech()).to(device)
        # get directory of the current file
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model.load_state_dict(torch.load(dir_path + "/weights.pth", map_location=device))
        self.model.eval()
        self.WaveGlow = waveglow.load.load_model(download=force_download_wg)
    
    def speak(self, text, play=False, save=False):
        '''
        Pass given text through the FastSpeech 1.0 model and the WaveGlow model to generate speech.

        :param text: Text to be spoken with at most 300 characters.
        :type text: str
        :param play: Whether to play the generated speech. Default is False.
        :type play: bool, optional
        :param save: Whether to save the generated speech as an audio file. Default is False.
        :type save: bool, optional
        :returns: The generated speech as an audio signal.
        :rtype: numpy.ndarray
        '''
        ascii_text = text.lower()
        sequence = np.array([self.symbol_to_id[s] for s in ascii_text if s in self.symbol_to_id ])
        sequence_inds = np.array([i+1 for w, i in enumerate(sequence)])
        
        sequence, sequence_inds = sequence[np.newaxis, :], sequence_inds[np.newaxis, :]
        
        sequence = torch.from_numpy(sequence).long() if device != torch.device('cuda') else torch.from_numpy(sequence).cuda().long()
        sequence_inds = torch.from_numpy(sequence_inds).long() if device != torch.device('cuda') else torch.from_numpy(sequence_inds).cuda().long()

        with torch.no_grad():
            mel = self.model.module.forward(sequence, sequence_inds)
        mel = mel.contiguous().transpose(1, 2)   
        
        audio = waveglow.inference.get_wav(mel, self.WaveGlow).cpu().numpy()
        if save:
            # get the directory of the current file
            dir_path = os.path.dirname(os.path.realpath(__file__))
            # save the audio file in the current directory
            write(dir_path + "/sample.wav", 22050, audio.astype('int16'))
            
        if play:
            write('sample.wav', 22050, audio.astype('int16'))
            playsound("sample.wav")
            os.remove("sample.wav")
        return audio


# In[13]:


# if running from notebook
if __name__ == '__main__':
    get_ipython().system('jupyter nbconvert --to script FastSpeech.ipynb')

