import torch
import torch.nn as nn
from typing import Callable, Optional
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import math

class PositionalEmbedding(nn.Module): # venilla
    def __init__(self, d_model, max_len=200):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    

def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)

## custom encoder

class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask:Optional[Tensor]=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v = None,):
        super(MultiHeadAttention, self).__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]
        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention(d_k=self.d_k)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model,d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)
    
class EncoderLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model,d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn
    
class Encoder(nn.Module):
    def __init__(self,in_channel,sq_len, dropout, n_layers,d_model,n_heads,d_ff):
        super(Encoder, self).__init__()
        self.val_emb = nn.Linear(in_channel, d_model)
        self.pos_emb = positional_encoding('zeros', learn_pe=True, q_len=sq_len, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model,n_heads,d_ff) for _ in range(n_layers)])

    def forward(self, enc_inputs):

        enc_outputs = self.dropout(self.val_emb(enc_inputs) + self.pos_emb)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs,enc_self_attn_mask=None)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
    
class DecoderLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_ff):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_heads)
        self.dec_enc_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model,d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        if enc_outputs is not None:
            dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, attn_mask=None)
            dec_outputs = self.pos_ffn(dec_outputs)
            return dec_outputs, dec_self_attn, dec_enc_attn
        else:
            dec_outputs = self.pos_ffn(dec_outputs)
            return dec_outputs, dec_self_attn, None
# decoder
def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask



class EmbedDecoder(nn.Module):
    def __init__(self, c_out ,d_model,d_layers,dropout,n_heads,d_ff):
        super(EmbedDecoder, self).__init__()
        self.val_emb = nn.Linear(c_out, d_model)
        self.pos_emb = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model,n_heads, d_ff) for _ in range(d_layers)])

    def forward(self, dec_inputs, enc_outputs): # dec_inputs : [batch_size x target_len x channels]
        dec_outputs = self.dropout(self.val_emb(dec_inputs) + self.pos_emb(dec_inputs))
        device = dec_inputs.device

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask=None)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class EnDecoder(nn.Module):
    def __init__(self, in_channel,sq_len,d_model,d_layers,dropout,n_heads,d_ff):
        super(EnDecoder, self).__init__()
        self.val_emb = nn.Linear(in_channel, d_model)
        self.pos_emb = positional_encoding('zeros', learn_pe=True, q_len=sq_len, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        # self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(de_in_len+1, d_model),freeze=True)
        self.layers = nn.ModuleList([DecoderLayer(d_model,n_heads, d_ff) for _ in range(d_layers)])

    def forward(self, dec_inputs, enc_outputs): # dec_inputs : [batch_size x target_len x channels]
        dec_outputs = self.dropout(self.val_emb(dec_inputs) + self.pos_emb)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask=None)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class ChannelAttention(nn.Module):
    def __init__(self, input_channels, time_len, d_model=128):
        super(ChannelAttention, self).__init__()
        self.d_model = d_model
        self.input_channels = input_channels
        self.time_len = time_len

        # Linear layers for Q, K, V
        # input_channels = time_series_len
        self.query_linear = nn.Linear(input_channels, d_model)
        self.key_linear = nn.Linear(input_channels, d_model)
        self.value_linear = nn.Linear(input_channels, d_model)
        
        # Softmax for attention scores
        self.softmax = nn.Softmax(dim=-1)

        self.W_concat=nn.Linear(d_model * time_len,d_model)

    def forward(self, x):
        # x is of shape [batch, channels, time_series_len ]
        B,L,C=x.shape
        # print(B,L,C)

        # x = x.permute(0, 2, 1) # [batch, channels, time_series_len]
        # Compute Q, K, V
        Q = self.query_linear(x)  # [batch, channels, d_model]
        K = self.key_linear(x)    # [batch, channels, d_model]
        V = self.value_linear(x)  # [batch, channels, d_model]
        
        # Scaled dot-product attention
        # Transpose K for matrix multiplication
        K_transposed = K.transpose(-2, -1)  # [batch, d_model, channels]
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K_transposed)  # [batch, channels, channels]
        attention_scores = attention_scores / (self.d_model ** 0.5)
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_scores)  # [batch, channels, channels]
        
        # Compute the output
        output = torch.matmul(attention_weights, V)  # [batch, channels, d_model]
        output = output.contiguous()

        output = output.view(B, self.time_len * self.d_model)

        output = self.W_concat(output)

        
        return output


    
class ConvBlock2(nn.Module):
    def __init__(self, kernel_size, window, dropout, channel):
        super(ConvBlock2, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=kernel_size,
                      stride=1, bias=False, padding=4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=kernel_size, stride=1,
                       bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=kernel_size, stride=1,
                       bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        # self.dropout = nn.Dropout(dropout)
        self.patch = int(window//(2**3))
        self.fc1 = nn.Linear(128 * self.patch, channel) 

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.view(x.size(0), -1)  # 这里的-1表示自动计算展平后的特征数量
        # 通过全连接层 + 激活函数 + Dropout
        x = self.fc1(x)
        # x = self.dropout(x)
        return x