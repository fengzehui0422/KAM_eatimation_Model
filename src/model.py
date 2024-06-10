import torch
import sys
import torch.nn as nn
import math, copy
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config.dataset import load_dataset
from loss import MLHLoss1, MLHLoss2, MLHLoss3
from sklearn.model_selection import KFold
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) # create 4 linear layers
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        # print ('Before transform query: ' + str(query.size())) # (batch_size, seq_length, d_model)  

        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))] # (batch_size, seq_length, d_model), use first 3 self.linears
        query, key, value = [x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for x in (query, key, value)] # (batch_size, h, seq_length, d_k)

        # print ('After transform query: ' + str(query.size()))

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)
class OutNet(nn.Module):
    def __init__(self, input_dim):
        super(OutNet, self).__init__()
        self.linear_1 = nn.Linear(input_dim, globals()['linear1_unit'], bias=True)
        self.linear_2 = nn.Linear(globals()['linear1_unit'], globals()['linear2_unit'] , bias=True)
        self.linear_3 = nn.Linear(globals()['linear2_unit'], globals()['linear3_unit'], bias=True)
        self.linear_4 = nn.Linear(globals()['linear3_unit'], 1, bias=True)
        self.dropout_layer = nn.Dropout(0.1)    
        for layer in [self.linear_1, self.linear_2]:
            nn.init.xavier_normal_(layer.weight)

    def forward(self, sequence, ele, height, weight, lens):
        # if len(self.high_level_locs) > 0:
        #     sequence = torch.cat((sequen
        # ce, others[:, :, self.high_level_locs]), dim=2)
        sequence = self.linear_1(sequence)
        sequence = self.dropout_layer(sequence)
        sequence = self.linear_2(sequence)
        sequence = self.dropout_layer(sequence)
        sequence = self.linear_3(sequence)
        sequence = self.dropout_layer(sequence)
        sequence = self.linear_4(sequence)        
        X = sequence.shape[0]
        Y = sequence.shape[1]
        sequence = sequence.view(X,Y)
        sequence = sequence / height / weight
        # weight = others[:, 0, WEIGHT].unsqueeze(1).unsqueeze(2)
        # height = others[:, 0, HEIGHT].unsqueeze(1).unsqueeze(2)
        # sequence = torch.div(sequence, weight * GRAVITY * height / 100)
        return sequence
class InertialNet(nn.Module):
    def __init__(self, x_dim, hidden_size, dims,
                 heads=4, num_layers = 2,
                 ratio=4,
                 p=0.1,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(InertialNet, self).__init__()
        self.dims = dims
        self.heads = heads
        self.length = 123
        self.relu = nn.ReLU()
        self.rnn_layer1 = nn.LSTM(x_dim - 1, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.rnn_layer2 = nn.LSTM(x_dim - 1, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.rnn_layer3 = nn.LSTM(x_dim, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.rnn_layer4 = nn.LSTM(x_dim, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.rnn_layer5 = nn.LSTM(x_dim, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.rnn_layer6 = nn.LSTM(x_dim, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.dropout_layer = nn.Dropout(0.25)
        self.norm13 = nn.LayerNorm(dims*2) 
        self.norm1 = nn.BatchNorm1d(self.length)   
        self.norm2 = nn.BatchNorm1d(self.length)   
        self.norm3 = nn.BatchNorm1d(self.length)   
        self.norm4 = nn.BatchNorm1d(self.length)   
        self.norm5 = nn.BatchNorm1d(self.length)   
        self.norm6 = nn.BatchNorm1d(self.length)   
        self.att1 = MultiHeadAttention(heads, dims)
        self.att2 = MultiHeadAttention(heads, dims)
        self.att3 = MultiHeadAttention(heads, dims)
        self.att4 = MultiHeadAttention(heads, dims)
        self.att5 = MultiHeadAttention(heads, dims)
        self.att6 = MultiHeadAttention(heads, dims)
        self.att7 = MultiHeadAttention(heads, dims)
        self.att8 = MultiHeadAttention(heads, dims)
        self.att9 = MultiHeadAttention(heads, dims)
        self.att10 = MultiHeadAttention(heads, dims)
        self.att11 = MultiHeadAttention(heads, dims)
        self.att12 = MultiHeadAttention(heads, dims) 
    def forward(self, sequence, ele, lens):
        right_z, left_z, right_x, left_x, right_ele, left_ele, right_max, left_max = (
            sequence[:, :, [2, 5, 8, 11]], sequence[:, :, [14, 17, 20, 23]],
            sequence[:, :, [0, 3, 6, 9]], sequence[:, :, [12, 15, 18, 21]],
            sequence[:, :, 24:27], sequence[:, :, 27:30],
            ele[:, :, 0:3], ele[:, :, 3:6]
        )

        right_ele = right_ele * right_max
        right_ele, _ = self.rnn_layer1(right_ele)
        left_ele = left_ele * left_max
        left_ele, _ = self.rnn_layer2(left_ele)   
        right_z, _ = self.rnn_layer3(right_z)
        left_z, _ = self.rnn_layer4(left_z)
        right_x, _ = self.rnn_layer5(right_x)
        left_x, _ = self.rnn_layer6(left_x)

        zt1 = self.att1(right_z, left_z, left_z, mask=None)
        zt2 = self.att2(left_z, right_z, right_z, mask=None)
        rz = torch.cat((zt2, zt1), dim=2)
        xt1 = self.att3(right_x, left_x, left_x, mask=None)
        xt2 = self.att4(left_x, right_x, right_x, mask=None)
        rx = torch.cat((xt2, xt1), dim=2)
        fz1 = self.att5(right_z, right_ele, right_ele, mask=None)
        fz2 = self.att6(right_ele, right_z, right_z, mask=None)
        fz3 = self.att7(left_z, left_ele, left_ele, mask=None)
        fz4 = self.att8(left_ele, left_z, left_z, mask=None)
        fz = torch.cat((fz1 + fz2, fz3 + fz4), dim=2) 
        fx1 = self.att9(right_x, right_ele, right_ele, mask=None)
        fx2 = self.att10(right_ele, right_x, right_x, mask=None)
        fx3 = self.att11(left_x, left_ele, left_ele, mask=None)
        fx4 = self.att12(left_ele, left_x, left_x, mask=None)
        fx = torch.cat((fx1 + fx2, fx3 + fx4), dim=2)
        out = fx * rz - rx * fz
        return out
class DirectNet(nn.Module):
    """ Implemented based on the paper "Efficient low-rank multimodal fusion with modality-specific factors" """
    def __init__(self, input_size, hidden_size, dims, heads=2, num_layers=1):
        super(DirectNet, self).__init__()
        self.attn = InertialNet(input_size, hidden_size, dims,
                 heads=2, num_layers = 1)
        self.out_net = OutNet(globals()['lstm_unit']*2)
    def __str__(self):
        return 'Direct fusion net'
    def forward(self, acc_x, mea, lens):
        mea = mea.unsqueeze(1).repeat(1, sequence_length, 1)
        ele, height, weight = mea[ : , : , 0 : 6 ], mea[: , : , 6], mea[: ,  : , 7]
        acc_h = self.attn(acc_x, ele, lens)
        sequence = self.out_net(acc_h, ele, height, weight, lens)
        return sequence
