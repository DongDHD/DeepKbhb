import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.utils.weight_norm import weight_norm


d_k = 32
d_v = 32


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=16):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(EmbeddingLayer, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)

    def forward(self, input_ids):
        x = self.src_emb(input_ids)
        embeddings = self.pos_emb(x.transpose(0, 1)).transpose(0, 1)
        return embeddings

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.3):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class MultiKernelCNN(nn.Module):
    def __init__(self, d_model, kernel_sizes=[2, 3, 5]):
        super(MultiKernelCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2),
                nn.ReLU()
            ) for k in kernel_sizes
        ])

    def forward(self, x):

        x = x.transpose(1, 2)  
        conv_outs = [conv(x) for conv in self.convs]
 
        max_len = min([out.size(-1) for out in conv_outs])
        conv_outs = [out[:, :, :max_len] for out in conv_outs]
        combined = torch.cat(conv_outs, dim=1)  
        return combined.transpose(1, 2) 

class FCNet(nn.Module):
    def __init__(self, dims, act='ReLU', dropout=0.):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class BCNet(nn.Module):

    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2, .5], k=3):
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim;
        self.q_dim = q_dim
        self.h_dim = h_dim;
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])  # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if None == h_out:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            logits = torch.einsum('bvk,bqk->bvqk', (v_, q_))
            return logits

        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v))
            q_ = self.q_net(q)
            logits = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
            return logits  # b x h_out x v x q

        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3)) 
            return logits.transpose(2, 3).transpose(1, 2) 

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v)  
        q_ = self.q_net(q)  
        logits = torch.einsum('bvk,bvq,bqk->bk', (v_, w, q_))
        if 1 < self.k:
            logits = logits.unsqueeze(1) 
            logits = self.p_net(logits).squeeze(1) * self.k 
        return logits

class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2, .5], k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.bc_net = BCNet(h_dim * self.k, h_dim * self.k, h_dim, h_out, act=act, dropout=dropout)

        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):

        att_map = att_map.view(att_map.size(0), att_map.size(1), -1)  

        v = v.squeeze(1)  
        q = q.squeeze(1)  
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', v, att_map, q)
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        v_ = self.v_net(v).unsqueeze(1)  
        q_ = self.q_net(q).unsqueeze(1)  

        if self.h_out <= self.c:
            att_maps = torch.einsum('xhyk,bvpk,bqpk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = v_.transpose(1, 2).unsqueeze(3) 
            q_ = q_.transpose(1, 2).unsqueeze(2)  
            d_ = torch.matmul(v_, q_)  
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)

        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)

        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i

        logits = self.bn(logits)
        return logits, att_maps


class PepNet(nn.Module):
    def __init__(self, vocab_size, d_model, esm_feature_dim, n_heads, d_ff, n_transformer_layers, dropout=0.5):
        super(PepNet, self).__init__()
        self.emb = EmbeddingLayer(vocab_size, d_model)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, d_ff, dropout=dropout) for _ in range(n_transformer_layers)]
        )
        self.cnn = MultiKernelCNN(d_model, kernel_sizes=[2, 3, 5])
        self.cnn_proj = nn.Linear(d_model * len([2, 3, 5]), d_model) 
        self.esm_proj = nn.Linear(esm_feature_dim, d_model)
        self.ban_layer = BANLayer(v_dim=d_model, q_dim=d_model, h_dim=d_model, h_out=n_heads, k=3)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, esm_features):
        seq_embeddings = self.emb(input_ids)
        transformer_out = self.transformer_blocks(seq_embeddings)


        cnn_out = self.cnn(transformer_out) 
        cnn_out = self.cnn_proj(torch.mean(cnn_out, dim=1)) 


        esm_out = self.esm_proj(esm_features)  
        ban_out, _ = self.ban_layer(cnn_out.unsqueeze(1), esm_out.unsqueeze(1))  

        logits = self.fc(ban_out)
        return logits
