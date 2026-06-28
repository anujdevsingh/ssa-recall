"""Tiny sequence model with a swappable token mixer.

Two mixers for M1:
  - CausalSelfAttention : full softmax attention (the recall ceiling)
  - LinearAttention     : fixed-state linear attention (the recall-weak baseline)

# ponytail: LinearAttention uses the parallel cumsum form (verified == the naive
# time-loop); swap to flash-linear-attention kernels in M2 for the real baselines.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, L, D = x.shape
        hd = D // self.h
        qkv = self.qkv(x).reshape(B, L, 3, self.h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(o)


class LinearAttention(nn.Module):
    """Causal linear attention with elu+1 feature map and a fixed-size KV state."""
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, L, D = x.shape
        hd = D // self.h
        qkv = self.qkv(x).reshape(B, L, 3, self.h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        # parallel causal form: S_t = cumsum_t (k_t outer v_t), Z_t = cumsum_t k_t
        S = torch.cumsum(k.unsqueeze(-1) * v.unsqueeze(-2), dim=2)   # (B,h,L,hd,hd)
        Z = torch.cumsum(k, dim=2)                                   # (B,h,L,hd)
        num = torch.einsum("bhli,bhlij->bhlj", q, S)
        den = (q * Z).sum(-1, keepdim=True) + 1e-6
        o = (num / den).transpose(1, 2).reshape(B, L, D)
        return self.proj(o)


MIXERS = {"attention": CausalSelfAttention, "linear": LinearAttention}


class Block(nn.Module):
    def __init__(self, d, heads, mixer_cls):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.mix = mixer_cls(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.mix(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SeqModel(nn.Module):
    def __init__(self, vocab, mixer="attention", d=64, heads=2, layers=2, max_len=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, heads, MIXERS[mixer]) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        L = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(L, device=x.device))
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln(h))
