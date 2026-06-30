"""HierSparseAttention — M2b: content-based sparse attention with a SUB-QUADRATIC selector.

Drop-in zoology sequence-mixer (interface matches zoology.mixers.*: __init__(d_model, **kw),
forward(u: (B,L,D)) -> (B,L,D)). Same three-branch shape as DeepSeek NSA
(sliding-window + coarse + fine selection) with ONE thing changed: the coarse stage that
decides *which blocks to attend to* is HIERARCHICAL, not a dense scan.

WHY THIS IS THE M2b CONTRIBUTION (read before touching the math):
  M2a showed NSA rides the attention recall ceiling on MQAR, BUT NSA's selector is quadratic.
  The quadratic cost is NOT the top-k. It is the COARSE branch: every query attends over ALL
  n/compress_block_size compressed blocks  ->  O(n^2 / B). That same dense score feeds top-k.
  So to make *selection* cheap you must make the *coarse stage* cheap.

  HierSparseAttention replaces the dense coarse attention with a 2-level tree:
    level-1: super-block summaries   (n / (cb*g) of them)   <- scored densely (small)
    level-0: compressed blocks       (only the g members of the top-s1 super-blocks)  <- gathered
  Per query the coarse stage scores  nsup + s1*g  nodes instead of nblk = n/cb.
  With g ~ sqrt(nblk) -> nsup ~ sqrt(nblk) and s1*g ~ sqrt(nblk) -> O(sqrt(n)) per query
  -> O(n^1.5) total. THE HARNESS MUST SET g = round(sqrt(nblk)) PER SEQUENCE LENGTH; a fixed g
  stays O(n^2). The fine blocks for exact attention fall out of the descent.

  => preserves NSA's recall mechanism (exact attention over selected past blocks) at
     sub-quadratic selection cost. M2a = the recall half; this = the cheap half = the paper.

GRADIENT (the #1 correctness trap): top-s / top-k descent is hard in the forward pass. Gradients
reach the scoring projection via (a) the softmax over the *visited* candidate-block scores in the
coarse output, and (b) a straight-through gate on the selected importance values feeding the fine
branch. Hard pruning in the gradient path would stop the selector learning *which* blocks matter
-> recall collapse. The self-test asserts grad reaches qkv.weight.

CAUSALITY: coarse uses only *fully-past* blocks (block_end <= t) so block-mean summaries never
include future tokens. The current/recent region is covered by the sliding window. Clean & exact.

INSTRUMENTATION: after each forward, self.coarse_pairs_per_query (= nsup + s1*g) and
self.dense_pairs_per_query (= nblk, what NSA pays) let M3 plot MEASURED algorithmic selection cost
vs length. NOTE: this prototype scores via gather+mask, so wall-clock is NOT the kernel a fused
sparse impl would hit; report the scored-pair COUNT as the cost metric and treat wall-clock as
indicative only ("complexity moved, not removed" cuts both ways — say so in the paper).

UNTESTED LOCALLY (Windows torch is broken). Gate 1 = run `python src/hier_nsa.py` on Colab/Kaggle
(the self-test). Gate 2 = one seq128 MQAR run matches NSA recall before any sweep.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e9


def _block_pool(x, block):
    """(B,h,N,d) -> (B,h,ceil(N/block),d) mean over each block; N padded with zeros."""
    B, h, N, d = x.shape
    pad = (-N) % block
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    nb = x.shape[2] // block
    return x.view(B, h, nb, block, d).mean(3), nb


def _gather_blocks(src, blk_idx):
    """src:(B,h,N,d), blk_idx:(B,h,L,M) block ids -> (B,h,L,M,d). Invalid ids must be pre-clamped."""
    B, h, N, d = src.shape
    L, M = blk_idx.shape[2], blk_idx.shape[3]
    idx = blk_idx.clamp(0, N - 1).unsqueeze(-1).expand(B, h, L, M, d)
    return src.unsqueeze(2).expand(B, h, L, N, d).gather(3, idx)


class HierSparseAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, sliding_window_size=32,
                 compress_block_size=16, selection_block_size=16,
                 num_selected_blocks=4, superblock_group=4,
                 num_selected_superblocks=2, causal=True, **kw):
        super().__init__()
        # prototype ties the coarse leaf = compress = selection block (one block granularity)
        assert compress_block_size == selection_block_size, "keep compress==selection block size"
        self.h = num_heads
        self.w = sliding_window_size
        self.cb = compress_block_size
        self.k_sel = num_selected_blocks
        self.g = superblock_group              # harness sets ~ round(sqrt(nblk)); fixed g => O(n^2)
        self.s1 = num_selected_superblocks
        self.causal = causal
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.gate = nn.Linear(d_model, 3 * num_heads)
        self.proj = nn.Linear(d_model, d_model)
        self.coarse_pairs_per_query = 0
        self.dense_pairs_per_query = 0          # = nblk, what NSA's coarse branch scores

    def forward(self, u):
        B, L, D = u.shape
        h, hd = self.h, D // self.h
        cb, g, s1, k_sel = self.cb, self.g, self.s1, self.k_sel
        dev = u.device
        scale = 1.0 / math.sqrt(hd)
        q, k, v = self.qkv(u).reshape(B, L, 3, h, hd).permute(2, 0, 3, 1, 4)  # each (B,h,L,hd)
        t = torch.arange(L, device=dev)                                       # query positions

        # ---- summaries over fully-past blocks / super-blocks ----
        bk, nblk = _block_pool(k, cb)                 # (B,h,nblk,hd)
        bv, _ = _block_pool(v, cb)
        sk, nsup = _block_pool(bk, g)                 # (B,h,nsup,hd)  (nblk padded to multiple of g)
        blk_end = (torch.arange(nblk, device=dev) + 1) * cb          # first t for which block is past
        sup_end = (torch.arange(nsup, device=dev) + 1) * g * cb
        self.coarse_pairs_per_query = nsup + s1 * g
        self.dense_pairs_per_query = nblk

        # ---- level 1: score query vs super summaries, pick top-s1 past super-blocks ----
        ss = torch.einsum("bhld,bhsd->bhls", q, sk) * scale          # (B,h,L,nsup)
        ss = ss.masked_fill(sup_end[None, None, None, :] > t[None, None, :, None], NEG)
        s1e = min(s1, nsup)
        sup_idx = ss.topk(s1e, dim=-1).indices                       # (B,h,L,s1e)

        # ---- level 0: gather member blocks of selected supers, score them ----
        member = (sup_idx.unsqueeze(-1) * g + torch.arange(g, device=dev)).reshape(B, h, L, s1e * g)
        valid = member < nblk                                        # padded supers overrun nblk
        cbk = _gather_blocks(bk, member)                             # (B,h,L,s1e*g,hd)
        cbv = _gather_blocks(bv, member)
        bs = torch.einsum("bhld,bhlmd->bhlm", q, cbk) * scale        # (B,h,L,s1e*g)
        be = blk_end[member.clamp(0, nblk - 1)]                      # block_end per candidate
        bs = bs.masked_fill((be > t[None, None, :, None]) | ~valid, NEG)

        # coarse output: softmax over visited candidate blocks (this is the gradient path to scores).
        # zero the branch where NO block is fully-past, else the all-masked softmax leaks future tokens.
        coarse_ok = (bs.max(-1, keepdim=True).values > NEG / 2).float()
        coarse = torch.einsum("bhlm,bhlmd->bhld", bs.softmax(-1).nan_to_num(0.0), cbv) * coarse_ok

        # ---- fine selection: top-k candidate blocks -> exact attention over their raw tokens ----
        ke = min(k_sel, s1e * g)
        fin = bs.topk(ke, dim=-1)
        fin_blk = torch.gather(member, -1, fin.indices)              # (B,h,L,ke) chosen block ids
        gate_k = torch.sigmoid(fin.values).unsqueeze(-1)             # straight-through importance gate
        tok = (fin_blk.unsqueeze(-1) * cb + torch.arange(cb, device=dev)).reshape(B, h, L, ke * cb)
        tok_ok = tok < L
        kf = _gather_blocks(k, tok)                                  # gather tokens, treat tok as ids
        vf = _gather_blocks(v, tok)
        fs = torch.einsum("bhld,bhlmd->bhlm", q, kf) * scale         # (B,h,L,ke*cb)
        fs = fs.masked_fill((tok > t[None, None, :, None]) | ~tok_ok, NEG)
        # broadcast the per-block straight-through gate across its cb tokens
        fs = fs + gate_k.expand(B, h, L, ke, cb).reshape(B, h, L, ke * cb).log().nan_to_num(NEG, NEG, NEG)
        fine_ok = (fs.max(-1, keepdim=True).values > NEG / 2).float()
        fine = torch.einsum("bhlm,bhlmd->bhld", fs.softmax(-1).nan_to_num(0.0), vf) * fine_ok

        # ---- sliding window (banded, causal) ----
        off = torch.arange(self.w, device=dev)
        wpos = t[None, :] - off[:, None]                             # (w,L) positions t,t-1,...
        wpos_t = wpos.t()                                           # (L,w)
        widx = wpos_t.clamp(0, L - 1)[None, None].expand(B, h, L, self.w)
        kw = _gather_blocks(k, widx)
        vw = _gather_blocks(v, widx)
        wscore = torch.einsum("bhld,bhlmd->bhlm", q, kw) * scale
        wscore = wscore.masked_fill(wpos_t[None, None] < 0, NEG)
        win = torch.einsum("bhlm,bhlmd->bhld", wscore.softmax(-1).nan_to_num(0.0), vw)

        # ---- gated combine (per-head sigmoid weights, NSA-style, not normalized) ----
        gates = torch.sigmoid(self.gate(u)).reshape(B, L, h, 3).permute(0, 2, 1, 3)  # (B,h,L,3)
        out = (gates[..., 0:1] * coarse + gates[..., 1:2] * fine + gates[..., 2:3] * win)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


def _selftest():
    torch.manual_seed(0)
    B, L, D = 2, 128, 64
    m = HierSparseAttention(D, num_heads=4, compress_block_size=16, selection_block_size=16,
                            num_selected_blocks=4, superblock_group=3, num_selected_superblocks=2)
    x = torch.randn(B, L, D, requires_grad=True)
    y = m(x)
    assert y.shape == (B, L, D), y.shape
    assert torch.isfinite(y).all(), "NaN/inf in output"

    # causality: perturbing position p must not change outputs at positions < p.
    # check an early p (catches future-leak via block summaries) and the midpoint.
    for p in (3, L // 2):
        x2 = x.detach().clone(); x2[:, p] += 5.0
        y2 = m(x2)
        assert torch.allclose(y[:, :p], y2[:, :p], atol=1e-5), f"causality violated at p={p}"

    # gradient reaches the scoring projection (selector must be learnable)
    y.sum().backward()
    assert m.qkv.weight.grad is not None and m.qkv.weight.grad.abs().sum() > 0, "no grad to qkv"
    assert m.gate.weight.grad is not None, "no grad to gate"

    print(f"OK  out={tuple(y.shape)}  coarse_pairs/q={m.coarse_pairs_per_query} "
          f"vs NSA-dense/q={m.dense_pairs_per_query}  (saving "
          f"{m.dense_pairs_per_query / m.coarse_pairs_per_query:.2f}x)")


if __name__ == "__main__":
    _selftest()
