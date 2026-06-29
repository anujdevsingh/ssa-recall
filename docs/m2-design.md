# M2 — designing an SSA-style mixer and testing it on the recall frontier

## What SubQ's report actually tells us (and what it hides)

From the SubQ-1.1-Small technical report (Subquadratic AI, 2026):

- **SSA = Subquadratic Sparse Attention** — a *content-dependent* sparse attention with **linear**
  compute **and** memory in sequence length. Result: 64.5× fewer attention FLOPs at 1M tokens,
  99.12% RULER@128K, 98% single-needle recall at 12M while attending to **0.13% of token pairs**.
- **The one idea that matters** (their §2.3, §5.6): the hard part of sparse attention is not the
  sparsity — it's the **cost of *deciding* which positions to keep**. DeepSeek's NSA / DSA uses a
  "Lightning Indexer" that scores *all* query–key pairs → **quadratic**, and that selector overtakes
  the attention it serves past ~52K tokens (16× the attention cost at 1M, 190× at 12M). SSA's claim
  is that its **selection step is itself linear** — "selection, retrieval, and attention steps are
  each linear," so it's linear *end-to-end*, not just in the attention read.
- **Naming hint:** their cost table labels SSA as **"DBSA"** → Dynamic **Block** Sparse Attention.
  So the mechanism is almost certainly **block-sparse** with a **cheap (linear) block selector**.
- **The mechanism itself is undisclosed.** They publish three *requirements*, not the algorithm:
  1. dense-attention-level retrieval from **arbitrary positions** (routing must be content-dependent),
  2. **sub-quadratic end-to-end, including the selection/indexing stage**,
  3. **full-context training** + standard **autoregressive** decoding.

So we can't copy SSA. We **design our own** mechanism that satisfies those three requirements, and
measure it on the same recall frontier we built in M1.

## Our SSA-style design (what we implement & test)

A block-sparse, content-routed attention with three branches (the NSA shape, which is the closest
public reference — and zoology already ships it):

1. **Compress.** Split the K/V stream into blocks of size `B`; produce one learned summary vector
   per block. Cost O(n); gives `n/B` coarse keys. *(meets req-1's coverage, cheaply)*
2. **Select (content-dependent).** Each query scores the block summaries and keeps the **top-k
   blocks**, then does exact attention over the tokens inside those blocks. Arbitrary-position access
   because any block can be chosen by content. *(meets req-1 retrieval)*
3. **Local window.** Each query always attends to a fixed sliding window of recent tokens (local
   structure). *(stabilises training)*
   - The three branch outputs are merged by a learned gate; decoding stays causal/autoregressive
     *(meets req-3)*.

**Where req-2 lives — and our actual contribution.** Plain top-k block scoring still scores every
query against every block (`O(n²/B)` — NSA's "complexity moved, not removed"). SubQ's secret is
making *selection itself* linear. So our research knob is the **selector**:
- **Baseline selector:** score all `n/B` block summaries (NSA-style). Reference point.
- **Our variant:** a **hierarchical / coarse-to-fine selector** — summaries of summaries, so a query
  descends a short tree instead of scanning all blocks → selection cost grows ~`O(n log n)` or
  `O(n)`, not `O(n²/B)`. This is the SSA-shaped "cheap selector" (their DBSA-vs-DSA Table 3) cast as
  something we can implement and measure at small scale.

## The experiment (the M2 figure)

Add an **SSA/NSA curve** to the exact M1 plot (recall accuracy vs `d_model`, MQAR seq 128, 8 KV
pairs, vocab 8192):

```
~0.99 ── attention   (full memory, O(n²))          ← M1 ceiling
   ?  ── SSA (ours)   content-based block-sparse    ← M2: where does it land?
~0.13 ── base_conv    (fixed state, forgets)        ← M1 floor
```

- Rides the ceiling → **content-based sparse selection preserves recall** at a fraction of the
  compute. The headline result.
- Falls to the floor → block selection didn't capture the bindings; also a real, publishable finding.

Two axes, both reported: **recall accuracy** (does it remember?) and **selection cost / sparsity**
(how cheap is the selector?) — the second is where the hierarchical-selector variant earns its keep.

## Staging

1. **M2a — reference.** Drop zoology's built-in `zoology.mixers.deepseek_nsa.SparseAttention` into
   the M1 sweep (config: `experiments/m2_ssa_frontier.py`). Smoke-test ONE NSA run first (block
   sizes must fit `seq_len`), then run the 3-arch × d_model sweep. This gives the content-based
   sparse-attention curve vs the M1 attention/base_conv baselines.
2. **M2b — our selector (the novelty).** Implement the hierarchical/linear block selector as a new
   mixer, drop it on the same plot, and compare recall **and** selector FLOPs against NSA.

M2a is mostly plumbing on a validated harness; M2b is the contribution.
