# KDA / Kimi Linear — where the strongest fixed-state model sits relative to this project

*Context note, 2026-07-23. Capture of the framing worked out while M2c was in flight, so the
write-up (M5) and the M3 baseline choice don't have to re-derive it.*

## What KDA is

**Kimi Delta Attention (KDA)** is Moonshot AI's linear-attention module, introduced in
[Kimi Linear (arXiv 2510.26692, Oct 2025)](https://arxiv.org/abs/2510.26692). Lineage:
DeltaNet → Gated DeltaNet (GDN) → KDA. Like all delta-rule models it maintains a
**constant-size matrix state** per head (d_k × d_v), updated by targeted erase-then-write
rather than Mamba-style uniform decay. KDA's additions over GDN:

- **Per-channel forget gates** — each feature dimension controls its own decay (GDN gates
  per head). Finer temporal control over what the fixed state keeps.
- A hardware-efficient **chunked algorithm** using a constrained Diagonal-Plus-Low-Rank
  transition (cheaper than general DPLR, closer to the classical delta rule).

The production architecture, **Kimi Linear**, is a **hybrid: 3 KDA layers per 1 full-attention
(MLA) layer** (global MLA layers run without positional encoding; KDA carries position).
Reported at 48B-A3B scale: ≈75% KV-cache reduction and up to ~6× decoding throughput at 1M
context, matching or beating the full-MLA baseline under matched-budget training.

**Status as of 2026-07-23:** this is now a frontier production architecture, not a paper
curiosity — **Kimi K3 (2.8T params, released 2026-07-16, open weights announced for
2026-07-27)** is KDA-dominant with periodic full-attention layers
([vLLM production preview, 2026-07-22](https://vllm.ai/blog/2026-07-22-kimi-k3-preview)).
Dedicated kernels exist
([FlashKDA, open-sourced 2026-04-30](https://github.com/MoonshotAI/FlashKDA)), and
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) ships a
training-ready layer (`fla/layers/kda.py`, added Oct 2025 —
[PR #621](https://github.com/fla-org/flash-linear-attention/pull/621)).

## Where it sits in our taxonomy

KDA is the best-engineered member of the **fixed-state** family — the *other* answer to the
recall wall, not a rival version of ours:

| | KDA (fixed-state) | hier / NSA / SSA (sparse exact attention) |
|---|---|---|
| History storage | compressed into a constant-size state | kept exactly (full KV cache) |
| Recall mechanism | hope the state kept the right binding | fetch the right blocks, attend exactly |
| Cost | O(n) time, O(1) state | sub-quadratic compute, O(n) memory |
| What bounds recall | **state capacity** | **selector quality** (← the M2c finding) |

## Reading the Kimi paper's MQAR claims precisely

The paper reports KDA reaching ~100% on MQAR at seq 256–2048, converging ~2× faster than GDN.
Take this seriously but precisely:

1. Delta-rule models were already the strongest fixed-state family on MQAR — targeted
   overwrite is much better at key→value binding than decay-only models (Mamba,
   gated conv). KDA would **not** sit at our base_conv floor (~0.13); it starts far above it.
2. *Any* fixed-state model solves MQAR when its state is large relative to the number of
   KV pairs. The Zoology framing — ours — is that recall is a **frontier against state
   size**: crank `num_kv_pairs` at fixed capacity and a constant-size state must
   eventually drop pairs, by an information-capacity argument no gating mechanism escapes.
   Better gating moves the wall; it cannot remove it.
3. The Kimi synthetics do **not** stress that axis the way our difficulty-scaled sweep does
   (128 pairs at L2048). Where KDA lands on *our* frontier is an open empirical question —
   likely well above Mamba/DeltaNet, plausibly still below the exact-retrieval ceiling at
   high pair counts. That is a runnable M3 experiment, not a threat to the thesis.

## Why this strengthens the paper's motivation

- **The hybrid ratio is a confession.** Moonshot pairs every 3 KDA layers with a full
  quadratic-attention layer because pure linear attention is not trusted for retrieval at
  scale. The industry's current answer to the recall wall is *pay O(n²) on ¼ of layers as
  recall insurance*. SubQ claims the opposite extreme (sparse-only, no full attention,
  selector linear too). Our project tests exactly the load-bearing middle question: **can
  the exact-attention component itself be sub-quadratic without losing recall?** K3
  shipping on a hybrid makes that question more current, not less.
- **Composability.** The two lines are orthogonal and literally stackable: a hybrid's full
  attention layers could be replaced by recall-preserving sparse ones (hier), compounding
  the KV/compute savings. Discussion-section point, not an experiment we run.
- **Timing.** With K3 weights going public, an independent small-scale map of
  *fixed-state (incl. KDA-class) vs. sparse-exact recall, and what the selector costs*
  lands at peak relevance.

## Concrete impact on the plan

- **M2c: no change.** Selector-budget attribution is orthogonal to all of this.
- **M3: add KDA as a fixed-state baseline** alongside (or ahead of) Mamba/DeltaNet, via
  `fla.layers.kda` wrapped as a zoology mixer — same integration path already planned for
  the other fla baselines. If hier holds recall where even KDA's curve bends down at
  matched state, that is a far stronger figure than beating Mamba.
- **M5 write-up: cite Kimi Linear + K3 in the motivation** as (a) the current linear
  SOTA and (b) evidence that frontier deployments still buy recall with exact-attention
  layers — then position our question as whether that exact-attention component needs to
  be quadratic at all.

## Sources

- [Kimi Linear: An Expressive, Efficient Attention Architecture (arXiv 2510.26692)](https://arxiv.org/abs/2510.26692)
- [MoonshotAI/Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear) · [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA)
- [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) · [KDA PR #621](https://github.com/fla-org/flash-linear-attention/pull/621)
- [FlashKDA kernels write-up (MarkTechPost, 2026-04-30)](https://www.marktechpost.com/2026/04/30/moonshot-ai-open-sources-flashkda-cutlass-kernels-for-kimi-delta-attention-with-variable-length-batching-and-h20-benchmarks/)
- [vLLM: Kimi K3 production preview (2026-07-22)](https://vllm.ai/blog/2026-07-22-kimi-k3-preview) · [DataCamp K3 overview](https://www.datacamp.com/blog/kimi-k3)
