# M1 — Experimental Design (Option A: reproduce Zoology, then extend)

## Why this design (the methodological point)

Recall research is **not** "show attention beats a linear model." That is trivially true and
confounded: if you let the ceiling model be bigger or train longer, you have proven nothing. The
real object of study is the **Pareto frontier of recall vs. recurrent state size** (Arora et al.,
Zoology / Based). Architectures differ in *how much state they need* to recall a given number of
associations:

- **Attention** keeps an O(n) KV cache → recalls many associations at *small* model width. It is
  the ceiling because of its memory, not its parameter count.
- **Linear / SSM models** compress to a **fixed-size state** → recall degrades once the number of
  associations exceeds what that state holds; required state grows with the number of KV pairs.
- **Content-based sparse attention (NSA / SSA)** keeps exact token representations but selects
  *computation* sparsely → the hypothesis is that it preserves recall like attention at
  sub-quadratic cost. **This is our contribution's target.**

## Research question

> On the recall-vs-state-size frontier, does content-based sparse attention (DeepSeek NSA, and
> later an SSA-style variant) sit on or near the attention ceiling while remaining sub-quadratic —
> i.e., does it dominate the fixed-state linear/SSM baselines at matched budget?

## Controlled experiment

| Element | Choice |
|---|---|
| **Independent variable** | model dimension `d_model` (≈ state size for recurrent models), swept e.g. {64, 128, 256, 512} |
| **Held fixed** | MQAR difficulty (`input_seq_len`, `num_kv_pairs`, `vocab_size`), training recipe (epochs, optimizer, LR schedule), seed protocol |
| **Controls** | matched `d_model` across architectures; same data; LR tuned per-arch via a small sweep so no architecture is handicapped by optimization (the lesson from the toy: undertuning looks like an architectural deficit) |
| **Dependent variable** | held-out MQAR recall accuracy |
| **Architectures** | `MHA` (attention ceiling) · `Based` and/or `GLA` (fixed-state baseline) · **DeepSeek NSA** (sparse-attention reference) |
| **Target figure** | accuracy vs. `d_model`, one curve per architecture (this is Zoology Fig. 2) |

Expected shape: attention flat-high across `d_model`; Based/GLA rising with `d_model` but lagging;
NSA near attention. The **gap between the efficient curve and the attention ceiling, as a function
of state size**, is the result — not any single number.

## Why reproduce Zoology rather than hand-roll

The hand-rolled toy (`src/`, retained as a sanity check) taught us the induction-head phase
transition, but rebuilding the harness means re-deriving training recipes Zoology already
validated, and re-implementing baselines (GLA, Based, DeltaNet, **NSA**) that already exist there.
Reproducing `zoology/experiments/paper_configs/iclr24_zoology_figure2` with their code is the
credible, faster foundation. NSA being built-in means our SSA contribution plugs straight into a
validated testbed.

## M1 done =

A reproduced accuracy-vs-`d_model` plot on MQAR showing the attention ceiling above a fixed-state
baseline that closes the gap only as `d_model` grows — at matched budget, LR-tuned per arch. Then
M2 adds/evaluates the SSA-style sparse mixer on the *same* frontier.

## Staging (de-risk install before any sweep)

See `docs/m1-zoology-setup.md`. Order: (1) clone + minimal install, (2) one known-good MQAR run to
prove the environment, (3) add `fla` for the efficient/sparse mixers, (4) reduced `d_model` sweep
across the 3–4 architectures, (5) plot the frontier.
