# ssa-recall

Small-scale study of **recall-preserving sub-quadratic attention**. Does content-based sparse
selection preserve associative recall better than the fixed-size state of linear/SSM models?

This is the **engine**, not the car: we test the *mechanism* on tiny models + short synthetic
sequences (MQAR). We are **not** building a 12M-context LLM.

## M1 — reproduce the recall-vs-state frontier (Option A)

The credible M1 is **reproducing Zoology's Figure 2** — recall accuracy vs. model dimension
(state size) for attention vs. efficient architectures on MQAR — using the validated
[`zoology`](https://github.com/HazyResearch/zoology) harness, which already implements the
baselines (GLA, Based, DeltaNet, Mamba) **and DeepSeek native sparse attention** (our SSA-family
reference).

- **The science / design:** [`docs/m1-design.md`](docs/m1-design.md) — research question,
  controlled variables, target figure. Read this first.
- **The runbook:** [`docs/m1-zoology-setup.md`](docs/m1-zoology-setup.md) — staged Colab setup.

Why not hand-roll: tuning a ceiling model until it "wins" confounds architecture with capacity.
The result is a *matched-budget scaling frontier*, not a single accuracy number.

## The toy (`src/`, `run_m1.py`) — sanity check only

A from-scratch MQAR + tiny attention-vs-linear model. Kept as a pedagogical sanity check (it's how
we found the induction-head phase-transition / warmup lesson). **Not** the basis for results — the
Zoology reproduction is. Run it with `python run_m1.py`.

## Next

- M2: add/evaluate the **SSA-style sparse-attention** mixer on the same frontier, vs. NSA + the
  fixed-state baselines.
- M3+: recall-vs-length and recall-vs-compute frontiers. Full plan in the KnowledgeBank wiki:
  `wiki/analyses/ssa-recall-research-6month-plan.md`.
