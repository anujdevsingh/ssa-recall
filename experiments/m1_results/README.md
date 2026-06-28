# M1 results (committed artifacts)

This folder holds the **permanent record** of the M1 recall-vs-state-size experiment, so every
result is tracked in git and reproducible:

- `sweep.log` — the raw `zoology.launch` output for all 24 runs (every config + training curve).
- `recall_frontier.png` — the parsed figure: recall accuracy vs `d_model`, one curve per architecture.
- `RESULTS.md` — the parsed frontier table (best accuracy per architecture × `d_model`).

Config that produced it: [`experiments/m1_recall_frontier.py`](../m1_recall_frontier.py).
Design rationale: [`docs/m1-design.md`](../../docs/m1-design.md).

> Transient outputs under `results/` are git-ignored; only curated artifacts copied here are committed.
