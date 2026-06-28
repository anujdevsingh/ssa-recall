# ssa-recall

Small-scale study of **recall-preserving sub-quadratic attention**. Does content-based sparse
selection preserve associative recall better than the fixed-size state of linear/SSM models?

This is the **engine**, not the car: we test the *mechanism* on tiny models + short synthetic
sequences (MQAR). We are **not** building a 12M-context LLM.

## M1 (Week 1) — Reproduce the recall gap

Goal: show the known result — **softmax attention solves MQAR; a fixed-state linear-attention
baseline degrades as the number of key→value pairs grows.** If we can't reproduce this gap, we
fix that before doing anything else.

```bash
pip install -r requirements.txt
python src/mqar.py        # sanity-check the data generator
python run_m1.py --smoke  # ~1 min: tiny run, confirms everything wires up
python run_m1.py          # the real M1 sweep -> prints a table + saves results/recall_gap.png
```

Expected shape of the result: attention accuracy stays ~high across `num_kv_pairs`;
linear-attention accuracy falls off as pairs increase. That falloff is the recall wall this
whole project is about.

## Layout

```
src/mqar.py     synthetic MQAR data + a self-check
src/models.py   tiny seq model with swappable mixer: causal attention | linear attention
src/train.py    train one (mixer, num_kv_pairs) config, return recall accuracy
run_m1.py       sweep over num_kv_pairs for both mixers, print table + plot
```

## Next (later months, not now)

- M2: swap the toy linear-attention loop for real baselines via `flash-linear-attention`
  (Mamba-2, DeltaNet, Gated DeltaNet), and add the **SSA-style sparse-attention** mixer.
- M3+: recall-vs-length and recall-vs-compute frontiers. See the full plan in the KnowledgeBank
  wiki: `wiki/analyses/ssa-recall-research-6month-plan.md`.
