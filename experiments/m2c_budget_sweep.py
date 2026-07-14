"""M2c — selector-budget sweep at L2048: does MORE budget buy back the recall lost to sparsity?

M2b's headline: at L2048 hier (34 pairs/q) beats nsa (128 pairs/q) on recall (0.834 vs 0.771),
but both trail dense attention (1.000). Claimed: "recall lost at L2048 is lost to block-sparsity
itself, not the selector." THIS experiment tests that claim before we write it down.

All runs are hier at L2048, lr 3.2e-4, same data/seed as M2b. Two knobs move:
  S1    = super-blocks visited  -> coarse selector budget (pairs/q = nsup + S1*g)
  K_SEL = fine blocks attended  -> sparsity budget (exact-attention tokens = K_SEL*CB + window)

At L2048: nblk=128, g=11, nsup=12. The 5 points:
  run_id               S1  K_SEL  coarse pairs/q   what it isolates
  hier-L2048-s2-k4      2      4        34         M2b baseline re-anchored at batch 8 (control)
  hier-L2048-s4-k4      4      4        56         selector budget 1.6x, sparsity fixed
  hier-L2048-s12-k4    12      4       144         selector SATURATED (>= nsa's 128), sparsity fixed
  hier-L2048-s4-k8      4      8        56         sparsity budget 2x, selector modest
  hier-L2048-s12-k16   12     16       144         everything maxed: the hard-ceiling test

Readout: recall flat ~0.83 across all -> block-sparse attention has a budget-independent ceiling
here (claim survives, strongest version). k8/k16 recover toward 1.0 -> recall was sparsity-budget
bound (claim holds: selector wasn't the problem). s12-k4 recovers alone -> selector-bound (claim
WRONG — must know now, not post-writeup).

BATCH 8 (not M2b's 16): T4 does ~1.8x more updates/sec at batch 8 on this workload, so the
~112k-update plateau horizon fits Kaggle's 12h wall (~50ep of 2500 steps ~= 8-10h). Same trick
that made nsa-L2048 feasible; s2-k4 doubles as the batch-robustness check of M2b's 0.834.

Run (Kaggle): notebooks/kaggle_m2c_budget.ipynb. Env: ONLY=/EPOCHS=/SHARD/NSHARDS as in m2b.
PYTHONPATH must include src/ so "hier_nsa.HierSparseAttention" resolves.
"""
import math
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from zoology.config import TrainConfig, ModelConfig, DataConfig, ModuleConfig
from zoology.data.multiquery_ar import MQARConfig

vocab_size = 8192
d_model = 128
CB = 16
SEQ_LEN = 2048
NUM_KV = 128
BATCH = 8                      # see docstring: batch-8 recipe fits the 12h wall
LR = 3.2e-4                    # the only LR that works at L512+ (M2b repair sweep)
EARLY_STOP = 0.98
MAX_EPOCHS = int(os.environ.get("EPOCHS", 60))   # ~150k updates at batch 8 (plateau needs ~112k)

nblk = math.ceil(SEQ_LEN / CB)                    # 128
g = max(2, round(math.sqrt(nblk)))                # 11 — same per-length rule as m2b
nsup = math.ceil(nblk / g)                        # 12

BUDGETS = [(2, 4), (4, 4), (12, 4), (4, 8), (12, 16)]   # (S1, K_SEL)

train_data = [MQARConfig(num_examples=20_000, vocab_size=vocab_size,
                         input_seq_len=SEQ_LEN, num_kv_pairs=NUM_KV)]
test_data = [MQARConfig(num_examples=2_000, vocab_size=vocab_size,
                        input_seq_len=SEQ_LEN, num_kv_pairs=NUM_KV)]

configs = []
for s1, k_sel in BUDGETS:
    pairs = nsup + s1 * g
    print(f"[cost] s1={s1:>2} k_sel={k_sel:>2}  coarse_pairs/q={pairs:>4}  "
          f"(nsa=128)  exact_tokens={k_sel * CB + 32}")
    mixer = ModuleConfig(name="hier_nsa.HierSparseAttention",
                         kwargs={"num_heads": 4, "sliding_window_size": 32,
                                 "compress_block_size": CB, "selection_block_size": CB,
                                 "num_selected_blocks": k_sel, "superblock_group": g,
                                 "num_selected_superblocks": s1, "causal": True})
    configs.append(TrainConfig(
        data=DataConfig(train_configs=train_data, test_configs=test_data,
                        batch_size=BATCH, seed=123, cache_dir="/content/zoology_cache"),
        model=ModelConfig(vocab_size=vocab_size, max_position_embeddings=SEQ_LEN,
                          d_model=d_model, n_layers=2, sequence_mixer=mixer),
        max_epochs=MAX_EPOCHS, learning_rate=LR, weight_decay=0.1,
        early_stopping_metric="valid/accuracy", early_stopping_threshold=EARLY_STOP,
        run_id=f"hier-L{SEQ_LEN}-s{s1}-k{k_sel}",
    ))

_S = int(os.environ.get("SHARD", 0))
_N = int(os.environ.get("NSHARDS", 1))
configs = configs[_S::_N]

_ONLY = os.environ.get("ONLY", "")
if _ONLY:
    _keys = _ONLY.split(",")
    configs = [c for c in configs if any(c.run_id.startswith(k) for k in _keys)]
print(f"[shard] {_S}/{_N} only={_ONLY or 'all'} -> running {len(configs)} configs on this process")

# Fresh subprocess per config (same rationale as m2b: in-process reruns leak ~13GB CUDA memory).
if __name__ == "__main__":
    import subprocess
    import sys
    from datetime import datetime

    if os.environ.get("SUBPROC") == "1":
        from zoology.train import train
        config = configs[0]
        config.launch_id = os.environ["LAUNCH_ID"]
        train(config)
        sys.exit(0)

    launch_id = f"m2c-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    for config in configs:
        env = {**os.environ, "SUBPROC": "1", "ONLY": config.run_id,
               "LAUNCH_ID": launch_id, "SHARD": "0", "NSHARDS": "1"}
        r = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env)
        if r.returncode != 0:
            print(f"[FAIL] {config.run_id} exited {r.returncode} (OOM or crash) — continuing")
