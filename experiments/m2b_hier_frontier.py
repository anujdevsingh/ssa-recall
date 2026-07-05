"""M2b — the cheap selector: recall AND selection-cost vs SEQUENCE LENGTH.

M2a proved content-based sparse attention (NSA) preserves MQAR recall but its selector is
quadratic. M2b adds `hier` (src/hier_nsa.HierSparseAttention) — same 3-branch mechanism, but the
coarse stage is a 2-level tree so selection is O(n^1.5), not O(n^2/B).

Seq-128 only proves recall is preserved (C1). The CONTRIBUTION (C2: sub-quadratic cost) needs the
LENGTH AXIS, so this sweeps input_seq_len in {128, 512, 2048, 8192}. d_model is fixed (128 — where
attention/nsa/hier all rode the ceiling in M2a) to isolate length. num_kv_pairs scales with length
so recall stays a real test at every length.

g (superblock_group) is set to round(sqrt(nblk)) PER LENGTH — this is what makes hier sub-quadratic;
a fixed g is O(n^2) and defeats the whole point (see hier_nsa.py docstring).

Cost metric = coarse scored-pairs/query (printed below per length): nsa = nblk ; hier = nsup + s1*g.
That divergence as length grows IS the paper. (Wall-clock is indicative only — the prototype scores
via gather+mask, not a fused sparse kernel.)

SETUP on Kaggle/Colab (hier_nsa must be importable by zoology):
  export PYTHONPATH=/content/ssa-recall/src:$PYTHONPATH      # so "hier_nsa.HierSparseAttention" resolves
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True    # M2a T4 OOM fix
GATES before trusting a sweep:
  1) python /content/ssa-recall/src/hier_nsa.py              # self-test (shape/causality/grad)
  2) one hier run at seq128 must match NSA recall (~0.99) before reading the length curve
Run: python -m zoology.launch /content/ssa-recall/experiments/m2b_hier_frontier.py
"""
import math
import os

# New env name (PYTORCH_CUDA_ALLOC_CONF is the deprecated one); inherited by per-run subprocesses.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, ModuleConfig
from zoology.data.multiquery_ar import MQARConfig

vocab_size = 8192
d_model = 128
CB = 16                       # compress == selection block size (hier leaf)
S1 = 2                        # top super-blocks visited
K_SEL = 4                     # fine blocks attended exactly (matches M2a NSA num_selected_blocks)

# length -> (num_kv_pairs, batch_size). kv scales with length; batch shrinks for the T4.
# L8192 dropped from this sweep — dense attention OOMs the T4 (8192^2 score matrix > 100GB).
# That length point will be a hier-only follow-up experiment.
# L512 batch 128->32, L2048 64->16: nsa/hier OOM'd the 14.5GB T4 at the old sizes (repair sweep 2026-07-04).
LENGTHS = {128: (8, 256), 512: (32, 32), 2048: (128, 16)}
# 0.98 (not 0.99) early-stop: NSA converges to ~0.985 here and 98%+ MQAR == "rides the ceiling";
# 0.99 left NSA training 64 full epochs at every length. 40-epoch cap matches M2a's "~45ep" finding.
EARLY_STOP = 0.98
MAX_EPOCHS = int(os.environ.get("EPOCHS", 40))   # raise per-session if long lengths need it


def hier_g(seq_len):
    nblk = math.ceil(seq_len / CB)
    return max(2, round(math.sqrt(nblk)))         # g ~ sqrt(nblk)  =>  O(n^1.5) selection


def mixer(arch, seq_len):
    if arch == "attention":
        return ModuleConfig(name="zoology.mixers.attention.MHA",
                            kwargs={"dropout": 0.1, "num_heads": 4})
    if arch == "nsa":
        return ModuleConfig(name="zoology.mixers.deepseek_nsa.SparseAttention",
                            kwargs={"num_heads": 4, "sliding_window_size": 32,
                                    "compress_block_size": CB, "selection_block_size": CB,
                                    "num_selected_blocks": K_SEL, "causal": True})
    if arch == "hier":
        g = hier_g(seq_len)
        nblk = math.ceil(seq_len / CB)
        nsup = math.ceil(nblk / g)
        print(f"[cost] L={seq_len:>5}  nblk={nblk:>4}  g={g:>3}  "
              f"hier_pairs/q={nsup + S1 * g:>4}  vs  nsa_pairs/q={nblk:>4}  "
              f"({nblk / (nsup + S1 * g):.1f}x cheaper)")
        return ModuleConfig(name="hier_nsa.HierSparseAttention",
                            kwargs={"num_heads": 4, "sliding_window_size": 32,
                                    "compress_block_size": CB, "selection_block_size": CB,
                                    "num_selected_blocks": K_SEL, "superblock_group": g,
                                    "num_selected_superblocks": S1, "causal": True})
    raise ValueError(arch)


configs = []
for seq_len, (num_kv_pairs, batch_size) in LENGTHS.items():
    train_data = [MQARConfig(num_examples=20_000, vocab_size=vocab_size,
                             input_seq_len=seq_len, num_kv_pairs=num_kv_pairs)]
    test_data = [MQARConfig(num_examples=2_000, vocab_size=vocab_size,
                            input_seq_len=seq_len, num_kv_pairs=num_kv_pairs)]
    for arch in ["attention", "nsa", "hier"]:
        for lr in np.logspace(-3.5, -2, 3):
            configs.append(TrainConfig(
                data=DataConfig(train_configs=train_data, test_configs=test_data,
                                batch_size=batch_size, seed=123, cache_dir="/content/zoology_cache"),
                model=ModelConfig(vocab_size=vocab_size, max_position_embeddings=seq_len,
                                  d_model=d_model, n_layers=2, sequence_mixer=mixer(arch, seq_len)),
                max_epochs=MAX_EPOCHS, learning_rate=float(lr), weight_decay=0.1,
                early_stopping_metric="valid/accuracy", early_stopping_threshold=EARLY_STOP,
                run_id=f"{arch}-L{seq_len}-lr{lr:.1e}",
            ))

# Parallel-GPU sharding: launch N processes, each with SHARD=i NSHARDS=N
# CUDA_VISIBLE_DEVICES=i, and they round-robin-split the config list with no overlap.
# Default (no env) runs all configs on one GPU as before.
_S = int(os.environ.get("SHARD", 0))
_N = int(os.environ.get("NSHARDS", 1))
configs = configs[_S::_N]

# Repair runs: ONLY=hier-L2048,nsa-L2048 re-runs just the configs whose run_id
# starts with one of the given prefixes (applied after sharding).
_ONLY = os.environ.get("ONLY", "")
if _ONLY:
    _keys = _ONLY.split(",")
    configs = [c for c in configs if any(c.run_id.startswith(k) for k in _keys)]
print(f"[shard] {_S}/{_N} only={_ONLY or 'all'} -> running {len(configs)} configs on this process")

# Run directly with `python m2b_hier_frontier.py` (not `python -m zoology.launch ...`).
# Each config runs in a FRESH SUBPROCESS: the 2026-07-04 repair sweep showed ~13GB stays
# allocated after a run finishes in-process (empty_cache doesn't reclaim it), OOMing every
# subsequent nsa/hier config. A child process per run guarantees a clean CUDA context, and a
# crash (OOM or otherwise) only kills that child.
if __name__ == "__main__":
    import subprocess
    import sys
    from datetime import datetime

    if os.environ.get("SUBPROC") == "1":  # child: train the single config selected via ONLY=
        from zoology.train import train
        config = configs[0]
        config.launch_id = os.environ["LAUNCH_ID"]
        train(config)
        sys.exit(0)

    launch_id = f"shard{_S}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    for config in configs:
        # ONLY=<full run_id> selects exactly this config in the child (child re-derives the
        # full list, so SHARD/NSHARDS are reset — the parent already did the sharding).
        env = {**os.environ, "SUBPROC": "1", "ONLY": config.run_id,
               "LAUNCH_ID": launch_id, "SHARD": "0", "NSHARDS": "1"}
        r = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env)
        if r.returncode != 0:
            print(f"[FAIL] {config.run_id} exited {r.returncode} (OOM or crash) — continuing")
