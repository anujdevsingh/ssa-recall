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
import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, ModuleConfig
from zoology.data.multiquery_ar import MQARConfig

vocab_size = 8192
d_model = 128
CB = 16                       # compress == selection block size (hier leaf)
S1 = 2                        # top super-blocks visited
K_SEL = 4                     # fine blocks attended exactly (matches M2a NSA num_selected_blocks)

# length -> (num_kv_pairs, batch_size). kv scales with length; batch shrinks for the T4.
LENGTHS = {128: (8, 256), 512: (32, 256), 2048: (128, 64), 8192: (512, 16)}


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
                max_epochs=64, learning_rate=float(lr), weight_decay=0.1,
                early_stopping_metric="valid/accuracy", early_stopping_threshold=0.99,
                run_id=f"{arch}-L{seq_len}-lr{lr:.1e}",
            ))
