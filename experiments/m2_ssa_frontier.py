"""M2a — add the content-based sparse-attention (NSA) curve to the M1 recall frontier.

Same MQAR setting as M1 (seq 128, 8 KV pairs, vocab 8192), same d_model sweep, plus a third
architecture: zoology's DeepSeek Native Sparse Attention — the closest public reference to SubQ's
SSA (content-dependent, block-sparse, learned top-k selection). We measure whether it tracks the
attention ceiling or falls to the base_conv floor.

NOTE: block sizes must divide / fit seq_len=128. SMOKE-TEST ONE NSA RUN before the full sweep:
  python -m zoology.launch experiments/m2_ssa_frontier.py   # will error fast if NSA kwargs mismatch
If NSA errors on a kwarg, read zoology/mixers/deepseek_nsa.py SparseAttention.__init__ and adjust.

Run:  python -m zoology.launch /content/ssa-recall/experiments/m2_ssa_frontier.py
"""
import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, ModuleConfig
from zoology.data.multiquery_ar import MQARConfig

input_seq_len = 128
num_kv_pairs = 8
vocab_size = 8192

train_data = [MQARConfig(num_examples=20_000, vocab_size=vocab_size,
                         input_seq_len=input_seq_len, num_kv_pairs=num_kv_pairs)]
test_data = [MQARConfig(num_examples=2_000, vocab_size=vocab_size,
                        input_seq_len=input_seq_len, num_kv_pairs=num_kv_pairs)]


def mixer(arch):
    if arch == "attention":
        return ModuleConfig(name="zoology.mixers.attention.MHA",
                            kwargs={"dropout": 0.1, "num_heads": 1})
    if arch == "base_conv":
        return ModuleConfig(name="zoology.mixers.base_conv.BaseConv",
                            kwargs={"l_max": input_seq_len, "kernel_size": [3, -1, 3, -1]})
    if arch == "nsa":
        # d_model is injected by zoology; provide the rest. Block sizes fit seq_len=128.
        return ModuleConfig(name="zoology.mixers.deepseek_nsa.SparseAttention",
                            kwargs={"num_heads": 4,
                                    "sliding_window_size": 32,
                                    "compress_block_size": 16,
                                    "selection_block_size": 16,
                                    "num_selected_blocks": 4,
                                    "causal": True})
    raise ValueError(arch)


configs = []
for d_model in [64, 128, 256]:
    for arch in ["attention", "base_conv", "nsa"]:
        for lr in np.logspace(-3.5, -2, 3):
            configs.append(TrainConfig(
                data=DataConfig(
                    train_configs=train_data,
                    test_configs=test_data,
                    batch_size=256,
                    seed=123,
                    cache_dir="/content/zoology_cache",
                ),
                model=ModelConfig(
                    vocab_size=vocab_size,
                    max_position_embeddings=input_seq_len,
                    d_model=d_model,
                    n_layers=2,
                    sequence_mixer=mixer(arch),
                ),
                max_epochs=64,
                learning_rate=float(lr),
                weight_decay=0.1,
                early_stopping_metric="valid/accuracy",
                early_stopping_threshold=0.99,
                run_id=f"{arch}-d{d_model}-lr{lr:.1e}",
            ))
