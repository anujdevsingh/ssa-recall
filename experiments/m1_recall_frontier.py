"""M1 — recall-vs-state-size frontier on MQAR (Zoology reproduction).

Independent variable: d_model (state size).  Held fixed: MQAR difficulty + training recipe.
Architectures: MHA (attention ceiling) vs BaseConv (fixed-state efficient baseline).
A small LR sweep per point so neither arch is handicapped by optimization (we take the best
LR per (arch, d_model) when plotting). This is one point of Zoology Figure 2.

Run:  python -m zoology.launch /content/ssa-recall/experiments/m1_recall_frontier.py
"""
import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, ModuleConfig
from zoology.data.multiquery_ar import MQARConfig

# --- fixed MQAR difficulty (a Zoology Fig.2 point) -------------------------------------
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
    raise ValueError(arch)


# --- the sweep: d_model x architecture x small LR sweep --------------------------------
configs = []
for d_model in [64, 128, 256, 512]:
    for arch in ["attention", "base_conv"]:
        for lr in np.logspace(-3.5, -2, 3):   # ~[3.2e-4, 1.8e-3, 1e-2]
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
                max_epochs=48,
                learning_rate=float(lr),
                weight_decay=0.1,
                early_stopping_metric="valid/accuracy",
                early_stopping_threshold=0.99,
                run_id=f"{arch}-d{d_model}-lr{lr:.1e}",
            ))
