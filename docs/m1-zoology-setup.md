# M1 — Zoology setup on Colab (staged)

Goal of this stage: get the **Zoology** harness installed and prove it trains on MQAR, before we
build any sweep. Install is the risky part (some CUDA kernels don't build on Colab), so we stage it.

> Runtime → Change runtime type → **GPU** (T4 is fine).

## Stage 1 — clone + minimal install + prove it runs

```python
# clone
!git clone https://github.com/HazyResearch/zoology.git
%cd zoology

# minimal install (skips the problematic mamba_ssm / causal-conv1d kernels on purpose)
!pip install -e . -q

# avoid wandb login friction for now
import os
os.environ["WANDB_MODE"] = "offline"

# known-good single run: confirms the harness trains on MQAR with attention (MHA)
!python -m zoology.launch zoology/experiments/basic_examples/basic.py
```

If that run trains and reports a test accuracy, the environment is good. **Stop here and report the
number** before going further.

> Note: `mamba_ssm` / `causal-conv1d` are intentionally NOT installed — they often fail to build on
> Colab and we don't need them. Our efficient/sparse baselines (GLA, Based, DeltaNet, NSA) come from
> `fla` (next stage), which is pure-Triton and works on T4.

## Stage 2 — add the efficient + sparse mixers

```python
# flash-linear-attention provides GLA / DeltaNet / Gated DeltaNet kernels (needs Python 3.10+, T4 ok)
!pip install flash-linear-attention -q
# analysis extras for the plotting helper
!pip install -e ".[analysis]" -q
```

Sanity-check which mixer module paths exist in this checkout (so our config uses real names):

```python
!ls zoology/mixers
!sed -n '1,40p' zoology/experiments/basic_examples/basic_sweep.py   # template we copy from
```

The mixers we want, by module path (confirm against the `ls` above):
- `zoology.mixers.attention.MHA`            — attention ceiling
- `zoology.mixers.based.Based` (or `gla`)   — fixed-state efficient baseline
- DeepSeek **native sparse attention**       — the SSA-family reference (find its path in `mixers/`)

## Stage 3 — reduced d_model sweep (the M1 figure)

We will write `experiments/m1_recall_frontier.py` by adapting `basic_sweep.py`: keep MQAR fixed
(`input_seq_len`, `num_kv_pairs`, `vocab_size`), and instead of sweeping learning rate, sweep
`d_model ∈ {64,128,256,512}` × architecture ∈ {MHA, Based/GLA, NSA}, with a small LR sweep per
point so nothing is handicapped by optimization. Run:

```python
!python -m zoology.launch experiments/m1_recall_frontier.py
```

Then plot accuracy vs `d_model` (one curve per architecture) using
`zoology/analysis/mqar_plotting_example.py` as the template. That plot is **M1 done**.

> We write the Stage-3 config together once Stage 1 proves install works and Stage 2 confirms the
> real mixer module names in your checkout — so the config uses verified paths, not guesses.
