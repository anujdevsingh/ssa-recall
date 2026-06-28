"""Build the M1 frontier figure + table from the measured sweep.

These are the best valid/accuracy values (over the 3-LR sweep) read from the Zoology
sweep run on 2026-06-28 (Colab T4). The runtime died during the final base_conv-d512
run, so d512 base_conv is from a partial run (plateaued ~0.13) and attention-d512 did
NOT converge within the 48-epoch cap (an optimization artifact, not a capability limit).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# best valid/accuracy over the LR sweep, per (arch, d_model)
ATTENTION = {64: 0.998, 128: 0.996, 256: 0.996, 512: 0.145}  # d512 did NOT converge (48ep cap)
BASE_CONV = {64: 0.097, 128: 0.127, 256: 0.131, 512: 0.128}  # d512 partial (runtime died ep15)
CLEAN = [64, 128, 256]   # sizes where attention cleanly converged

here = os.path.dirname(__file__)

plt.figure(figsize=(6.2, 4.2))
dims = [64, 128, 256, 512]
# clean (converged) segment, solid
plt.plot(CLEAN, [ATTENTION[d] for d in CLEAN], "o-", color="#1f77b4", label="attention (converged)")
plt.plot(dims, [BASE_CONV[d] for d in dims], "s-", color="#d62728", label="base_conv (gated conv)")
# d512 attention: non-converged, hollow marker + annotation
plt.plot([512], [ATTENTION[512]], "o", mfc="white", mec="#1f77b4", label="attention d512 (did not converge, 48ep)")
plt.annotate("did not converge\n(48-epoch cap)", xy=(512, ATTENTION[512]), xytext=(300, 0.32),
             fontsize=8, color="#1f77b4",
             arrowprops=dict(arrowstyle="->", color="#1f77b4"))

plt.xscale("log", base=2)
plt.xticks(dims, [str(d) for d in dims])
plt.xlabel("d_model  (state size)")
plt.ylabel("MQAR recall accuracy")
plt.title("M1: recall vs. state size  (MQAR: seq 128, 8 KV pairs, vocab 8192)")
plt.ylim(0, 1.02)
plt.grid(alpha=0.3)
plt.legend(fontsize=8, loc="center right")
plt.tight_layout()
out = os.path.join(here, "recall_frontier.png")
plt.savefig(out, dpi=120)
print("saved", out)

# table
print("\n  d_model | attention | base_conv")
print("  --------+-----------+----------")
for d in dims:
    star = "  (*non-converged)" if d == 512 else ""
    print(f"  {d:7d} | {ATTENTION[d]:9.3f} | {BASE_CONV[d]:8.3f}{star}")
