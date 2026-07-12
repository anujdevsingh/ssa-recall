"""Build the M2b frontier figure + table from the measured length sweep.

Best valid/accuracy per (mixer, length), parsed from the raw sweep logs:
  L128  — Kaggle 2xT4 2026-07-03 (commit 74ecfe7), best over 3 LRs
  L512  — Kaggle T4 2026-07-04/05 (commits e2d484d, 38d63c7), lr 3.2e-4
          (the only LR that converged for any mixer at L512+)
  L2048 — Brev L40S 48GB 2026-07-10 (commit c677c95), lr 3.2e-4
Caveats live in RESULTS.md: nsa-L2048 ran batch 8 (memory-forced) and was
truncated at ep45/70 after a 6-epoch plateau; everything else batch-16 recipe.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LENGTHS = [128, 512, 2048]
# best valid/accuracy
ATTENTION = {128: 0.991, 512: 0.988, 2048: 1.000}
HIER      = {128: 0.995, 512: 0.983, 2048: 0.834}
NSA       = {128: 0.981, 512: 0.978, 2048: 0.771}
# selector-stage scored pairs per query (printed by the harness at launch)
COST_HIER = {128: 9,  512: 18,  2048: 34}
COST_NSA  = {128: 8,  512: 32,  2048: 128}
COST_DENSE = {L: L / 2 for L in LENGTHS}  # avg causal pairs/query, reference

here = os.path.dirname(__file__)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

# --- panel 1: recall vs length ---
ax1.plot(LENGTHS, [ATTENTION[L] for L in LENGTHS], "o-", color="#1f77b4", label="attention (dense)")
ax1.plot(LENGTHS, [HIER[L] for L in LENGTHS], "^-", color="#2ca02c", label="hier (sub-quadratic selector, ours)")
ax1.plot(LENGTHS, [NSA[L] for L in LENGTHS], "s-", color="#d62728", label="nsa (quadratic-coarse selector)")
ax1.annotate("0.834", xy=(2048, 0.834), xytext=(-38, 6), textcoords="offset points",
             fontsize=8, color="#2ca02c")
ax1.annotate("0.771*", xy=(2048, 0.771), xytext=(-42, -14), textcoords="offset points",
             fontsize=8, color="#d62728")
ax1.set_xscale("log", base=2)
ax1.set_xticks(LENGTHS, [str(L) for L in LENGTHS])
ax1.set_xlabel("sequence length  (KV pairs scale with it: 8 / 32 / 128)")
ax1.set_ylabel("MQAR recall accuracy (best)")
ax1.set_title("M2b: recall vs. length")
ax1.set_ylim(0, 1.05)
ax1.grid(alpha=0.3)
ax1.legend(fontsize=8, loc="lower left")

# --- panel 2: selector cost vs length ---
ax2.plot(LENGTHS, [COST_DENSE[L] for L in LENGTHS], "--", color="#888888", label="dense attention (all past keys)")
ax2.plot(LENGTHS, [COST_NSA[L] for L in LENGTHS], "s-", color="#d62728", label="nsa coarse stage: O(n·n/cb)")
ax2.plot(LENGTHS, [COST_HIER[L] for L in LENGTHS], "^-", color="#2ca02c", label="hier 2-level tree: O(n^1.5)")
ax2.annotate("3.8x cheaper", xy=(2048, 34), xytext=(700, 60), fontsize=9, color="#2ca02c",
             arrowprops=dict(arrowstyle="->", color="#2ca02c"))
ax2.set_xscale("log", base=2)
ax2.set_yscale("log", base=2)
ax2.set_xticks(LENGTHS, [str(L) for L in LENGTHS])
ax2.set_xlabel("sequence length")
ax2.set_ylabel("selector scored-pairs / query")
ax2.set_title("M2b: selection cost vs. length")
ax2.grid(alpha=0.3, which="both")
ax2.legend(fontsize=8, loc="upper left")

fig.suptitle("MQAR, vocab 8192, d_model 128, 2 layers   (*nsa-L2048: batch 8, truncated ep45 after plateau)",
             y=1.00, fontsize=9)
fig.tight_layout()
out = os.path.join(here, "m2b_frontier.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print("saved", out)

print("\n  length | attention |   hier |    nsa | hier/nsa selector cost")
print("  -------+-----------+--------+--------+-----------------------")
for L in LENGTHS:
    print(f"  {L:6d} | {ATTENTION[L]:9.3f} | {HIER[L]:6.3f} | {NSA[L]:6.3f} |"
          f" {COST_HIER[L]:3d} vs {COST_NSA[L]:3d}  ({COST_NSA[L]/COST_HIER[L]:.1f}x)")
