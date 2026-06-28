"""Parse a zoology.launch stdout log into the recall-vs-state frontier.

For each run we take the best valid/accuracy reached; for each (arch, d_model) we take the best
over the LR sweep. Prints a table and saves results/m1_frontier.png.

Usage:  python experiments/parse_frontier.py results_m1.log
"""
import collections
import re
import sys

RID = re.compile(r"run_id='([^']+)'")
ACC = re.compile(r"valid/accuracy[=\s]+([0-9.]+)")


def parse(path):
    best = {}          # run_id -> best valid acc
    cur = None
    for line in open(path, errors="ignore"):
        m = RID.search(line)
        if m:
            cur = m.group(1)
            best.setdefault(cur, 0.0)
        if cur is not None:
            for tok in ACC.findall(line):
                best[cur] = max(best[cur], float(tok.rstrip(".")))
    return best


def frontier(best):
    out = collections.defaultdict(float)   # (arch, d_model) -> best over lr
    for rid, acc in best.items():
        if "-d" not in rid or "-lr" not in rid:
            continue
        arch, rest = rid.split("-d", 1)
        d_model = int(rest.split("-lr")[0])
        out[(arch, d_model)] = max(out[(arch, d_model)], acc)
    return out


def main(path):
    fr = frontier(parse(path))
    archs = sorted({a for a, _ in fr})
    dims = sorted({d for _, d in fr})

    print(f"\n  recall accuracy vs d_model  (best over LR sweep)\n")
    header = "  d_model " + "".join(f"| {a:>10} " for a in archs)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for d in dims:
        row = f"  {d:7d} " + "".join(f"| {fr.get((a, d), float('nan')):10.3f} " for a in archs)
        print(row)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import os
        os.makedirs("results", exist_ok=True)
        plt.figure(figsize=(6, 4))
        for a in archs:
            ys = [fr.get((a, d), float("nan")) for d in dims]
            plt.plot(dims, ys, "o-", label=a)
        plt.xscale("log", base=2)
        plt.xlabel("d_model (state size)")
        plt.ylabel("MQAR recall accuracy")
        plt.title("M1: recall-vs-state frontier")
        plt.ylim(0, 1.02)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/m1_frontier.png", dpi=120)
        print("\n  saved results/m1_frontier.png")
    except ImportError:
        print("\n  (matplotlib not installed — table only)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_m1.log")
