"""Parse a zoology.launch stdout log into the recall-vs-state frontier."""
import collections, re, sys
RID = re.compile(r"run_id='([^']+)'")
ACC = re.compile(r"valid/accuracy[=\s]+([0-9.]+(?:[eE][-+]?[0-9]+)?)")

def parse(path):
    best, cur = {}, None
    for line in open(path, errors="ignore"):
        m = RID.search(line)
        if m:
            cur = m.group(1); best.setdefault(cur, 0.0)
        if cur is not None:
            for tok in ACC.findall(line):
                v = float(tok.rstrip("."))
                if v <= 1.0:                      # accuracy can't exceed 1
                    best[cur] = max(best[cur], v)
    return best

def frontier(best):
    out = collections.defaultdict(float)
    for rid, acc in best.items():
        if "-d" not in rid or "-lr" not in rid: continue
        arch, rest = rid.split("-d", 1)
        d = int(rest.split("-lr")[0])
        out[(arch, d)] = max(out[(arch, d)], acc)
    return out

def main(path):
    fr = frontier(parse(path))
    archs = sorted({a for a, _ in fr}); dims = sorted({d for _, d in fr})
    print("\n  recall accuracy vs d_model  (best over LR sweep)\n")
    header = "  d_model " + "".join(f"| {a:>10} " for a in archs)
    print(header); print("  " + "-" * (len(header) - 2))
    for d in dims:
        print(f"  {d:7d} " + "".join(f"| {fr.get((a,d), float('nan')):10.3f} " for a in archs))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        os.makedirs("results", exist_ok=True) if (os:=__import__("os")) else None
        plt.figure(figsize=(6,4))
        for a in archs: plt.plot(dims, [fr.get((a,d), float("nan")) for d in dims], "o-", label=a)
        plt.xscale("log", base=2); plt.xlabel("d_model (state size)")
        plt.ylabel("MQAR recall accuracy"); plt.title("M2: recall-vs-state frontier")
        plt.ylim(0,1.02); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig("results/m2_frontier.png", dpi=120); print("\n  saved results/m2_frontier.png")
    except ImportError: print("\n  (matplotlib not installed)")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_m1.log")
