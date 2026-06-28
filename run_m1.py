"""M1 sweep: recall accuracy vs num_kv_pairs, for attention vs linear attention.

    python run_m1.py --smoke   # quick wiring check (~1 min on CPU)
    python run_m1.py           # the real sweep
"""
import argparse
import os
import torch

from src.train import train_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to verify wiring")
    args = ap.parse_args()

    if args.smoke:
        pairs = [4, 16]
        steps, n_train = 80, 1000
    else:
        pairs = [4, 8, 16, 32, 64]
        steps, n_train = 2000, 12000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  steps={steps}  pairs={pairs}\n")

    results = {"attention": [], "linear": []}
    for mixer in ("attention", "linear"):
        for n in pairs:
            acc = train_one(mixer, num_kv_pairs=n, steps=steps, n_train=n_train,
                            device=device)
            results[mixer].append(acc)
            print(f"{mixer:10s}  N={n:3d}  recall_acc={acc:.3f}")

    print("\n  num_kv_pairs | attention | linear")
    print("  -------------+-----------+-------")
    for i, n in enumerate(pairs):
        print(f"  {n:12d} | {results['attention'][i]:9.3f} | {results['linear'][i]:6.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("results", exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.plot(pairs, results["attention"], "o-", label="attention (ceiling)")
        plt.plot(pairs, results["linear"], "s-", label="linear (fixed state)")
        plt.xlabel("num_kv_pairs")
        plt.ylabel("recall accuracy")
        plt.title("MQAR recall gap (M1)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/recall_gap.png", dpi=120)
        print("\nsaved results/recall_gap.png")
    except ImportError:
        print("\n(matplotlib not installed — skipped plot)")


if __name__ == "__main__":
    main()
