"""Plot results.csv from run_sweep.sh: one small chart per measure, one line per engine.
Usage: python plot_sweep.py results.csv
Columns are read by name. A chart is skipped if its column is not in the file.
A level with failed requests is skipped, because it is not a fair point on the curve.
"""
import csv
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (column name, chart title, y label)
CHARTS = [
    ("tok_s", "Throughput (higher is better)", "output tokens/s"),
    ("ttft_p99_ms", "TTFT p99 (lower is better)", "ms"),
    ("tpot_ms", "TPOT mean (lower is better)", "ms per token"),
]


def load(path):
    """Return {engine: [row, ...]} with rows sorted by concurrency."""
    data = {}
    for row in csv.DictReader(open(path)):
        if int(row.get("failed") or 0) > 0:  # a level with failed requests is not a fair point
            print(f"skipping {row['engine']} at concurrency {row['concurrency']}: {row['failed']} requests failed")
            continue
        data.setdefault(row["engine"], []).append(row)
    for rows in data.values():
        rows.sort(key=lambda r: int(r["concurrency"]))
    return data


def main(path):
    data = load(path)
    first = next(iter(data.values()))[0]
    charts = [c for c in CHARTS if c[0] in first]
    fig, axes = plt.subplots(1, len(charts), figsize=(5 * len(charts), 4))
    if len(charts) == 1:
        axes = [axes]
    for ax, (col, title, ylabel) in zip(axes, charts):
        for engine, rows in data.items():
            xs = [int(r["concurrency"]) for r in rows]
            ys = [float(r[col]) for r in rows]
            ax.plot(xs, ys, marker="o", label=engine)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("concurrency")
        ax.set_xscale("log", base=2)
        ax.legend()
    plt.tight_layout()
    plt.savefig("sweep.png", dpi=150)
    print("wrote sweep.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.csv")
