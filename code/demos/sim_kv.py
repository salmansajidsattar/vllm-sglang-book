"""Sim 1: KV cache memory. Preallocated (max length) vs paged (blocks).
Runs on CPU. Saves kv_waste.png (in docs/assets if that folder exists, else here)
"""
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_LEN = 2048   # tokens reserved per request in the naive way
BLOCK = 16       # tokens per block in the paged way


def make_requests(n, seed=0):
    random.seed(seed)
    # actual length (prompt + output) of each request, mostly short
    return [min(MAX_LEN, int(random.lognormvariate(5.5, 0.8))) for _ in range(n)]


def naive_slots(lengths):
    return len(lengths) * MAX_LEN


def paged_slots(lengths):
    blocks = sum((l + BLOCK - 1) // BLOCK for l in lengths)
    return blocks * BLOCK


def main():
    counts = [8, 16, 32, 64, 128]
    used, naive, paged = [], [], []
    for n in counts:
        lens = make_requests(n)
        used.append(sum(lens))
        naive.append(naive_slots(lens))
        paged.append(paged_slots(lens))
        print(f"n={n:4d} used={used[-1]:7d} naive={naive[-1]:7d} "
              f"({100*used[-1]/naive[-1]:.0f}% used)  paged={paged[-1]:7d} "
              f"({100*used[-1]/paged[-1]:.0f}% used)")

    x = range(len(counts))
    w = 0.28
    plt.figure(figsize=(7, 4))
    plt.bar([i - w for i in x], used, w, label="tokens actually needed")
    plt.bar(list(x), paged, w, label="paged (blocks of 16)")
    plt.bar([i + w for i in x], naive, w, label="preallocated (max 2048)", hatch="//")
    plt.xticks(list(x), counts)
    plt.xlabel("concurrent requests")
    plt.ylabel("KV cache slots (tokens)")
    plt.title("KV cache memory: preallocated vs paged")
    plt.legend()
    plt.tight_layout()
    out = ("../../docs/assets/" if os.path.isdir("../../docs/assets") else "") + "kv_waste.png"  # book folder, or here
    plt.savefig(out, dpi=600)
    print("saved:", os.path.abspath(out))


if __name__ == "__main__":
    main()
