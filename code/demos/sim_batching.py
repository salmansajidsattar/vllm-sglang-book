"""Sim 2: static batching vs continuous batching.
Each decode step makes 1 token for every running request.
Runs on CPU. Saves batching.png (in docs/assets if that folder exists, else here)
"""
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLOTS = 8  # max requests running at once


def make_lengths(n=32, seed=1):
    random.seed(seed)
    return [random.randint(5, 120) for _ in range(n)]  # output tokens per request


def static_batching(lengths):
    """Take SLOTS requests, run until the slowest finishes, then next batch."""
    steps, busy = 0, []
    for i in range(0, len(lengths), SLOTS):
        batch = lengths[i:i + SLOTS]
        for t in range(max(batch)):
            busy.append(sum(1 for l in batch if l > t))
        steps += max(batch)
    return steps, busy


def continuous_batching(lengths):
    """Refill a free slot at the very next step."""
    waiting = list(lengths)
    running = []
    steps, busy = 0, []
    while waiting or running:
        while waiting and len(running) < SLOTS:
            running.append(waiting.pop(0))
        busy.append(len(running))
        running = [r - 1 for r in running if r - 1 > 0]
        steps += 1
    return steps, busy


def main():
    lengths = make_lengths()
    s_steps, s_busy = static_batching(lengths)
    c_steps, c_busy = continuous_batching(lengths)
    print(f"static:     {s_steps} steps, GPU slot use {sum(s_busy)/(s_steps*SLOTS):.0%}")
    print(f"continuous: {c_steps} steps, GPU slot use {sum(c_busy)/(c_steps*SLOTS):.0%}")

    plt.figure(figsize=(8, 4))
    plt.plot(s_busy, label=f"static ({s_steps} steps)")
    plt.plot(c_busy, "--", label=f"continuous ({c_steps} steps)")
    plt.ylim(0, SLOTS + 1)
    plt.xlabel("decode step")
    plt.ylabel("requests running")
    plt.title(f"Static vs continuous batching ({SLOTS} slots, {len(lengths)} requests)")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    plt.tight_layout()
    out = ("../../docs/assets/" if os.path.isdir("../../docs/assets") else "") + "batching.png"  # book folder, or here
    plt.savefig(out, dpi=600, bbox_inches="tight")
    print("saved:", os.path.abspath(out))


if __name__ == "__main__":
    main()
