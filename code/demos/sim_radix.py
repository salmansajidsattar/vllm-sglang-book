"""Sim 3: prefix cache (radix tree) on a multi-turn chat workload.
A tiny trie of token ids. We count how many prompt tokens are
found in the cache (hits) versus must be computed (misses).
Each user starts with a 200 token system prompt, then adds a 40 token
message each turn. The model reply (60 tokens) is put in the tree too,
so the next turn can reuse it.
Runs on CPU. Saves prefix_hits.png (in docs/assets if that folder exists, else here)
"""
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYSTEM_PROMPT = list(range(1000, 1200))  # 200 shared tokens


def new_node():
    return {}


def match_and_insert(root, tokens):
    """Walk the trie. Return how many leading tokens were already cached."""
    node, hit, matching = root, 0, True
    for t in tokens:
        if t in node:
            if matching:
                hit += 1
        else:
            matching = False
            node[t] = new_node()
        node = node[t]
    return hit


def run(use_cache, n_users=20, n_turns=6, seed=2):
    random.seed(seed)
    root = new_node()
    histories = [[] for _ in range(n_users)]
    hits, total, curve, per_turn = 0, 0, [], []
    for turn in range(n_turns):
        t_hits, t_total = 0, 0
        for u in range(n_users):
            # a new user message is appended (40 tokens)
            histories[u] += [random.randint(2000, 9000) for _ in range(40)]
            prompt = SYSTEM_PROMPT + histories[u]
            h = match_and_insert(root, prompt) if use_cache else 0
            t_hits += h
            t_total += len(prompt)
            # the model reply (60 tokens) is added, and stored in the tree
            histories[u] += [random.randint(2000, 9000) for _ in range(60)]
            if use_cache:
                match_and_insert(root, SYSTEM_PROMPT + histories[u])
        hits += t_hits
        total += t_total
        per_turn.append(t_hits / t_total)  # hit rate in this turn only
        curve.append(hits / total)  # cumulative hit rate up to this turn
    return hits, total, curve, per_turn


def main():
    h, t, with_cache, per_turn = run(True)
    _, _, no_cache, _ = run(False)
    print(f"with prefix cache: {h}/{t} prompt tokens reused ({h/t:.0%})")
    print("hit rate per turn :", " ".join(f"{x:.0%}" for x in per_turn))
    print("cumulative rate   :", " ".join(f"{x:.0%}" for x in with_cache))
    print("without cache: 0% reused")

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, 7), [c * 100 for c in with_cache], marker="o", label="prefix cache")
    plt.plot(range(1, 7), [c * 100 for c in no_cache], marker="o", label="no cache")
    plt.xlabel("conversation turn")
    plt.ylabel("cumulative prompt tokens reused (%)")
    plt.title("Cumulative prefix reuse, 20-user chat workload")
    plt.legend()
    plt.tight_layout()
    out = ("../../docs/assets/" if os.path.isdir("../../docs/assets") else "") + "prefix_hits.png"  # book folder, or here
    plt.savefig(out, dpi=600)
    print("saved:", os.path.abspath(out))


if __name__ == "__main__":
    main()
