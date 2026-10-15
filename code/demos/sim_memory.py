"""Sim 4: GPU memory budget. How many requests fit?
Edit the numbers at the top for your model and GPU. Saves memory_budget.png (in docs/assets if that folder exists, else here)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- model (Llama-3.1-8B style) ---
PARAMS_B = 8.0
LAYERS = 32
KV_HEADS = 8
HEAD_DIM = 128
# --- GPU ---
GPU_GB = 80
UTIL = 0.90          # gpu_memory_utilization (vLLM) ~ mem_fraction_static (SGLang)
OVERHEAD_GB = 4      # activations, CUDA graphs, etc (rough)


# Rounded estimates: this script counts 1 GB as 1e9 bytes, but GPU memory is
# really in GiB (an "80 GB" card has about 80 GiB), so real numbers differ a little.
def kv_bytes_per_token(dtype_bytes):
    # 2 = keys and values
    return 2 * LAYERS * KV_HEADS * HEAD_DIM * dtype_bytes


def budget(weight_bytes, kv_dtype_bytes, ctx):
    weights_gb = PARAMS_B * weight_bytes
    kv_gb = GPU_GB * UTIL - weights_gb - OVERHEAD_GB
    if kv_gb <= 0:  # weights do not fit on one GPU: no room for any cache
        return weights_gb, 0.0, 0, 0
    per_token = kv_bytes_per_token(kv_dtype_bytes)
    tokens = int(kv_gb * 1e9 / per_token)
    return weights_gb, kv_gb, tokens, tokens // ctx


def main():
    ctx = 4096
    cases = [("BF16 weights, BF16 KV", 2, 2),
             ("FP8 weights, BF16 KV", 1, 2),
             ("FP8 weights, FP8 KV", 1, 1)]
    names, weights, kvs, users = [], [], [], []
    for name, wb, kb in cases:
        w, kv, tokens, n = budget(wb, kb, ctx)
        if kv == 0:
            print(f"{name}: the weights ({w:.0f} GB) do not fit on one GPU. Users: 0.")
            print("  Use more GPUs (tensor parallel) or a smaller model.")
            names.append(name); weights.append(min(w, GPU_GB)); kvs.append(0); users.append(0)
            continue
        print(f"{name:24s} weights={w:5.1f}GB  kv={kv:5.1f}GB  "
              f"tokens={tokens:8d}  users@{ctx}={n}")
        names.append(name)
        weights.append(w)
        kvs.append(kv)
        users.append(n)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.barh(names, weights, label="weights")
    ax.barh(names, kvs, left=weights, label="KV cache")
    for i, n in enumerate(users):
        ax.text(weights[i] + kvs[i] + 1, i, f"{n} users", va="center")
    ax.set_xlabel(f"GB on a {GPU_GB} GB GPU")
    ax.set_title(f"Memory budget, {PARAMS_B:.0f}B model, {ctx} tokens per user")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    ax.set_xlim(0, GPU_GB + 15)
    plt.tight_layout()
    out = ("../../docs/assets/" if os.path.isdir("../../docs/assets") else "") + "memory_budget.png"  # book folder, or here
    plt.savefig(out, dpi=600)
    print("saved:", os.path.abspath(out))


if __name__ == "__main__":
    main()
