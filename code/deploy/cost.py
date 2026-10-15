"""Cost per million output tokens.
Usage: python cost.py GPU_PRICE_PER_HOUR TOTAL_TOKENS_PER_SECOND [NUM_GPUS]
TOTAL_TOKENS_PER_SECOND is the speed of ALL the GPUs together, not per GPU.
Examples:
  python cost.py 1.20 2400        # one GPU, 2400 tok/s
  python cost.py 1.20 4800 2      # two GPUs, 4800 tok/s in total
"""
import sys

price = float(sys.argv[1])
tok_s = float(sys.argv[2])
gpus = int(sys.argv[3]) if len(sys.argv) > 3 else 1

per_hour = price * gpus
tokens_per_hour = tok_s * 3600
cost = per_hour / tokens_per_hour * 1_000_000
print(f"GPU cost per hour : ${per_hour:.2f}")
print(f"tokens per hour   : {tokens_per_hour:,.0f}")
print(f"cost per 1M tokens: ${cost:.3f}")
