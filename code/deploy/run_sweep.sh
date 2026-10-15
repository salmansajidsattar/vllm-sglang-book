#!/usr/bin/env bash
# Run the same benchmark on one engine at several concurrency levels.
# Usage: bash run_sweep.sh ENGINE_NAME BASE_URL MODEL
#   bash run_sweep.sh vllm   http://localhost:8000  Qwen/Qwen2.5-1.5B-Instruct
#   bash run_sweep.sh sglang http://localhost:30000 Qwen/Qwen2.5-1.5B-Instruct
# Extra bench.py flags can go in EXTRA, for example:
#   EXTRA="--unique-prompts --ignore-eos" bash run_sweep.sh vllm http://localhost:8000 MODEL
set -eo pipefail
ENGINE=$1
URL=$2
MODEL=$3
CSV=${CSV:-results.csv}
EXTRA=${EXTRA:-}
# at least 200 requests per level, so the p99 is not just the slowest request
MIN_REQ=${MIN_REQ:-200}   # set MIN_REQ=1000 for a p99 you can trust

# warm up first: the first requests are always slow
python ../demos/bench.py --url "$URL" --model "$MODEL" --concurrency 4 --num-requests 8 $EXTRA | tail -3

for C in 1 4 8 16 32 64; do
  echo "== $ENGINE concurrency $C"
  python ../demos/bench.py --url "$URL" --model "$MODEL" \
    --concurrency "$C" --num-requests $((C * 4 > MIN_REQ ? C * 4 : MIN_REQ)) $EXTRA \
    --engine "$ENGINE" --csv "$CSV"
done
[ -f "$CSV" ] || { echo "no results were written: every request failed"; exit 1; }
echo "done -> $CSV"
