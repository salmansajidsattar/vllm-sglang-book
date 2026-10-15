"""One small benchmark for BOTH engines (they both speak the OpenAI API).

  python bench.py --url http://localhost:8000 --model MODEL --concurrency 16

Measures per request: TTFT (time to first token), TPOT (time per output
token after the first), and overall output tokens/second.

Tokens are counted from the "usage" the server sends in the last streamed
chunk (the request sets stream_options include_usage). If the server sends no
usage, the script counts streamed chunks instead and says so.

Options that change what you measure:
  --unique-prompts  put a random tag at the start of every prompt, so almost
                    no prefix is shared (the chat template header still is).
                    Default: off, every request sends the SAME prompt at
                    temperature 0. That inflates prefix-cache hits and n-gram
                    speculative acceptance. Do not use the default for cache
                    or speculative decoding tests. Use --unique-prompts, or
                    better, your own varied prompts.
  --ignore-eos      ask the server not to stop at the end-of-text token, so
                    every request makes max-tokens tokens. Default: off.
                    Both vLLM and SGLang accept "ignore_eos" in the request.
  --temperature     sampling temperature sent with every request (default 0).
                    We set it so the server's own defaults do not matter.

Needs: pip install httpx
"""
import argparse
import asyncio
import json
import math
import os
import random
import statistics
import string
import time

import httpx

STATE = {"now": 0, "peak": 0}  # requests in flight now, and the most seen
PROMPT = "Explain how a GPU executes a matrix multiplication. " * 20


def make_prompt(unique):
    if not unique:
        return PROMPT
    # a random start (8 letters) means no two prompts share a prefix
    tag = "".join(random.choices(string.ascii_lowercase, k=8))
    return f"[{tag}] " + PROMPT


async def one_request(client, args):
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": make_prompt(args.unique_prompts)}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},  # vLLM and SGLang both accept it
    }
    if args.ignore_eos:
        body["ignore_eos"] = True
    STATE["now"] += 1
    STATE["peak"] = max(STATE["peak"], STATE["now"])
    start = time.perf_counter()
    first, n_chunks, n_usage = None, 0, None
    try:
        async with client.stream("POST", args.url + "/v1/chat/completions", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line.endswith("[DONE]"):
                    continue
                obj = json.loads(line[6:])
                usage = obj.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    n_usage = usage["completion_tokens"]  # the exact count
                choices = obj.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}
                if delta.get("content"):
                    n_chunks += 1
                    if first is None:
                        first = time.perf_counter()
    finally:
        STATE["now"] -= 1
    end = time.perf_counter()
    if first is None:
        raise RuntimeError("no tokens received")
    n_tokens = n_usage if n_usage is not None else n_chunks
    ttft = first - start
    tpot = (end - first) / max(n_tokens - 1, 1)
    return ttft, tpot, n_tokens, n_usage is not None


async def run(args):
    if args.unique_prompts:
        print("prompts         : random tag at the start, almost no shared prefix")
    else:
        print("WARNING: every request sends the SAME prompt at temperature 0. This "
              "inflates prefix-cache hits and n-gram speculative acceptance. For "
              "cache or speculative tests use --unique-prompts or your own varied prompts.")
    sem = asyncio.Semaphore(args.concurrency)
    # httpx allows only 100 connections by default. Remove the limit, or
    # --concurrency above 100 would silently run at 100.
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=300, limits=limits) as client:

        async def guarded():
            async with sem:
                return await one_request(client, args)

        t0 = time.perf_counter()
        # return_exceptions=True: one failed request does not stop the run
        outs = await asyncio.gather(*[guarded() for _ in range(args.num_requests)],
                                    return_exceptions=True)
        wall = time.perf_counter() - t0

    results = [o for o in outs if not isinstance(o, BaseException)]
    errors = [o for o in outs if isinstance(o, BaseException)]
    print(f"requests ok     : {len(results)}/{args.num_requests}")
    if errors:
        print(f"requests failed : {len(errors)}")
        for msg in sorted({f"{type(e).__name__}: {e}" for e in errors})[:3]:
            print(f"  error: {msg}")
    if not results:
        print("no successful requests, nothing to report")
        raise SystemExit(1)
    print(f"peak in flight  : {STATE['peak']} (--concurrency {args.concurrency})")
    if STATE["peak"] < min(args.concurrency, args.num_requests):
        print("WARNING: the peak is below --concurrency, so the load was lower than you asked for.")
    if len(results) < 100:
        print(f"WARNING: only {len(results)} good requests. With fewer than 100, "
              "p99 is just the slowest request. Use 1000 or more for p99, or read p95.")
    elif len(results) < 1000:
        print(f"WARNING: only {len(results)} good requests. p99 then rests on fewer than "
              "10 slow requests and moves a lot. Use 1000 or more for p99, or read p95.")

    ttfts = sorted(r[0] for r in results)
    tpots = [r[1] for r in results]
    tokens = sum(r[2] for r in results)
    p95 = ttfts[math.ceil(len(ttfts) * 0.95) - 1]  # nearest rank
    p99 = ttfts[math.ceil(len(ttfts) * 0.99) - 1]
    by_usage = sum(r[3] for r in results)
    if by_usage == len(results):
        method = "server usage (completion_tokens)"
    elif by_usage == 0:
        method = "streamed chunk count (the server sent no usage, so this is an estimate)"
    else:
        method = f"mixed: usage for {by_usage} requests, chunk count for the rest"
    print(f"token counting  : {method}")
    print(f"TTFT p50/p95/p99: {statistics.median(ttfts)*1000:.0f} / {p95*1000:.0f} / {p99*1000:.0f} ms")
    print(f"TPOT mean       : {statistics.mean(tpots)*1000:.1f} ms")
    print(f"output tokens/s : {tokens / wall:.0f}")
    print(f"tokens per reply: {tokens / len(results):.0f} on average (max-tokens {args.max_tokens})")
    if args.csv:  # append one row, used by deploy/run_sweep.sh
        new = not os.path.exists(args.csv)
        with open(args.csv, "a") as f:
            if new:
                f.write("engine,concurrency,ttft_p50_ms,ttft_p99_ms,tpot_ms,tok_s,ok,failed\n")
            f.write(f"{args.engine},{args.concurrency},"
                    f"{statistics.median(ttfts)*1000:.0f},{p99*1000:.0f},"
                    f"{statistics.mean(tpots)*1000:.1f},{tokens / wall:.0f},"
                    f"{len(results)},{args.num_requests - len(results)}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="engine", help="label for the CSV row")
    p.add_argument("--csv", default="", help="append a result row to this file")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--num-requests", type=int, default=200,
                   help="use 1000 or more if you want to trust the p99")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--unique-prompts", action="store_true",
                   help="random tag at the start of each prompt (almost no shared prefix)")
    p.add_argument("--ignore-eos", action="store_true",
                   help="send ignore_eos=true, so every reply has max-tokens tokens")
    asyncio.run(run(p.parse_args()))
