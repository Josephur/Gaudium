#!/usr/bin/env python3
"""Parallel HTTP check of a Gaudium server's lanes (`--parallel N`).

Two steps, against a running server:

  1. `--write-ref ref.json`: every (prompt, mode) request once, one at a time (best against a
     server started with `--parallel 1`): the reference token streams.
  2. `--check ref.json --concurrency 1,2,4,8`: for each concurrency c, c requests sent at once
     (distinct prompts), for every mode. Each stream is compared with its reference, token by
     token, and the total decode throughput of the batch is reported.

Modes: `greedy` (temperature 0) and `sampled` (temperature 1.0, top-k 20, top-p 0.95 and a fixed
seed per prompt): a seeded sampled stream must equal its reference as exactly as a greedy one.

Only the Python standard library is used. Results go to stdout and, with `--out`, to a JSON file.
"""

import argparse
import json
import threading
import time
import urllib.request

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Explain in three short paragraphs why the sky is blue.",
    "List ten countries in Africa and their capitals as a numbered list.",
    "Write a short story of about 150 words about a lighthouse keeper who finds a message in a bottle.",
    "What are the differences between TCP and UDP? Answer with a compact comparison table.",
    "Translate into French: 'The library opens at nine and closes at six, except on Sundays.'",
    "Give a step-by-step recipe for a simple tomato soup.",
    "Write a SQL query that returns the five customers with the highest total order value.",
    "Summarize the plot of Romeo and Juliet in five sentences.",
    "Write a haiku about autumn rain, then explain its imagery in two sentences.",
    "Describe how a hash table handles collisions, with one example in C.",
    "What is the derivative of x^3 * sin(x)? Show the steps.",
    "Write a bash script that counts the lines of every .rs file under a directory.",
    "Name five common logical fallacies and give a one-line example of each.",
    "Explain what a Gaudi 2 accelerator is used for, in plain words, in under 120 words.",
    "Write a JSON object describing a book with title, author, year and three keywords.",
]


def chat(base, model, prompt, mode, seed, max_tokens, api_key, timeout):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        "gaudium_return_tokens": True,
    }
    if mode == "greedy":
        body["temperature"] = 0.0
    else:
        body.update({"temperature": 1.0, "top_k": 20, "top_p": 0.95, "seed": seed})
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    wall = time.monotonic() - t0
    g = resp.get("gaudium", {})
    t = resp.get("timings", {})
    return {
        "tokens": g.get("tokens", []),
        "wall_s": wall,
        "prefill_ms": t.get("prompt_ms", 0.0),
        "decode_ms": t.get("predicted_ms", 0.0),
        "n": t.get("predicted_n", 0),
        "queued_ms": g.get("queued_ms", 0.0),
        "method": g.get("engine_method", ""),
    }


def key(i, mode):
    return f"{i}:{mode}"


def first_diff(a, b):
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--prompts", type=int, default=8, help="distinct prompts (at most %d)" % len(PROMPTS))
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--modes", default="greedy,sampled")
    ap.add_argument("--seed", type=int, default=1234, help="seed of prompt 0 (prompt i uses seed + i)")
    ap.add_argument("--write-ref", default="")
    ap.add_argument("--check", default="")
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    modes = [m for m in a.modes.split(",") if m]
    n = min(a.prompts, len(PROMPTS))
    out = {"base": a.base, "model": a.model, "max_tokens": a.max_tokens, "modes": modes}

    if a.write_ref:
        ref = {}
        for i in range(n):
            for m in modes:
                r = chat(a.base, a.model, PROMPTS[i], m, a.seed + i, a.max_tokens, a.api_key, a.timeout)
                ref[key(i, m)] = r
                print(f"ref prompt {i:2d} {m:7s}: {r['n']:4d} tokens, decode {r['n'] / max(r['decode_ms'], 1e-9) * 1e3:7.1f} tok/s, "
                      f"prefill {r['prefill_ms']:7.1f} ms ({r['method']})", flush=True)
        with open(a.write_ref, "w") as f:
            json.dump(ref, f)
        out["ref"] = ref

    if a.check:
        with open(a.check) as f:
            ref = json.load(f)
        runs = []
        bad = 0
        for c in [int(x) for x in a.concurrency.split(",") if x]:
            for m in modes:
                idx = [i % n for i in range(c)]
                res = [None] * c
                errs = [None] * c

                def worker(j):
                    try:
                        i = idx[j]
                        res[j] = chat(a.base, a.model, PROMPTS[i], m, a.seed + i, a.max_tokens, a.api_key, a.timeout)
                    except Exception as e:  # noqa: BLE001 (reported below)
                        errs[j] = str(e)

                t0 = time.monotonic()
                th = [threading.Thread(target=worker, args=(j,)) for j in range(c)]
                for t in th:
                    t.start()
                for t in th:
                    t.join()
                wall = time.monotonic() - t0
                same = 0
                diffs = []
                for j in range(c):
                    if errs[j] or res[j] is None:
                        diffs.append({"slot": j, "error": errs[j]})
                        continue
                    want = ref[key(idx[j], m)]["tokens"]
                    d = first_diff(res[j]["tokens"], want)
                    if d is None:
                        same += 1
                    else:
                        diffs.append({"slot": j, "prompt": idx[j], "first_diff": d,
                                      "got_len": len(res[j]["tokens"]), "ref_len": len(want)})
                bad += c - same
                ok = [r for r in res if r]
                toks = sum(r["n"] for r in ok)
                # decode throughput: tokens after the first, over the span from the first prefill's
                # end to the last stream's end (prefills of later streams fall inside it)
                dec = sum(max(r["n"] - 1, 0) for r in ok)
                span = max((r["wall_s"] for r in ok), default=0.0) - min(
                    (r["queued_ms"] / 1e3 + r["prefill_ms"] / 1e3 for r in ok), default=0.0)
                row = {
                    "concurrency": c, "mode": m, "identical": same, "of": c,
                    "tokens": toks, "wall_s": wall, "total_tok_s": toks / max(wall, 1e-9),
                    "decode_tok_s": dec / max(span, 1e-9),
                    "per_stream_decode_tok_s": [r["n"] / max(r["decode_ms"], 1e-9) * 1e3 for r in ok],
                    "prefill_ms": [r["prefill_ms"] for r in ok],
                    "diffs": diffs,
                }
                runs.append(row)
                print(f"c={c} {m:7s}: {same}/{c} identical, {toks} tokens in {wall:6.2f}s = {row['total_tok_s']:7.1f} tok/s total "
                      f"(decode {row['decode_tok_s']:7.1f} tok/s; per stream "
                      f"{sum(row['per_stream_decode_tok_s']) / max(len(ok), 1):6.1f})"
                      + (f"  DIFF {diffs}" if diffs else ""), flush=True)
        out["runs"] = runs
        out["mismatches"] = bad
        print(f"mismatching streams: {bad}")

    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
