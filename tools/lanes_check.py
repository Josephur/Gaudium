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

`--serial` checks every prompt alone, one after another (a single-stream server's streams).

Options add prompts: `--long 6000,10000` long prompts of about that many words (their streams
cover several KV chunks), `--image` one prompt with a generated PNG (an image request runs alone
on a lane server), `--suite <suite.json>` the image cases of a vision suite file (its images and
prompts; an answer without the case's expected words fails the check). `--turns` checks routing
after the concurrent runs: every long prompt's next turn (its reply and one more user message)
must take over rows of the lane that held it and equal its reference; with `--park S` the card is
released first (a server with a short `--idle-park`), and the rows must come back from the host-RAM
tier (`--cache-ram`).

Only the Python standard library is used. Results go to stdout and, with `--out`, to a JSON file.
The exit status is 1 when a check fails (a mismatching or failed stream, a next turn that reused
nothing), else 0.
"""

import argparse
import base64
import json
import struct
import threading
import time
import urllib.request
import zlib

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


def long_prompt(words):
    """About `words` words of numbered sentences, then a question about them."""
    parts = []
    i = 0
    while len(parts) < words:
        i += 1
        parts.extend(f"Note {i}: the {['red', 'green', 'blue', 'amber'][i % 4]} crate in row {i * 7 % 97} holds {i * 13 % 89} parts.".split())
    return " ".join(parts[:words]) + "\n\nWhich rows hold amber crates? Answer briefly."


def png_data_url(w=96, h=96):
    """A small RGB gradient as a PNG data URL (no image library needed)."""
    rows = b"".join(b"\x00" + bytes(v for x in range(w) for v in (x * 255 // w, y * 255 // h, 128)) for y in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


IMAGE_MARK = "<image>"
# the next user message of a long prompt's conversation (`--turns`)
FOLLOW_UP = "Now list them in ascending order."


def wait_parked(base, timeout):
    """Waits until the server has released the card (its engine state is `parked`)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            with urllib.request.urlopen(base.rstrip("/") + "/gaudium/engine", timeout=5) as r:
                if json.loads(r.read()).get("state") == "parked":
                    print(f"server parked after {time.monotonic() - t0:.1f}s", flush=True)
                    return True
        except Exception:  # noqa: BLE001 (the server may be busy releasing)
            pass
        time.sleep(1)
    print(f"server not parked after {timeout}s", flush=True)
    return False
# image prompts of a vision suite file (`--suite`): prompt -> (data URLs, expected word groups)
SUITE_IMAGES = {}


def file_data_url(path):
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def load_suite(path):
    """The image cases of a vision suite file, as prompts of their own (their images and the
    words their answers must contain)."""
    import os
    with open(path) as f:
        suite = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    out = []
    for case in suite.get("cases", []):
        urls = [file_data_url(os.path.join(base, im)) for im in case["images"]]
        prompt = IMAGE_MARK + f"[{case['name']}] " + case["prompt"]
        SUITE_IMAGES[prompt] = (urls, case.get("expect", []))
        out.append(prompt)
    return out


def expect_ok(prompt, content):
    """Does an answer to a suite image prompt contain one word of every expected group?"""
    _, groups = SUITE_IMAGES.get(prompt, (None, []))
    low = content.lower()
    return all(any(w.lower() in low for w in g) for g in groups)


def chat(base, model, prompt, mode, seed, max_tokens, api_key, timeout, history=None):
    if prompt in SUITE_IMAGES:
        urls, _ = SUITE_IMAGES[prompt]
        text = prompt[len(IMAGE_MARK):].split("] ", 1)[1]
        content = [{"type": "image_url", "image_url": {"url": u}} for u in urls] + [{"type": "text", "text": text}]
    elif prompt.startswith(IMAGE_MARK):
        content = [{"type": "image_url", "image_url": {"url": png_data_url()}},
                   {"type": "text", "text": prompt[len(IMAGE_MARK):]}]
    else:
        content = prompt
    body = {
        "model": model,
        "messages": (history or []) + [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": False,
        "gaudium_return_tokens": True,
    }
    if prompt in SUITE_IMAGES:
        # as the vision suite asks them: a short answer, no thinking
        body["chat_template_kwargs"] = {"enable_thinking": False}
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
    msg = resp.get("choices", [{}])[0].get("message", {})
    return {
        "tokens": g.get("tokens", []),
        "content": msg.get("content") or "",
        "reused": g.get("prefix", {}).get("reused_tokens", 0),
        "tier": g.get("prefix", {}).get("tier", ""),
        "wall_s": wall,
        "prefill_ms": t.get("prompt_ms", 0.0),
        "decode_ms": t.get("predicted_ms", 0.0),
        "n": t.get("predicted_n", 0),
        "queued_ms": g.get("queued_ms", 0.0),
        "method": g.get("engine_method", ""),
        "launches": t.get("launches", 0),
        "draft_n": t.get("draft_n", 0),
        "draft_accepted": t.get("draft_n_accepted", 0),
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
    ap.add_argument("--long", default="", help="extra long prompts of about these many words")
    ap.add_argument("--image", action="store_true", help="one extra prompt with an image")
    ap.add_argument("--suite", default="", help="a vision suite file: its image cases as extra prompts, each answer checked for its expected words")
    ap.add_argument("--turns", action="store_true", help="next turns of the long prompts must reuse rows (and equal their references)")
    ap.add_argument("--park", type=float, default=0, help="with --turns: wait up to this many seconds for the server to release the card before the next turns, which must then come back from host memory (--cache-ram)")
    ap.add_argument("--serial", action="store_true", help="with --check: every prompt alone, one after another")
    a = ap.parse_args()
    modes = [m for m in a.modes.split(",") if m]
    n = min(a.prompts, len(PROMPTS))
    first_long = n
    for w in [int(x) for x in a.long.split(",") if x]:
        PROMPTS.insert(n, long_prompt(w))
        n += 1
    long_idx = list(range(first_long, n))
    if a.image:
        PROMPTS.insert(n, IMAGE_MARK + "Describe the colours of this image in one sentence.")
        n += 1
    if a.suite:
        for p in load_suite(a.suite):
            PROMPTS.insert(n, p)
            n += 1
    out = {"base": a.base, "model": a.model, "max_tokens": a.max_tokens, "modes": modes}

    if a.write_ref:
        ref = {}
        for i in range(n):
            for m in modes:
                r = chat(a.base, a.model, PROMPTS[i], m, a.seed + i, a.max_tokens, a.api_key, a.timeout)
                ref[key(i, m)] = r
                sem = ""
                if PROMPTS[i] in SUITE_IMAGES:
                    sem = ", expected words " + ("found" if expect_ok(PROMPTS[i], r["content"]) else "MISSING")
                print(f"ref prompt {i:2d} {m:7s}: {r['n']:4d} tokens, decode {r['n'] / max(r['decode_ms'], 1e-9) * 1e3:7.1f} tok/s, "
                      f"prefill {r['prefill_ms']:7.1f} ms ({r['method']}){sem}", flush=True)
        if a.turns:
            # each long prompt's first turn and next turn, one request at a time
            for i in long_idx:
                first = chat(a.base, a.model, PROMPTS[i], "greedy", a.seed + i, a.max_tokens, a.api_key, a.timeout)
                hist = [{"role": "user", "content": PROMPTS[i]}, {"role": "assistant", "content": first["content"]}]
                nxt = chat(a.base, a.model, FOLLOW_UP, "greedy", a.seed, 32, a.api_key, a.timeout, history=hist)
                ref[f"turn1:{i}"] = first
                ref[f"turn2:{i}"] = nxt
                print(f"ref turns of long prompt {i}: {first['n']} + {nxt['n']} tokens", flush=True)
        with open(a.write_ref, "w") as f:
            json.dump(ref, f)
        out["ref"] = ref

    if a.check:
        with open(a.check) as f:
            ref = json.load(f)
        runs = []
        bad = 0
        plan = [(1, [i]) for i in range(n)] if a.serial else [(int(x), None) for x in a.concurrency.split(",") if x]
        for c, only in plan:
            for m in modes:
                # the extra prompts (long, image) always take part: their streams are the hard ones
                extra = list(range(first_long, n))
                order = extra + [i for i in range(n) if i not in extra]
                idx = only if only is not None else [order[j % n] for j in range(c)]
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
                    if PROMPTS[idx[j]] in SUITE_IMAGES and not expect_ok(PROMPTS[idx[j]], res[j]["content"]):
                        diffs.append({"slot": j, "prompt": idx[j], "expected_words": "missing"})
                        d = d if d is not None else -1
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
                    "launches": sum(r["launches"] for r in ok),
                    "draft_n": sum(r["draft_n"] for r in ok),
                    "draft_accepted": sum(r["draft_accepted"] for r in ok),
                    "diffs": diffs,
                }
                # decode tokens per launch (each stream's first token comes from its prefill)
                row["tok_per_launch"] = dec / row["launches"] if row["launches"] else 0.0
                runs.append(row)
                print(f"c={c} {m:7s}: {same}/{c} identical, {toks} tokens in {wall:6.2f}s = {row['total_tok_s']:7.1f} tok/s total "
                      f"(decode {row['decode_tok_s']:7.1f} tok/s; per stream "
                      f"{sum(row['per_stream_decode_tok_s']) / max(len(ok), 1):6.1f})"
                      + (f", {row['tok_per_launch']:.2f} tok/launch, {row['draft_accepted']}/{row['draft_n']} drafts accepted"
                         if row["launches"] else "")
                      + (f"  DIFF {diffs}" if diffs else ""), flush=True)
        out["runs"] = runs
        out["mismatches"] = bad
        print(f"mismatching streams: {bad}")
        missing = [i for i in long_idx if f"turn1:{i}" not in ref or f"turn2:{i}" not in ref] if a.turns else []
        if missing:
            # a next turn is only checked against references of both its turns
            print(f"--turns: the reference file has no turn references for long prompt(s) {missing}; "
                  "write it again with --write-ref ... --turns (same prompts and --long)", flush=True)
            bad += len(missing)
            out["mismatches"] = bad
        elif a.turns and long_idx:
            # each long prompt again with its reply and one more message: the lane that holds the
            # conversation takes its rows over (the other long prompts ran in between), or with
            # --park the host-RAM tier gives them back after the card was released
            turns = []
            for i in long_idx:
                first = chat(a.base, a.model, PROMPTS[i], "greedy", a.seed + i, a.max_tokens, a.api_key, a.timeout)
                r1 = ref[f"turn1:{i}"]
                if first_diff(first["tokens"], r1["tokens"]) is not None:
                    bad += 1
                    print(f"turn 1 of long prompt {i}: DIFF against its reference", flush=True)
                # the history of the reference's first turn, so that the next turn's prompt is its
                content = r1["content"]
                hist = [{"role": "user", "content": PROMPTS[i]}, {"role": "assistant", "content": content}]
                turns.append((i, hist))
            if a.park:
                wait_parked(a.base, a.park)
            res2 = [None] * len(turns)

            def turn2(j):
                i, hist = turns[j]
                res2[j] = chat(a.base, a.model, FOLLOW_UP, "greedy", a.seed, 32, a.api_key, a.timeout, history=hist)

            th = [threading.Thread(target=turn2, args=(j,)) for j in range(len(turns))]
            for t in th:
                t.start()
            for t in th:
                t.join()
            for (i, _), nxt in zip(turns, res2):
                want = a.park and "host" or None
                ok = nxt is not None and nxt["reused"] > 0 and (want is None or nxt["tier"] == want)
                r2 = ref[f"turn2:{i}"]
                same = nxt is not None and first_diff(nxt["tokens"], r2["tokens"]) is None
                bad += (0 if ok else 1) + (0 if same else 1)
                print(f"turn 2 of long prompt {i}: {nxt and nxt['reused']} prompt rows taken over from {nxt and nxt['tier']} "
                      f"({'ok' if ok else 'NONE'}), {'identical to' if same else 'DIFF against'} its reference", flush=True)
                out.setdefault("turns", []).append({"prompt": i, "reused": nxt and nxt["reused"], "tier": nxt and nxt["tier"], "identical": same})
            out["mismatches"] = bad

    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    # a failed check (a mismatching or failed stream, a next turn that reused nothing) fails
    return 1 if out.get("mismatches") else 0


if __name__ == "__main__":
    raise SystemExit(main())
