#!/usr/bin/env python3
"""spec_server_check.py — is --spec-k output the same as plain greedy output? (S5.3c)

Greedy speculative decoding must emit exactly the target's own greedy text, up
to fp16 near-ties (a K+1-row verify GEMM accumulates in a different order than
a 1-row decode GEMM; offline, 8 of 16 prompts flipped once at a near-tie).

Usage:
  # 1. server WITHOUT --spec-k
  python3 tools/spec_server_check.py --save benchmarks/check_plain.json
  # 2. restart the server WITH --spec-k 2
  python3 tools/spec_server_check.py --save benchmarks/check_spec.json
  # 3. compare
  python3 tools/spec_server_check.py --compare benchmarks/check_plain.json benchmarks/check_spec.json

--concurrency > 1 exercises the batched path: several decoders (and prefills)
share every draft and verify pass.
"""
import argparse, json, sys, threading, urllib.request


def complete(url, prompt, max_tokens):
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    text, finish = "", None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            ch = json.loads(payload)["choices"][0]
            text += ch.get("text", "")
            finish = ch.get("finish_reason") or finish
    return text, finish


def save(args):
    with open(args.prompts) as f:
        prompts = [l.strip() for l in f if l.strip()][: args.n]
    out = [None] * len(prompts)
    nxt = [0]
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                i = nxt[0]; nxt[0] += 1
            if i >= len(prompts):
                return
            out[i] = dict(zip(("text", "finish"), complete(args.url, prompts[i], args.max_tokens)))
            out[i]["prompt"] = prompts[i]

    ts = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    for t in ts: t.start()
    for t in ts: t.join()
    with open(args.save, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {len(out)} completions to {args.save}")


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    same, diverged = 0, []
    for i, (x, y) in enumerate(zip(a, b)):
        if x["text"] == y["text"]:
            same += 1
            continue
        j = 0
        while j < min(len(x["text"]), len(y["text"])) and x["text"][j] == y["text"][j]:
            j += 1
        diverged.append((i, j, x["text"], y["text"]))
    print(f"{same}/{len(a)} identical, {len(diverged)} diverged")
    for i, j, x, y in diverged:
        print(f"\n#{i}: first difference at character {j} of {len(x)}")
        print(f"  plain: ...{x[max(0, j - 40): j + 40]!r}")
        print(f"  spec : ...{y[max(0, j - 40): j + 40]!r}")
    print("\nHow to read it: a divergence LATE in the text, where both continuations are")
    print("plausible, is an fp16 near-tie. A divergence in the first few tokens, or into")
    print("garbage, is a bug in spec_step.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080/v1/completions")
    ap.add_argument("--prompts", default="prompts.txt")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--save")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    elif args.save:
        save(args)
    else:
        sys.exit("need --save FILE or --compare A B")
