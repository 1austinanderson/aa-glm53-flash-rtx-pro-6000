#!/usr/bin/env python3
"""GLM-5.3-Flash prefill sweep (4k/16k/64k TOKENS, best-of-3, unique prompts = APC-proof)
+ decode at C1/C2/C4 (400 tok/stream). Same methodology as
/mnt/raid/ds4-dspark-patches/prefill_bench.py, but prompts are calibrated to the GLM
tokenizer and capped under max-model-len. Usage: glm_bench.py <label> [base] [ctx]"""
import json, random, sys, time, threading, urllib.request
label = sys.argv[1] if len(sys.argv) > 1 else "run"
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:12921/v1"
CTX = int(sys.argv[3]) if len(sys.argv) > 3 else 65536
WORDS = ("margin deposit yield accrual duration basis credit tier lease swap covenant tranche "
         "spread ledger accretion charter reserve capital syndicate warrant collar equity "
         "mezzanine subordinated senior").split()
MODEL = json.load(urllib.request.urlopen(BASE + "/models"))["data"][0]["id"]

def req(body, t=1800):
    r = urllib.request.urlopen(urllib.request.Request(BASE + "/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=t)
    return json.load(r)

def prompt_of(n_words):
    return " ".join(random.choice(WORDS) for _ in range(n_words)) + "\nSay OK."

def one_prefill(n_words):
    t0 = time.time()
    d = req({"model": MODEL, "messages": [{"role": "user", "content": prompt_of(n_words)}],
             "max_tokens": 2, "temperature": 0, "reasoning_effort": "low"})
    return d["usage"]["prompt_tokens"], time.time() - t0

# calibrate words -> tokens on a 2k-word probe
pt, _ = one_prefill(2000); tpw = pt / 2000
print(f"[{label}] model={MODEL} calib {tpw:.3f} tok/word", flush=True)

def prefill(target_tokens, reps=3):
    n_words = int(target_tokens / tpw)
    best = 0; pts = 0; wall = 0
    for _ in range(reps):
        p, dt = one_prefill(n_words)
        if p / dt > best: best, pts, wall = p / dt, p, dt
    return {"target": target_tokens, "prompt_tokens": pts, "wall_s": round(wall, 2), "prefill_tok_s": round(best)}

def decode_stream(i, max_tok, out):
    p = f"[{random.random()}] Write a detailed essay about community bank M&A, part {i}."
    t0 = time.time()
    d = req({"model": MODEL, "messages": [{"role": "user", "content": p}],
             "max_tokens": max_tok, "temperature": 0.8, "reasoning_effort": "low"})
    out.append((d["usage"]["completion_tokens"], time.time() - t0))

def decode(conc, max_tok=400):
    out = []; ts = [threading.Thread(target=decode_stream, args=(i, max_tok, out)) for i in range(conc)]
    t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]; wall = time.time() - t0
    toks = sum(n for n, _ in out)
    return {"conc": conc, "aggregate_tok_s": round(toks / wall, 1), "per_stream_tok_s": [round(n / d, 1) for n, d in out], "tokens": toks, "wall_s": round(wall, 1)}

decode(1, 150); time.sleep(10)  # warmup + settle
res = {"label": label, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "prefill": [], "decode": []}
for tgt in (4000, 16000, min(64000, CTX - 2048)):
    r = prefill(tgt); res["prefill"].append(r)
    print(f"[{label}] prefill {tgt//1000}k (pt={r['prompt_tokens']}): {r['prefill_tok_s']} tok/s best-of-3 ({r['wall_s']}s)", flush=True)
for c in (1, 2, 4):
    r = decode(c); res["decode"].append(r)
    print(f"[{label}] decode C{c}: {r['aggregate_tok_s']} tok/s aggregate, per-stream {r['per_stream_tok_s']}", flush=True)
with open("results/bench_results.jsonl", "a") as f: f.write(json.dumps(res) + "\n")
print(json.dumps(res))
