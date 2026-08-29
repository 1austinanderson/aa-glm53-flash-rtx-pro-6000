#!/usr/bin/env python3
"""Heavy stress for GLM-5.3-Flash on :12921 (DFlash + RAM offload). Usage: glm_stress_heavy.py <label> [base] [ctx]"""
import base64, json, random, subprocess, sys, threading, time, urllib.request, zlib, struct
label = sys.argv[1] if len(sys.argv) > 1 else "heavy"; BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:12921/v1"; CTX = int(sys.argv[3]) if len(sys.argv) > 3 else 393216
MODEL = json.load(urllib.request.urlopen(BASE + "/models"))["data"][0]["id"]
W = "margin deposit yield accrual duration basis credit tier lease swap covenant tranche spread ledger accretion charter reserve".split()
peak = {"used": 0, "minfree": 10**9, "shm": 0, "stop": False}
def sampler():
    while not peak["stop"]:
        try:
            o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()[:2]
            for l in o:
                u, f = [int(x) for x in l.split(",")]; peak["used"] = max(peak["used"], u); peak["minfree"] = min(peak["minfree"], f)
            s = subprocess.check_output(["df", "-m", "/dev/shm"], text=True).splitlines()[-1].split()[2]; peak["shm"] = max(peak["shm"], int(s))
        except Exception: pass
        time.sleep(0.5)
threading.Thread(target=sampler, daemon=True).start()
def req(body, t=3600):
    r = urllib.request.urlopen(urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=t); return json.load(r)
def metric(name):
    try:
        for l in urllib.request.urlopen(BASE.replace("/v1", "/metrics"), timeout=10).read().decode().splitlines():
            if l.startswith(name) and not l.startswith("#"): return float(l.split()[-1])
    except Exception: pass
    return None
def prompt(n_tok, seed=None):
    r = random.Random(seed) if seed is not None else random
    return " ".join(r.choice(W) for _ in range(int(n_tok / 1.33))) + "\nSummarize in one sentence."
def png():
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress((b"\x00" + bytes([200, 30, 30]) * 64) * 64)) + chunk(b"IEND", b"")
IMG = base64.b64encode(png()).decode()
errors, results = [], []
def par(fn, n):
    outs = [None] * n
    def w(i):
        try: outs[i] = fn(i)
        except Exception as e: errors.append(f"{type(e).__name__}: {str(e)[:160]}")
    ts = [threading.Thread(target=w, args=(i,)) for i in range(n)]; [t.start() for t in ts]; [t.join() for t in ts]; return outs
def step(name, fn):
    t0 = time.time(); e0 = len(errors)
    try: out = fn()
    except Exception as e: errors.append(f"{name}: {type(e).__name__}: {str(e)[:160]}"); out = None
    ok = len(errors) == e0; results.append((name, ok, round(time.time() - t0, 1), peak["minfree"]))
    print(f"[{label}] {name}: {'OK' if ok else 'FAIL'} {time.time()-t0:.1f}s minfree={peak['minfree']} MiB shm={peak['shm']} MiB", flush=True)
p0, o0 = metric("vllm:num_preemptions_total"), metric("vllm:kv_offload_total_bytes_total")
req({"model": MODEL, "messages": [{"role": "user", "content": "warm"}], "max_tokens": 16, "reasoning_effort": "low"})
big = min(CTX - 8192, 380000)
step(f"1x{big//1000}k near-max prefill", lambda: req({"model": MODEL, "messages": [{"role": "user", "content": prompt(big, 1)}], "max_tokens": 32, "reasoning_effort": "low"})["usage"]["prompt_tokens"])
step("C8x60k prefill burst", lambda: par(lambda i: req({"model": MODEL, "messages": [{"role": "user", "content": prompt(60000, 100 + i)}], "max_tokens": 16, "reasoning_effort": "low"})["usage"]["prompt_tokens"], 8))
step("C8x2k thinking decode (effort max)", lambda: par(lambda i: req({"model": MODEL, "messages": [{"role": "user", "content": f"[{i}] Prove the sum of the first n odd numbers is n^2, then give three generalizations with proofs."}], "max_tokens": 2048, "temperature": 1.0, "top_p": 0.95, "reasoning_effort": "max"})["usage"]["completion_tokens"], 8))
step("offload churn: 12x60k distinct", lambda: [req({"model": MODEL, "messages": [{"role": "user", "content": prompt(60000, 200 + i)}], "max_tokens": 8, "reasoning_effort": "low"})["usage"]["prompt_tokens"] for i in range(12)])
def rehit():
    out = []
    for i in (200, 201, 202):
        t0 = time.time(); req({"model": MODEL, "messages": [{"role": "user", "content": prompt(60000, i)}], "max_tokens": 8, "reasoning_effort": "low"}); out.append(round(time.time() - t0, 2))
    print(f"[{label}]   re-hit times (RAM hits if ~1-2s, cold ~17s): {out}", flush=True); return out
step("RAM re-hits of the 3 oldest prompts", rehit)
def mixed():
    end = time.time() + 300; n = [0]
    def one(i):
        r = random.Random(); k = r.random()
        if k < 0.3: body = {"model": MODEL, "messages": [{"role": "user", "content": prompt(40000)}], "max_tokens": 32, "reasoning_effort": "low"}
        elif k < 0.6: body = {"model": MODEL, "messages": [{"role": "user", "content": f"[{r.random()}] Write 700 words on deposit betas."}], "max_tokens": 1024, "reasoning_effort": "low"}
        elif k < 0.8: body = {"model": MODEL, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{IMG}"}}, {"type": "text", "text": "Describe this image in one line."}]}], "max_tokens": 64, "reasoning_effort": "low"}
        else: body = {"model": MODEL, "messages": [{"role": "user", "content": f"[{r.random()}] List ten US bank holding companies."}], "max_tokens": 256, "logprobs": True, "top_logprobs": 20, "reasoning_effort": "low"}
        req(body); n[0] += 1
    def worker(i):
        while time.time() < end:
            try: one(i)
            except Exception as e: errors.append(f"mixed: {type(e).__name__}: {str(e)[:160]}"); time.sleep(1)
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(8)]; [t.start() for t in ts]; [t.join() for t in ts]
    print(f"[{label}]   mixed loop: {n[0]} requests in 300s at concurrency 8", flush=True); return n[0]
step("5-min mixed loop @C8 (40k prefill / 1k decode / image / logprobs)", mixed)
peak["stop"] = True; time.sleep(1)
p1, o1 = metric("vllm:num_preemptions_total"), metric("vllm:kv_offload_total_bytes_total")
tot = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True).split()[0])
ok = not errors and peak["minfree"] >= 300
print(f"[{label}] RESULT {'PASS' if ok else 'FAIL'} peak_used={peak['used']}/{tot} MiB min_free={peak['minfree']} MiB shm_peak={peak['shm']} MiB preemptions={p0}->{p1} errors={len(errors)}", flush=True)
for e in errors[:8]: print("  ERR", e)
with open("results/stress_results.jsonl", "a") as f: f.write(json.dumps({"label": label, "heavy": True, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "steps": results, "peak_used_mib": peak["used"], "min_free_mib": peak["minfree"], "shm_peak_mib": peak["shm"], "preemptions": [p0, p1], "errors": errors[:20], "pass": ok}) + "\n")
