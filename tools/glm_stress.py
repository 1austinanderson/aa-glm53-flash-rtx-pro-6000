#!/usr/bin/env python3
"""Memory-stress battery for GLM-5.3-Flash: shapes the startup profiler never sees.
Runs C4x60k prefill burst, C4x2k thinking decode, logprobs/top-k, an image request,
and a mixed burst, while sampling nvidia-smi peak used + vLLM metrics.
Usage: glm_stress.py <label> [base] [ctx]"""
import base64, json, random, subprocess, sys, threading, time, urllib.request, zlib, struct
label = sys.argv[1] if len(sys.argv) > 1 else "run"
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000/v1"
CTX = int(sys.argv[3]) if len(sys.argv) > 3 else 131072
MODEL = json.load(urllib.request.urlopen(BASE + "/models"))["data"][0]["id"]
WORDS = "margin deposit yield accrual duration basis credit tier lease swap covenant tranche spread ledger".split()
peak = {"used": 0, "samples": 0, "stop": False}
def sampler():
    while not peak["stop"]:
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
            used = max(int(x) for x in out.split()[:2]); peak["used"] = max(peak["used"], used); peak["samples"] += 1
        except Exception: pass
        time.sleep(0.5)
threading.Thread(target=sampler, daemon=True).start()
def req(body, t=1800):
    r = urllib.request.urlopen(urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=t)
    return json.load(r)
def metrics():
    try:
        m = urllib.request.urlopen(BASE.replace("/v1", "/metrics"), timeout=10).read().decode()
        want = {}
        for line in m.splitlines():
            for k in ("vllm:num_preemptions_total", "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "spec_decode_num_accepted_tokens_total", "spec_decode_num_draft_tokens_total", "spec_decode_num_drafts_total"):
                if line.startswith(k) or (k in line and not line.startswith("#")):
                    want[k] = line.split()[-1]
        return want
    except Exception as e: return {"metrics_error": str(e)}
def png_1x1_red():
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    raw = b"\x00" + bytes([255, 0, 0]) * 64
    rows = raw * 64
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
results = {"label": label, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "steps": []}
def step(name, fn):
    t0 = time.time(); before = peak["used"]
    try: out = fn(); ok = True
    except Exception as e: out = f"{type(e).__name__}: {str(e)[:300]}"; ok = False
    rec = {"step": name, "ok": ok, "wall_s": round(time.time() - t0, 1), "peak_used_mib": peak["used"], "detail": out, "metrics": metrics()}
    results["steps"].append(rec); print(f"[{label}] {name}: {'OK' if ok else 'FAIL'} {rec['wall_s']}s peak_used={peak['used']} MiB {'' if ok else out}", flush=True)
    return ok
def prompt(n_tok): return " ".join(random.choice(WORDS) for _ in range(int(n_tok / 1.33))) + "\nSummarize in one sentence."
def par(fn, n):
    outs, errs = [], []
    def w(i):
        try: outs.append(fn(i))
        except Exception as e: errs.append(f"{type(e).__name__}: {str(e)[:200]}")
    ts = [threading.Thread(target=w, args=(i,)) for i in range(n)]; [t.start() for t in ts]; [t.join() for t in ts]
    if errs: raise RuntimeError("; ".join(errs[:2]))
    return outs
long_n = min(60000, CTX - 4096)
step("warm", lambda: req({"model": MODEL, "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 8, "reasoning_effort": "low"})["usage"]["completion_tokens"])
step(f"C4x{long_n//1000}k prefill burst", lambda: [r["usage"]["prompt_tokens"] for r in par(lambda i: req({"model": MODEL, "messages": [{"role": "user", "content": f"[{i}] " + prompt(long_n)}], "max_tokens": 16, "reasoning_effort": "low"}), 4)])
step("C4x2k thinking decode (effort max)", lambda: [r["usage"]["completion_tokens"] for r in par(lambda i: req({"model": MODEL, "messages": [{"role": "user", "content": f"[{i}] Prove that the sum of the first n odd numbers is n^2, then discuss three generalizations."}], "max_tokens": 2048, "temperature": 1.0, "top_p": 0.95}), 4)])
step("logprobs top-20 x C2", lambda: [len(r["choices"][0].get("logprobs", {}).get("content", [])) for r in par(lambda i: req({"model": MODEL, "messages": [{"role": "user", "content": f"[{i}] List ten US bank holding companies."}], "max_tokens": 256, "logprobs": True, "top_logprobs": 20, "reasoning_effort": "low"}), 2)])
img = base64.b64encode(png_1x1_red()).decode()
step("image request (vision tower)", lambda: req({"model": MODEL, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}, {"type": "text", "text": "What color is this image? One word."}]}], "max_tokens": 64, "reasoning_effort": "low"})["choices"][0]["message"].get("content"))
step("mixed burst: 2x40k prefill + 2x1k decode + 1 image", lambda: par(lambda i: req({"model": MODEL, "messages": [{"role": "user", "content": ([{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}, {"type": "text", "text": "Describe."}] if i == 4 else f"[{i}] " + (prompt(min(40000, CTX - 4096)) if i < 2 else "Write 800 words on deposit betas."))}], "max_tokens": 1024 if 2 <= i < 4 else 32, "reasoning_effort": "low"})["usage"]["completion_tokens"], 5))
peak["stop"] = True
total = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True).split()[0])
results["peak_used_mib"] = peak["used"]; results["device_total_mib"] = total; results["min_free_mib"] = total - peak["used"]
results["pass"] = all(s["ok"] for s in results["steps"])
print(f"[{label}] RESULT {'PASS' if results['pass'] else 'FAIL'} peak_used={peak['used']} / {total} MiB (min free {total - peak['used']} MiB)", flush=True)
with open("results/stress_results.jsonl", "a") as f: f.write(json.dumps(results) + "\n")
