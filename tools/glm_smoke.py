#!/usr/bin/env python3
"""Smoke battery for GLM-5.3-Flash on :12921 — reasoning split, tool call, decode speed."""
import json, sys, time, urllib.request
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:12921"
def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); r = json.load(urllib.request.urlopen(req, timeout=timeout)); return r, time.time() - t0
models = json.load(urllib.request.urlopen(BASE + "/v1/models"))["data"]; name = models[0]["id"]; print("model:", name)
# 1. plain chat w/ reasoning
r, dt = post("/v1/chat/completions", {"model": name, "messages": [{"role": "user", "content": "What is a bank's net interest margin? Two sentences."}], "max_tokens": 600, "temperature": 0.7})
m = r["choices"][0]["message"]; u = r["usage"]
print(f"[chat] {dt:.1f}s tokens out={u['completion_tokens']} ({u['completion_tokens']/dt:.1f} tok/s incl. prefill) reasoning_len={len(m.get('reasoning_content') or m.get('reasoning') or '')} finish={r['choices'][0]['finish_reason']}")
print("  content:", (m.get("content") or "")[:300].replace("\n", " "))
print("  reasoning keys present:", [k for k in ("reasoning_content", "reasoning") if m.get(k)])
# 2. thinking off
r, dt = post("/v1/chat/completions", {"model": name, "messages": [{"role": "user", "content": "Reply with exactly: OK"}], "max_tokens": 20, "chat_template_kwargs": {"enable_thinking": False}})
m = r["choices"][0]["message"]; print(f"[nothink] {dt:.1f}s content={m.get('content')!r} reasoning={bool(m.get('reasoning_content') or m.get('reasoning'))}")
# 3. tool call
tools = [{"type": "function", "function": {"name": "get_bank_metric", "description": "Fetch a metric for a bank", "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}, "metric": {"type": "string"}}, "required": ["ticker", "metric"]}}}]
r, dt = post("/v1/chat/completions", {"model": name, "messages": [{"role": "user", "content": "Get the NIM for PNFP using the tool."}], "tools": tools, "max_tokens": 800, "chat_template_kwargs": {"enable_thinking": False}})
m = r["choices"][0]["message"]; tc = m.get("tool_calls") or []
print(f"[tool] {dt:.1f}s finish={r['choices'][0]['finish_reason']} tool_calls={[ (t['function']['name'], t['function']['arguments']) for t in tc]}")
# 4. decode speed, single stream, thinking off, 300 tokens
r, dt = post("/v1/chat/completions", {"model": name, "messages": [{"role": "user", "content": "Write a 400-word plain-prose explanation of CECL for a bank board."}], "max_tokens": 300, "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}})
u = r["usage"]; print(f"[decode C1] {u['completion_tokens']} tok in {dt:.1f}s = {u['completion_tokens']/dt:.1f} tok/s")
# 5. two concurrent streams
import threading
res = []
def worker(i):
    r, dt = post("/v1/chat/completions", {"model": name, "messages": [{"role": "user", "content": f"Write a 400-word plain-prose note on deposit betas, variant {i}."}], "max_tokens": 300, "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}})
    res.append((r["usage"]["completion_tokens"], dt))
ts = [threading.Thread(target=worker, args=(i,)) for i in range(2)]; t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]; wall = time.time() - t0
print(f"[decode C2] aggregate {sum(n for n,_ in res)/wall:.1f} tok/s over {wall:.1f}s; per-stream {[round(n/d,1) for n,d in res]}")
