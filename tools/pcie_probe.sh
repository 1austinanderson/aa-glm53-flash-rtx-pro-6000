#!/bin/bash
# PCIe/SM probe during prefill + decode on :12921. Samples nvidia-smi dmon (rx/tx MB/s, SM%) at 1 Hz.
set -u
OUT=results/pcie_probe_$(date +%Y%m%d-%H%M%S).log
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader | head -2
nvidia-smi dmon -i 0,1 -s tu -d 1 -o T > "$OUT.dmon" 2>&1 &
DM=$!; sleep 3
python3 - "$OUT" <<'PY'
import json,random,sys,time,threading,urllib.request
B="http://127.0.0.1:12921/v1"; out=sys.argv[1]
W="margin deposit yield accrual duration basis credit tier lease swap covenant tranche spread ledger".split()
def req(b): return json.load(urllib.request.urlopen(urllib.request.Request(B+"/chat/completions",json.dumps(b).encode(),{"Content-Type":"application/json"}),timeout=900))
marks=[]
def mark(tag): marks.append((time.strftime("%H:%M:%S"), tag)); print(time.strftime("%H:%M:%S"), tag, flush=True)
req({"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"warm"}],"max_tokens":32,"reasoning_effort":"low"})
for rep in range(3):
    p=" ".join(random.choice(W) for _ in range(48000))+"\nSay OK."
    mark(f"prefill64k-{rep} start"); t0=time.time(); r=req({"model":"GLM-5.3-Flash","messages":[{"role":"user","content":p}],"max_tokens":2,"reasoning_effort":"low"}); dt=time.time()-t0
    mark(f"prefill64k-{rep} end {r['usage']['prompt_tokens']} tok {dt:.1f}s = {r['usage']['prompt_tokens']/dt:.0f} tok/s")
    time.sleep(3)
mark("decodeC4 start")
def s(i):
    req({"model":"GLM-5.3-Flash","messages":[{"role":"user","content":f"[{random.random()}] Write a long essay about community bank M&A, part {i}."}],"max_tokens":600,"temperature":0.8,"reasoning_effort":"low"})
ts=[threading.Thread(target=s,args=(i,)) for i in range(4)]; [t.start() for t in ts]; [t.join() for t in ts]
mark("decodeC4 end")
open(out+".marks","w").write("\n".join(f"{a} {b}" for a,b in marks)+"\n")
PY
sleep 2; kill $DM 2>/dev/null; sleep 1
python3 - "$OUT" <<'PY'
import sys,re
out=sys.argv[1]; rows=[]
for l in open(out+".dmon"):
    p=l.split()
    if len(p)<6 or not p[0][0].isdigit(): continue
    # dmon -o T -s tu columns: Time gpu rxpci txpci sm mem enc dec ...
    try: t,g,rx,tx,sm=p[0],int(p[1]),int(p[2]),int(p[3]),int(p[4])
    except: continue
    rows.append((t,g,rx,tx,sm))
marks=[l.split(" ",1) for l in open(out+".marks").read().splitlines()]
def window(tag):
    ss=[t for t,m in marks if m.startswith(tag) and 'start' in m]; ee=[t for t,m in marks if m.startswith(tag) and ' end' in m]
    return list(zip(ss,ee))
for tag in ("prefill64k","decodeC4"):
    for s,e in window(tag):
        sel=[r for r in rows if s<=r[0]<=e]
        for g in (0,1):
            gs=[r for r in sel if r[1]==g]
            if not gs: continue
            rx=[r[2] for r in gs]; tx=[r[3] for r in gs]; sm=[r[4] for r in gs]
            print(f"{tag} {s}-{e} GPU{g}: rx avg {sum(rx)/len(rx)/1000:.2f} GB/s max {max(rx)/1000:.2f} | tx avg {sum(tx)/len(tx)/1000:.2f} max {max(tx)/1000:.2f} | SM avg {sum(sm)/len(sm):.0f}% max {max(sm)}% (n={len(gs)})")
print("raw:", out+".dmon")
PY
