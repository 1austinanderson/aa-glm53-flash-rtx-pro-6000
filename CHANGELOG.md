# Changelog

All measurements are on the same box: 2× RTX PRO 6000 Blackwell Workstation (96 GB, SM120), PCIe Gen5, TP2, no NVLink.

## v2 — 2026-08-30

Dense host-RAM offload, prompt-end resume, bf16 KDA state, FP8 embedding table, quantized DFlash-2 drafter.

**Results (v1 → v2)**
- GPU KV pool **430,528 → 511,617 tokens** (1.09× → 1.39× a full-window request); KV claim 3.8 → 4.3 GiB.
- Host-RAM tier, 32 GiB: **~260k → ~1.2M tokens** (7.4 → ~2.0 GiB per 60k-token prompt). Total cache ~690k → ~1.7M.
- Context window 384k → **360k** (`--max-model-len 368640`) — at MNBT 2048 a 385k request OOMs in the sparse-indexer
  logits workspace.
- Prefill (cold) 4,624 / 4,990 / 5,130 → **5,269 / 5,500 / 5,461 tok/s** at 4k / 16k / 64k.
- Decode ctx 0, sustained: C8 goes from ~210–257 tok/s (the drafter stopped paying at 8 streams) to **529** with the
  quantized drafter, **557** with the FP8 one; C1/C2/C4 162 / 243 / 365.
- Re-sending a 47.7k-token prompt: 1.07 s → **0.5–0.7 s** on GPU, 3.6 s → **0.5 s** from host RAM, answer
  byte-identical; prefix hits go from 3,584-token blocks (14,336 with a coarse Mamba interval) to **512-token hash
  units** minus a 2,560-token back-off.
- `lavd-test` C8 × 10: aborted (5/8 admitted, 8 preemptions in 90 s) → **8 EXACT / 2 NEAR / 0 FAIL in 8.0 min**.
  `estonia` C8 × 30: 29/30 in ~19 min → **30/30 in 7.6 min**. New accuracy anchor: GSM8K-150 @C8 **145/150 (96.7%)**.

**Added**
- `patches/cstech-mamba-interval/` — `--mamba-block-size` honoured in align mode (KDA snapshot every 4 attention
  blocks) and a prefill split that stops at the prompt-end partial-tail boundary.
- `patches/cstech-coord/` — a non-participating KpoolTail manager no longer disables fine-grained prefix hits.
- `patches/cstech-partial-tail/` — Mamba partial tail registered early enough to survive the EAGLE drop.
- `patches/cstech-offload/config.sw-inert.py`, `scheduler.mamba-align.py`, `scheduler.partial-tail.py`,
  `partial_tail_dryrun.py` — drafter sliding-window group inert for offload; mamba re-round inside the convergence
  loop; partial-tail hand-off for inert groups, non-uniform block sizes and EAGLE; trailing-chunk drop limited to
  annotated draft groups.
- `patches/cstech-kda-state/` — `VLLM_KDA_STATE_DTYPE=bfloat16` for the KDA temporal state.
- `patches/fp8-embed/` + `patches/glm5next-fp8embed/` — per-row FP8 embedding table (`ModelOptFp8RowEmbeddingMethod`)
  and the model copy that passes the quant config to the embedding; `verify_fp8_embed.py` is the CPU gate.
- `patches/dflash2-quantattn/` — `_project_context_kv` runs the layer's own quantized forward and slices the K/V
  columns, so the drafter's attention can be quantized at all.
- `tools/build_embfp8.py`, `tools/quant_drafter_nvfp4_attnfp8.py`, `tools/verify_drafter_quant.py` — build and check
  the two derived checkpoints.

**Changed**
- `serve.sh` defaults to the v2 recipe and gains `MAMBA_BLOCK`, `EAGLE_DROP_UNITS`, `KDA_STATE_DTYPE`, `EMBED_FP8`,
  `DRAFTER_QUANT`; `CTX` 393216 → 368640, `MNBT` 1024 → 2048, `KV_MEM_BYTES` 3.8 → 4.3 GiB; TileLang and Triton JIT
  caches are now mounted (kernels were compiling during inference, 5–13 s spikes on first use of a shape).
  The v1 configuration is still reachable through those knobs.
- `README.md` gains the v2 section; v1's history is unchanged apart from a pointer row in its results table.

**Reverted / not shipped**
- MXFP8 A8 kernels on SM120 (`FlashInferCutlassMxfp8LinearKernel`): +4–5% prefill, −33% decode. Marlin stays.
- `--moe-backend flashinfer_b12x`: refused at worker init — the model sets `swiglu_limit=10.0` and the B12X MoE
  epilogue applies no SwiGLU clamp.
- K = 2, 4 and 5 (all measured, all worse than K=3).

## v1 — 2026-08-27/29

First working configuration: GLM-5.3-Flash (320B-A18B, NVFP4/MXFP8) at a 384k-token window with DFlash-2 speculative
decoding, 430,528-token KV pool, ~150 tok/s single-stream / ~355 at 4 streams, ~5.1k tok/s prefill, plus a 32 GB
host-RAM KV tier. Eight steps: host-bounce transport, MXFP8 mixed-precision attention (`patches/glm5next/`), base
tuning, DFlash-2 from upstream PR #52816 with two KV-layout fixes (`patches/dflash2/`, `cstech-attn/`, `cstech-kv/`),
an FP8 drafter, the sparse-indexer workspace cap (`cstech-indexer/`, +1.17 GiB KV), inert-in-position KV groups for
host-RAM offload (`cstech-offload/`) with InstantTensor, and the 3.8 GiB KV claim. See the README for the full
write-up.
