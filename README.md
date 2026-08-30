# GLM-5.3-Flash (320B-A18B, NVFP4/MXFP8) on 2× RTX PRO 6000 Blackwell — how it was made to work

Two RTX PRO 6000 Blackwell Workstation cards (96 GB each, SM120) on PCIe Gen5, TP2, no NVLink. Result: GLM-5.3-Flash at a
**384k-token context window**, DFlash-2 speculative decoding, **~150 tok/s single-stream / ~355 tok/s at 4 streams**,
**~5.1k tok/s prefill**, served over the usual OpenAI-compatible API. Everything below is what it actually took, in order, with the
failure that motivated each step. All patches are small bind-mounted overlays on a public image — nothing is rebuilt from source.
**This repo is the recipe:** `serve.sh` + `patches/` reproduce the serving stack; `tools/` holds the drafter quantizer and the
bench/stress scripts; `results/` the measured rows.

## What the checkpoint is made of
From the `safetensors` headers (258,757 tensors), on one card after the TP2 split it is ~86 GiB of weights:

```
checkpoint (both GPUs)                                              GiB
routed experts (NVFP4)        ██████████████████████████████████  163.27  93.4%
attention KDA+MLA (MXFP8)     █▎                                    5.89  3.4%
embed / lm_head (BF16)        ▎                                     2.36  1.4%
vision encoder (BF16)         ▎                                     1.05  0.6%
shared experts (MXFP8)        ▎                                     1.04  0.6%
dense MLP (BF16)              ▏                                     0.94  0.5%
norms / gates / mHC           ▏                                     0.10  0.1%
sparse indexer (MXFP8)        ▏                                     0.09  0.1%
                                                                 ─────
                                                               174.73
```

93% of the card is MoE experts; the vision tower is a rounding error.

## Requirements (what this was built and measured on)
- 2× NVIDIA RTX PRO 6000 Blackwell Workstation (96 GB, SM120) on PCIe Gen5, no NVLink; driver 595.84; the two cards must be
  `CUDA_VISIBLE_DEVICES=0,1` in `PCI_BUS_ID` order (any third card is left alone).
- Docker 29 + NVIDIA Container Toolkit 1.19 (`--gpus all`). The image is CUDA 13.0 (`cu130`), pulled from Docker Hub (8.7 GB compressed).
- Host RAM: the defaults use a 32 GB host-RAM KV tier inside a 48 GB `/dev/shm` (`--shm-size 48gb`); with `OFFLOAD_GIB=0` the
  `--shm-size` can drop to 32gb. Built on a 177 GB host.
- Disk: ~175 GB checkpoint + ~2 GB drafter (+1.5 GB for the FP8 build) + ~30 GB image unpacked + the compile caches.
- Python with `torch` and `safetensors` on the host only for `tools/quant_drafter_fp8.py` (one-off, CPU is fine).
- Boot ≈ 4.7 min with InstantTensor (first boot longer: FlashInfer JIT + CUDA-graph capture populate `CACHE_DIR`).

## Ingredients
- **Checkpoint:** `local-inference-lab/GLM-5.3-Flash-NVFP4-4p67` (174.75 GiB). ModelOpt MIXED_PRECISION: routed experts
  W4A16-NVFP4 g16, **attention + shared experts + sparse indexer MXFP8 g32**, dense MLP / embeddings / lm_head BF16.
- **Image:** `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1` — upstream vLLM (0.1.dev20051) with the NoPE sparse-MLA
  fix for SM120, FlashInfer 0.6.17, `FLASHINFER_MLA_SPARSE_SM120` backend, `fp8_ds_mla` KV. Derived image
  `glm53-cstech-it:20260828` = that + `pip install instanttensor` (`Dockerfile.instanttensor`).
- **Drafter:** `incoai/GLM-5.3-Flash-DFlash2` (1B) → re-quantized here to `GLM-5.3-Flash-DFlash2-FP8` (`tools/quant_drafter_fp8.py`).
- **Overlays:** `patches/` — each directory has the pristine file from the image plus a `.patch` diff.

## Quick start
```bash
# 1. models (≈177 GB + 1.5 GB) — put them under one dir, e.g. /mnt/raid/models/
#    local-inference-lab/GLM-5.3-Flash-NVFP4-4p67   (HF)
#    incoai/GLM-5.3-Flash-DFlash2                   (HF, bf16 drafter) → quantize:
#    python tools/quant_drafter_fp8.py /mnt/raid/models/incoai/GLM-5.3-Flash-DFlash2 /mnt/raid/models/incoai/GLM-5.3-Flash-DFlash2-FP8
# 2. image
docker pull cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1
docker build -f Dockerfile.instanttensor -t glm53-cstech-it:20260828 .      # optional: 89 s weight load instead of ~3 min
# 3. serve (defaults = the config in "What runs"; DRY_RUN=1 prints the docker command)
MODELS_DIR=/mnt/raid/models ./serve.sh
# knobs: PORT GPUS CTX SEQS MNBT UTIL KV_MEM_BYTES K (0 = no spec decode) OFFLOAD_GIB (0 = off) IMAGE LOAD_FORMAT
# no-drafter variant from the table below:
K=0 OFFLOAD_GIB=0 CTX=262144 SEQS=4 UTIL=0.985 KV_MEM_BYTES=0 ./serve.sh
```
Verified 2026-08-29: a clean boot from this repo with the defaults reproduces the 430,528-token pool and passes `lavd-test` 10/10 at C=6.
`serve.sh` mounts only the overlays a given configuration needs (K=0 skips the DFlash/attention/kv-layout mounts;
OFFLOAD_GIB=0 skips the offload connector and leaves `expandable_segments` on).

## What it took — eight steps

**1. Transport.** TP2 with `NCCL_P2P_DISABLE=1` + `--disable-custom-all-reduce` (host-bounce all-reduce). On this root complex the
peer path costs ~35% prefill; measured on DeepSeek-V4 first, holds for GLM. `CUDA_DEVICE_ORDER=PCI_BUS_ID` so the two cards enumerate first.

**2. Wire MXFP8 attention into the image — `patches/glm5next/`.** The image's `glm5next` hard-codes KDA/MLA/indexer projections as
BF16 (written for "NVFP4 experts + BF16 attention" quants); this quant has them MXFP8, so weight load dies with
`KeyError 'layers.0.self_attn.in_proj_qkvbfg_a.weight_scale'`. The overlay (`glm5next-mixed-precision.patch`):
(a) pass the quant config to the KDA/MLA projections only when it is `modelopt_mixed`, resolved per prefix — unlisted layers stay
unquantized; (b) add `packed_modules_mapping` for `in_proj_qkvbfg_a` / `fused_qkv_a_proj` / `wk_weights_proj` / `gate_up_proj`;
(c) FP8-dequant loaders bail out when the target already owns a `weight_scale`; (d) **the indexer's `wk_weights_proj` must stay
BF16** — its forward reads the raw `.weight` for the fp32 head-gate, so a Marlin-packed weight gives `mat1 and mat2 shapes cannot
be multiplied (4096x4096 and 768x128)` → MXFP8 `wk`/`weights_proj` are dequantized to BF16 on load (`_dequant_mxfp8`, e8m0 scale
= `exp2(s-127)`); (e) steer MXFP8 GEMM to **Marlin** with `VLLM_DISABLED_KERNELS=FlashInferCutedslMxfp8LinearKernel,
FlashInferCutlassMxfp8LinearKernel` — those two gate on `>= sm_100`, which SM120 passes although they are datacenter kernels.
The post-load Marlin repack logs `CUDACachingAllocator ... OOM` warnings; they are retried and are not fatal.
→ Live, no spec decode: C1 107 / C2 163 / C4 282 tok/s, prefill ~5.5k tok/s.

**3. Base tuning.** `--max-num-batched-tokens 4096 → 1024` is free (the Marlin W4A16 MoE is weight-bandwidth-bound) and buys
~30k KV tokens. The MTP head costs 2 GiB/GPU — rejected. `--max-model-len -1` (auto-fit) OOMs because the sparse-indexer buffers
scale from the 1M declared max.

**4. DFlash-2 — `patches/dflash2/`, `cstech-attn/`, `cstech-kv/`.** The image's fork predates upstream **vLLM PR #52816**
(DFlash-2 model + speculator); applied with `patch -F3`, two fuzz hunks hand-fixed (registry entry placement; DFlash2 dispatch
belongs under `method == "dflash"`, not `"dspark"`). `glm5next` gained **`SupportsEagle3` / `EagleModelMixin`** with mHC-aware
aux-hidden-state capture (`hc_contract(layer.hc_post(...), layer.n)` after the tapped layers) — the image had no Eagle3 interface.
Then two KV-layout bugs: `cstech-attn/attention.py` — the drafter's sliding-window spec picked block 16 and was padded from
64 KiB to 2.465 MiB, so one request "needed" 257 blocks (~3 GiB fixed); fix = use the Mamba-padded page as the SW budget and expand
to the largest multiple of 16 that fits. `cstech-kv/kv_cache_utils.py` (+119 lines) — `_get_kv_cache_groups_glm5_next` returned
None for the 5 SW layers → generic hybrid unify → `indexer.k_cache page size is not divisible by the maximum page size`; fix = the
SW layers become an extra slot group **co-owning the MLA slot tensors at disjoint block ids** (zero extra bytes per block; SW block
896 keeps the scheduler LCM at 3,584). `cstech-kv/kv_layout_dryrun.py` reproduces the boot log's layout to the byte on CPU.
→ First DFlash boot: K=3, 114k ctx / 2 seqs → C1 145 / C2 255.

**5. Quantize the drafter.** The bf16 drafter (2.18 GiB) ate KV. `GLM-5.3-Flash-DFlash2-FP8` = compressed-tensors, **MLP-only,
weight-only W8A16** (15 tensors, 0.75B, 1.5 GB). Attention projections MUST stay bf16: `qwen3_dflash.py:_project_context_kv` does
`F.linear(x, qkv_proj.weight[...])` on the raw weight, so fp8 there fails with `expected mat1 and mat2 to have the same dtype`;
the draft `config.json` ignore list must carry `re:.*self_attn.*` (`targets: [Linear]` would otherwise build attention as fp8
layers even with bf16 tensors). A first cut with dynamic A8 activations lost acceptance (1.7 → 1.29 extra tokens/step); W8A16
is lossless. K=2 lost to K=3 (1.09 vs 1.3–1.7 extra tokens/step).

**6. The memory cliff — `cstech-indexer/indexer.py`.** Available KV fell ~5.7 KB per *declared* context token, so 256k + DFlash
would not fit. Root cause: the sparse-indexer K-gather workspace, `get_max_prefill_buffer_size = max_model_len × 40 entries ×
132 B` (1.32 GiB at 256k, paid up front). The overlay sizes it by `max_num_seqs × max_model_len` (41–66 MiB, 8× margin over one
request's compressed context; hard `ValueError` instead of a silent short slice; `VLLM_INDEXER_PREFILL_BUFFER_ENTRIES_PER_TOKEN=40`
restores upstream) → **+1.17 GiB KV**. Also measured: DFlash's fixed per-request cost is 4 KDA groups × (2+K) block ids ≈ 518 MiB
at K=3 — the real context limiter with speculation on.

**7. Host-RAM KV offload + InstantTensor — `cstech-offload/`.** vLLM's native offload (`--kv-offloading-backend native`) asserted at
connector init (`tokens_per_block % tokens_per_hash`) because of the un-hashable KpoolTail group; the scheduler and worker map KV
groups **positionally**, so dropping the group can never work → keep every group in position and make that one inert
(`layer_names=()`). Verified: 60k prompt cold 17.5 s → RAM hit 1.49 s, needle answer byte-identical across the hit. 16 GiB thrashed
under 4×60k; 32 GiB (`--shm-size 48gb`) holds. InstantTensor cuts weight load from 160–220 s to 89 s (total boot ≈ 4.7 min).

**8. The KV claim.** At 384k / 8 seqs an explicit `--kv-cache-memory` of 4.8 GiB OOMs in CUDA-graph capture, 4.3 GiB OOMs on a
380k request, **3.8 GiB** holds (min free 8 MiB only on a full-window request). The weights take ~88 GB per card, so this is what
is left — not a conservative choice.

## What runs
| | |
|---|---|
| KV pool | **430,528 tokens** (3.8 GiB fp8 = 1.09× a 384k request; 8 seqs only if each ≤ ~53k) + 32 GB host-RAM prefix tier |
| Prefill (cold, median of 3) | **4k 4,624 · 16k 4,990 · 64k 5,130 tok/s** — flat; compute-bound (PCIe ~5 of ~26 GB/s) |
| Decode, ctx 0, 4,096 tokens/stream | **C1 150 · C2 237 · C4 355 tok/s** (per-stream 150 / 119 / 89); C8 ~210 — the drafter stops paying at 8 streams, use K=1 or no spec for batch traffic |
| DFlash-2 K=3 acceptance | 1.5–1.9 extra tokens/step, prompt-dependent (structured reasoning drafts best; free-form essays worst) |
| Long-context sanity | `llm-inference-bench --test-profile estonia` (133k prompt): 29/30 correct, p50 4,609 completion tokens |
| No-spec alternative | drop `--speculative-config`, ctx 262144 / seqs 4 / util 0.985 → 357k pool, C1 107 / C4 282 |
| **Superseded** | this is the v1 (2026-08-28/29) build. The shipped configuration is **v2 (2026-08-30)** — 511,617-token pool, ~1.2M-token host-RAM tier, prompt-end resume, 360k window: see the v2 table in the next section |

## v2 (2026-08-30) — dense RAM offload, prompt-end resume, bf16 KDA state, FP8 embedding, quantized drafter

v1 above is the build that made the model serve at all. v2 is a night spent on the two things that cost the most:
the host-RAM tier was storing 30 KB per cached token, and a cached prompt could only be resumed on a 14,336-token
boundary. Same hardware, same image, same overlay method — twelve overlays instead of six.

### What v2 delivers

| | v1 (2026-08-28/29) | v2 (2026-08-30) |
|---|---|---|
| Context window | 384k (`--max-model-len 393216`) | **360k (368640)** — at MNBT 2048 a 385k request OOMs in the sparse-indexer logits workspace; 360k passes with 651 MiB free |
| GPU KV pool | 430,528 tokens = 1.09× the window (3.8 GiB claim) | **511,617 tokens = 1.39× the window** (4.3 GiB claim) |
| Host-RAM tier, 32 GiB | 7.4 GiB per 60k prompt → ~4 prompts, ~260k tokens | **~2.0 GiB per 60k prompt → ~1.2M tokens** (22 × 47.7k prompts measured resident) |
| Cache in total | ~690k tokens | **~1.7M tokens** |
| Prefill, cold | 4k 4,624 · 16k 4,990 · 64k 5,130 tok/s | **4k 5,269 · 16k 5,500 · 64k 5,461 tok/s** (MNBT 1024 → 2048) |
| Decode ctx 0, sustained (45 s cells, 2k gen) | C1 ~150 · C2 ~237 · C4 ~355 · C8 ~210–257 — "the drafter stops paying at 8 streams" | **C1 162 · C2 243 · C4 365 · C8 529** with the quantized drafter (FP8 drafter: 157 · 262 · 360 · **557**) |
| Re-send of a 47.7k-token prompt | 1.07 s, prefix hit 43,008 | **0.5–0.7 s, hit 44,800** |
| Same prompt after GPU eviction (served from RAM) | 3.6 s, hit 28,672 | **0.5 s, hit 44,800**, answer byte-identical to the cold one |
| Prefix-hit granularity | 3,584-token blocks; 14,336 with a coarse Mamba interval | **512-token hash units**, minus a 2,560-token back-off under speculation |
| 8-stream long-answer batch (`lavd-test` C8 × 10) | aborted — 5 of 8 admitted, 8 preemptions in 90 s | **8 EXACT / 2 NEAR / 0 FAIL, 8.0 min wall** |
| `estonia` C8 × 30 (133k prompts) | 29/30, ~19 min | **30/30, 7.6 min** (p50 completion 4,462 tokens) |
| GSM8K-150 @ C8 | not measured | **145/150 = 96.7%**, 3.1 s/problem — recorded as the accuracy anchor |

The pool grew while the window shrank: 511,617 tokens is 1.39 requests' worth of a full 360k context, against v1's 1.09.

### What it took — eight more steps

**v2-1. Persist the JIT caches.** The first request in each new prompt-length bucket after a boot paid 7–22 s. Not the
engine: TileLang (the mHC kernel) and Triton (DFlash draft + rejection sampling) were compiling *during* inference on
first use of a shape. Bind-mount `~/.tilelang` and `~/.triton` alongside the vLLM/FlashInfer caches and the spikes are
paid once, ever.

**v2-2. Make the host-RAM tier dense — `cstech-offload/config.sw-inert.py`, `scheduler.mamba-align.py`,
`cstech-mamba-interval/{interface,mamba_hybrid}.py`.** A 32 GiB tier held about four 60k prompts: 7.4 GiB each,
146 chunks. Two causes, both padding:
(a) **3.3 of every 7.4 GiB was the drafter.** The DFlash sliding-window group has 5 layers holding ~1.8 MB of real
state, stored in 54 MB chunks at block 896. It is speculative state — it can always be recomputed. The offload config
overlay marks that group inert (`layer_names=()`) *in position*, the same trick v1 used for the KpoolTail group, so the
scheduler and worker keep their positional KV-group mapping.
(b) **the 4 KDA groups each wrote a full chunk per attention block.** `--mamba-block-size 14336` snapshots the KDA
temporal state every 4 attention blocks instead of every one. The flag existed and did nothing useful: the platform
ignored it in align mode (fixed in `interface.py` + `mamba_hybrid.py`), and the offload scheduler rounded a
complete-chunk hit to the Mamba align *outside* its convergence loop, which is wrong at 14,336 (fixed in
`scheduler.mamba-align.py`). → **7.4 GiB → ~2.0 GiB per 60k prompt, 38 chunks; ~260k → ~1.0M tokens per 32 GiB.**

**v2-3. Resume at the prompt end, not on a block boundary — `cstech-partial-tail/`, `cstech-coord/`,
`cstech-mamba-interval/sched_scheduler.partial-tail.py`, `cstech-offload/scheduler.partial-tail.py`.** v1's dead-end
list said a coarse Mamba interval works but floors RAM hits at 14,336 tokens. It was worse than that: a 47.7k prompt
re-sent after eviction hit only 28,672 and took 3.6 s. The cause was on the *load* side, not a lost store — the
trailing chunk was dropped for **all** groups, not just the annotated draft groups. Four fixes move together:
- `single_type_kv_cache_manager.py` — the Mamba manager registers the prompt-end partial tail N hash units early, so it
  survives the attention groups' EAGLE drop; the FullAttention drop becomes N units too.
- `kv_cache_coordinator.py` — a non-participating KpoolTail manager no longer switches fine-grained (hash-unit) prefix
  hits off for everyone; the eagle margin is N units.
- `sched_scheduler.partial-tail.py` — a prefill split stops at the prompt-end partial-tail boundary rather than the
  next Mamba block.
- `scheduler.partial-tail.py` — the offload hand-off is generalised to inert groups, non-uniform block sizes
  (3,584 / 14,336) and EAGLE; the load-side trailing-chunk drop is restricted to annotated draft groups; RAM hits get
  the same N-unit eagle cap.

→ a 47.7k re-send hits 44,800 in 0.5–0.7 s on GPU and 0.5 s from RAM, and the answer is byte-identical across the hit.
A 5k prompt, which used to miss entirely below 14k, now hits at the back-off boundary.
**The back-off has to clear the drafter's window.** At a 1-unit back-off one boot produced *zero* accepted drafts: the
drafter's 2,048-token sliding window was stale relative to the resumed prefix. 5 units × 512 tokens = 2,560 > 2,048.
Three CPU dry-runs reproduce all of this without a GPU: `cstech-offload/partial_tail_dryrun.py` (74 checks),
`cstech-offload/kv_offload_dryrun.py`, `cstech-mamba-interval/mamba_interval_dryrun.py`.

**v2-4. MNBT 1024 → 2048, and the window pays for it.** The bigger prefill chunk is worth ~8% prefill, and costs about
0.35 GiB at peak. At 384k / 3.6 GiB a 385k request then OOMs inside the sparse-indexer logits workspace. 360k passes
with 651 MiB free, so `--max-model-len` becomes 368640 — and the pool still ends up larger, in both tokens and
multiples of a full-window request.

**v2-5. K=5 had drifted back in; K=3 is right.** DFlash's fixed per-request cost is 4 KDA groups × (2+K) slot rows:
725 MB at K=5, 518 MB at K=3. On `lavd-test` C8 that is the admission cap, not the KV pool — K=5 admitted 3 streams
(7 EXACT / 2 NEAR / 1 FAIL), K=3 admitted 4 (9/1/0). K=2 (pool +11k, decode −10 to −15% everywhere) and K=4
(−4 to −8% at C2–C8) were both measured and both lose. **K=3 is final; K2, K4 and K5 are all worse.**

**v2-6. bf16 KDA temporal state — `cstech-kda-state/mamba_utils.py` + `VLLM_KDA_STATE_DTYPE=bfloat16`.**
`--mamba-cache-dtype` alone is a no-op here: `kda_state_dtype` hard-codes fp32 for the *temporal* state. Halving it
takes each KDA row from 4 to 2 MB per layer, which halves every pool row (25.9 → 14.8 MB) and lets the platform
re-derive a **2,048-token attention block** (was 3,584), a 512-token drafter block and a **512-token hash unit** — this
is what makes the fine-grained hits of v2-3 fine-grained. Pool 428,931 (1.16×) at the same claim; fixed cost per
request 518 → ~296 MB, which is why all 8 streams now admit. Quality gates on that boot: needle at 47.7k and at 120k
(10/50/90% depth) all correct, re-hit answers identical, acceptance 1.42, `lavd-test` C8 8 EXACT / 2 NEAR / 0 FAIL.
With the fixed cost halved, the KV claim then moved 3.6 → 3.85 GiB (pool 458,216 = 1.24×). The gate for a claim
increase is the memory-stress battery, not `nvidia-smi`: the reported "min free" is the allocator's reservation and
reads identically at 3.6 and 3.85.

**v2-7. FP8 embedding table — `patches/fp8-embed/`, `patches/glm5next-fp8embed/`, `tools/build_embfp8.py`.**
Embeddings + lm_head are 2.36 GiB of BF16 in a checkpoint that is otherwise 4-bit. The embedding is the half that is
safe: it is a lookup, so per-row (per-vocab-entry) scaling has no accumulation to spoil. `build_embfp8.py` derives
`GLM-5.3-Flash-NVFP4-4p67-embfp8` — one rewritten non-expert shard (3.4 GiB), `embed_tokens` in fp8 e4m3 with an fp32
`weight_scale [154880]`, relative symlinks for the other 171 GiB, and quant-config edits (`group_fp8_embed`,
`quantized_layers[…embed_tokens] = FP8_EMBED_ROW`). Two overlays read it: `fp8-embed/modelopt.py` adds
`ModelOptFp8RowEmbeddingMethod`, and `glm5next-fp8embed/` is the v1 model copy with the quant config actually handed to
the embedding. The DFlash drafter shares the embedding module, so it costs no extra VRAM there. CPU verification
(`fp8-embed/verify_fp8_embed.py`): max abs error 0.0025, min row cosine 0.9996, zero rows over the e4m3 bound, TP2
lookups exact. → **0.295 GiB/GPU freed, KV claim 3.85 → 4.1 GiB, pool 487,500 (1.32×)**; lm_head stays BF16.

**v2-8. Quantized DFlash-2 drafter — `patches/dflash2-quantattn/`, `tools/quant_drafter_nvfp4_attnfp8.py`.** v1's
drafter was MLP-only W8A16 because of a real code limitation, documented in v1 step 5: `_project_context_kv` builds the
fused context K/V projection by row-slicing the **raw** `qkv_proj.weight` and calling `F.linear` on it, which dies with
`expected mat1 and mat2 to have the same dtype` the moment attention is quantized — hence `re:.*self_attn.*` in the v1
ignore list. v2 fixes the code instead of the checkpoint: `_qkv_weight_is_raw()` is evaluated once at load time; raw
weights keep v1's single fused GEMM, quantized weights run each layer's **own** `qkv_proj` forward and keep the K/V
*columns* of the output (`F.linear(x, W[q:]) == F.linear(x, W)[…, q:]`). It is a Python bool fixed at load, so it is
CUDA-graph and `torch.compile` safe. `quant_drafter_nvfp4_attnfp8.py` then produces
`GLM-5.3-Flash-DFlash2-NVFP4-attnFP8` (0.98 GiB): MLP → NVFP4 W4A16 g16 (mean relative error 9.4%), attention q/k/v/o →
FP8 W8A16 per-channel (2.2%); `fc`, the candidate selector and the conv kernel projections stay BF16.
`tools/verify_drafter_quant.py` checks dequantization error, which compressed-tensors scheme vLLM resolves per linear,
the patched projection against the original, and the safetensors round-trip — the patched path matches raw weights
to 0.0. Only ~0.25 GiB/GPU is actually saved (the BF16 leftovers are replicated across ranks) → **KV 4.3 GiB,
pool 511,617 (1.39×)**, for −7% decode at C2 and −5% at C8. That trade is stated below, not hidden.

### Why the RAM tier is 3× less dense than VRAM

~30 KB/token in host RAM against 9.7 KB/token in VRAM, and this is structural, not a bug. A CPU chunk is a whole GPU
slot row — 11 tensors, 27 MB per worker — with only one group's layers actually live (MLA fills ~80%, KDA ~70%), and
the 4 KDA groups each store a full chunk every 14,336 tokens, which is 15 KB/token on its own, as much as the MLA KV.
That last term is the dial: with prompt-end resume in place, interior snapshots only serve *different* follow-ups that
share part of a prefix, so a coarser interval is nearly free (see the open list).

### The launch command, as a delta

Everything in v1's `docker run` still applies. What changes:

```diff
- --max-model-len 393216                 # 384k
+ --max-model-len 368640                 # 360k: 385k OOMs in the indexer logits workspace at MNBT 2048
- --max-num-batched-tokens 1024
+ --max-num-batched-tokens 2048
- --kv-cache-memory 4080218931           # 3.8 GiB
+ --kv-cache-memory 4617089843           # 4.3 GiB
+ --mamba-block-size 14336               # KDA state snapshot every 4 attention blocks
+ -e VLLM_EAGLE_DROP_UNITS=5             # 5 x 512-token hash units = 2,560 > the drafter's 2,048 window
+ -e VLLM_KDA_STATE_DTYPE=bfloat16       # temporal state 4 -> 2 MB/layer; block re-derives to 2,048
- <models>/local-inference-lab/GLM-5.3-Flash-NVFP4-4p67
+ <models>/local-inference-lab/GLM-5.3-Flash-NVFP4-4p67-embfp8
- "model":"<models>/incoai/GLM-5.3-Flash-DFlash2-FP8"
+ "model":"<models>/incoai/GLM-5.3-Flash-DFlash2-NVFP4-attnFP8"
  # mounts: glm5next -> glm5next-fp8embed ; dflash2 -> dflash2-quantattn
  #         cstech-offload/config.py -> config.sw-inert.py ; scheduler.py -> scheduler.partial-tail.py
+ #         cstech-mamba-interval/{interface,mamba_hybrid,sched_scheduler.partial-tail}.py
+ #         cstech-coord/kv_cache_coordinator.py, cstech-partial-tail/single_type_kv_cache_manager.py
+ #         cstech-kda-state/mamba_utils.py, fp8-embed/modelopt.py
+ #         $CACHE/tilelang:/root/.tilelang, $CACHE/triton:/root/.triton
```

`serve.sh` defaults to all of it; `DRY_RUN=1 ./serve.sh` prints your exact command. New knobs, all defaulting to the
v2 recipe:

| knob | default | 
|---|---|
| `CTX` / `MNBT` / `KV_MEM_BYTES` | `368640` / `2048` / `4617089843` (4.3 GiB) |
| `MAMBA_BLOCK` | `14336` — KDA snapshot interval. `0` mounts none of the interval/partial-tail/coordinator overlays and drops `--mamba-block-size` (v1 behaviour) |
| `EAGLE_DROP_UNITS` | `5` — hash units of prefix-hit back-off. Empty string = don't set it |
| `KDA_STATE_DTYPE` | `bfloat16` — empty string = the image's fp32 temporal state, and the `cstech-kda-state` overlay is not mounted |
| `EMBED_FP8` | `1` — mounts `glm5next-fp8embed` + `fp8-embed/modelopt.py` and defaults `MODEL` to the `-embfp8` checkpoint. `0` = v1 |
| `DRAFTER_QUANT` | `1` — mounts `dflash2-quantattn` and defaults `DRAFTER` to `…-DFlash2-NVFP4-attnFP8`. `0` = v1's `dflash2` + `…-DFlash2-FP8` |

The v1 configuration is still reachable:
`MAMBA_BLOCK=0 KDA_STATE_DTYPE= EAGLE_DROP_UNITS= EMBED_FP8=0 DRAFTER_QUANT=0 CTX=393216 MNBT=1024 KV_MEM_BYTES=4080218931 ./serve.sh`.

### v2 overlays (`patches/`; pristine copy + diff alongside each, same as v1)

| dir / file | mounted over | why |
|---|---|---|
| `cstech-mamba-interval/{interface,mamba_hybrid}.py` | `platforms/interface.py`, `v1/worker/gpu/model_states/mamba_hybrid.py` | honour `--mamba-block-size` in align mode (KDA snapshot every 4 attention blocks) |
| `cstech-mamba-interval/sched_scheduler.partial-tail.py` | `v1/core/sched/scheduler.py` | a prefill split stops at the prompt-end partial-tail boundary, backed off N hash units under EAGLE |
| `cstech-coord/kv_cache_coordinator.py` | `v1/core/kv_cache_coordinator.py` | a non-participating KpoolTail manager no longer disables fine-grained (hash-unit) prefix hits |
| `cstech-partial-tail/single_type_kv_cache_manager.py` | `v1/core/single_type_kv_cache_manager.py` | Mamba partial tail registered N units early so it survives the attention groups' eagle drop |
| `cstech-offload/config.sw-inert.py` | `kv_connector/v1/offloading/config.py` | v1's inert-in-position trick extended to the drafter's sliding-window group (3.3 of every 7.4 GiB) |
| `cstech-offload/scheduler.partial-tail.py` | `kv_connector/v1/offloading/scheduler.py` | mamba re-round inside the convergence loop; partial-tail hand-off for inert groups, non-uniform block sizes and EAGLE; trailing-chunk drop limited to annotated draft groups |
| `cstech-kda-state/mamba_utils.py` | `model_executor/layers/mamba/mamba_utils.py` | `VLLM_KDA_STATE_DTYPE=bfloat16` for the KDA temporal state (`--mamba-cache-dtype` alone is a no-op) |
| `fp8-embed/modelopt.py` | `model_executor/layers/quantization/modelopt.py` | `ModelOptFp8RowEmbeddingMethod` — per-row FP8 embedding lookup |
| `glm5next-fp8embed/` | `vllm/models/glm5next` | v1's `glm5next/` plus: pass the quant config to the embedding |
| `dflash2-quantattn/` | 9 files | v1's `dflash2/` plus: `_project_context_kv` runs the layer's own quantized forward and slices the K/V columns |

Intermediate steps are kept so the chain is reproducible: `cstech-offload/scheduler.mamba-align.py` (v2-2) is the
baseline for `scheduler.partial-tail.patch` (v2-3), and `cstech-mamba-interval/sched_scheduler.py` is the baseline for
`sched_scheduler.partial-tail.patch`. CPU dry-runs sit next to the files they exercise.

### Building the two derived checkpoints

Both are one-off, CPU-only, and run inside the serving image (they need only `torch` + `safetensors`).

```bash
# 1. FP8 embedding table: <models>/…-NVFP4-4p67 -> <models>/…-NVFP4-4p67-embfp8
#    Output is ~3.4 GiB: one rewritten non-expert shard, edited config.json / hf_quant_config.json /
#    model.safetensors.index.json, and relative symlinks for everything else.
mkdir -p $MODELS_DIR/local-inference-lab/GLM-5.3-Flash-NVFP4-4p67-embfp8
docker run --rm --entrypoint python3 \
  -v $MODELS_DIR/local-inference-lab/GLM-5.3-Flash-NVFP4-4p67:/src:ro \
  -v $MODELS_DIR/local-inference-lab/GLM-5.3-Flash-NVFP4-4p67-embfp8:/dst \
  -v $PWD/tools:/work:ro glm53-cstech-it:20260828 /work/build_embfp8.py
# verify on CPU (max abs err, per-row cosine, e4m3 bound, TP2 lookups):
#   patches/fp8-embed/verify_fp8_embed.py — see its header for the mounts

# 2. Quantized drafter: bf16 incoai/GLM-5.3-Flash-DFlash2 -> …-DFlash2-NVFP4-attnFP8 (0.98 GiB)
python3 tools/quant_drafter_nvfp4_attnfp8.py \
  $MODELS_DIR/incoai/GLM-5.3-Flash-DFlash2 $MODELS_DIR/incoai/GLM-5.3-Flash-DFlash2-NVFP4-attnFP8
python3 tools/verify_drafter_quant.py \
  $MODELS_DIR/incoai/GLM-5.3-Flash-DFlash2 $MODELS_DIR/incoai/GLM-5.3-Flash-DFlash2-NVFP4-attnFP8   # exit 0 = all PASS
```

`build_embfp8.py` writes **relative** symlinks on purpose: the engine sees the tree through the models bind mount, so
an absolute host symlink would dangle inside the container. Set `FP8EMB_SRC_NAME` if your source directory is not named
`GLM-5.3-Flash-NVFP4-4p67`.

### Measuring v2

Everything in v1's *Measuring* section still applies. Added this round:
- Two CPU-only checkpoint gates that need no GPU and exit non-zero on failure:
  `patches/fp8-embed/verify_fp8_embed.py` (embedding dequant error, per-row cosine, e4m3 bound, TP2 lookups) and
  `tools/verify_drafter_quant.py` (drafter dequant error, resolved compressed-tensors schemes, the patched projection
  against the original, safetensors round-trip).
- Three offload/scheduler dry-runs that reproduce the boot-time layout and the hit arithmetic on CPU:
  `patches/cstech-offload/partial_tail_dryrun.py`, `patches/cstech-offload/kv_offload_dryrun.py`,
  `patches/cstech-mamba-interval/mamba_interval_dryrun.py`.
- Batch profiles used as gates, not just benchmarks: `llm-inference-bench --test-profile lavd-test` at C=8 (long
  answers — this is what exposes the DFlash fixed-cost admission cap) and `--test-profile estonia` at C=8 × 30.
- A GSM8K-150 run at C=8 as the accuracy anchor for future kernel and quantization experiments: **145/150** on the
  shipped Marlin config. Re-run it before believing any kernel swap.
- Prefix-hit checks worth automating: re-send a ~47.7k prompt, confirm the reported hit is 44,800 and the answer is
  byte-identical; then force GPU eviction and confirm the RAM hit is the same 44,800 in ~0.5 s.

### Dead ends (v2)

- **MXFP8 A8 kernels on SM120** (letting vLLM pick `FlashInferCutlassMxfp8LinearKernel` instead of forcing Marlin).
  It boots and it really does run W8A8: prefill **+4–5%** (5,473 / 5,724 / 5,793 tok/s) — but decode collapses to
  C1 103.6 / C2 159.2 / C4 270.0 against Marlin's 154.8 / 237.0 / 334.5. The CUTLASS kernel is slow at decode's tiny M
  and the choice is static per layer, so you cannot have both. Reverted to Marlin. A prefill-only lane could use it.
- **`--moe-backend flashinfer_b12x`.** Refused at worker init: `Model sets swiglu_limit=10.0, but the explicitly
  requested moe_backend='flashinfer_b12x' does not apply the SwiGLU clamp`. GLM-5.3-Flash clamps the SiLU gate at ±10
  and the B12X CuTe-DSL MoE epilogue has no clamp. Every suggested alternative needs W4A4 activation scales or MXFP4
  weights, which this checkpoint does not have.
- **The B12X kernel stack as a whole.** Benchmarked here at roughly 2× slower than cstechdev+Marlin on both prefill and
  decode, and its PCIe all-reduce is pull-based — peer *reads*, which is exactly the path this root complex gets wrong.
  Also worth knowing: `b12x` on PyPI and the `b12x` built from source in the vendor image share a version string but
  are different packages; only the source build has the GLM path.
- **A 385k request at MNBT 2048** — OOM in the sparse-indexer logits workspace. Hence the 360k window.
- **A 1-hash-unit eagle back-off** — technically a bigger prefix hit, but the drafter's 2,048-token window goes stale
  and acceptance drops to zero accepted drafts. Back off more than the drafter's window.
- **K = 2, 4, 5.** All measured, all worse than K=3 (K=2 buys 11k pool for 10–15% decode).
- Not a dead end but a correction to v1's list: *"`--mamba-block-size 14336` … partial hash hits are off on this stack
  so RAM hits floor at 14,336 tokens"* — that is what v2-3 fixed. `cutlass_scaled_fp4_mm` is likewise compiled in and
  `cutlass_scaled_mm_supports_fp4(120)` is `True`; the MoE picks Marlin because the experts are W4A16 with no
  activation scales, not because the cards lack FP4.

### The trades, stated plainly

- **The bf16 recurrent state is a quality trade.** Halving the KDA temporal state is not free by construction; what can
  be said is that it was gated — needle at 47.7k and 120k (10/50/90% depth) correct, re-hit answers byte-identical,
  `lavd-test` C8 8/2/0, GSM8K-150 145/150 — and nothing regressed on those. Set `KDA_STATE_DTYPE=` to get fp32 back at
  a cost of ~83k pool tokens and double the per-request fixed cost.
- **The quantized drafter costs decode to buy pool.** +24k pool tokens for −7% at C2 and −5% at C8 (flat at C1/C4).
  The quantization itself is faithful — the patched projection matches raw weights to 0.0 and acceptance is unchanged
  in distribution — the loss is the extra small GEMMs. `DRAFTER_QUANT=0` reverts to the v1 FP8 drafter and a
  487,500-token pool. Reasonable people can pick either.
- **The 360k window is a real reduction** from v1's 384k, bought back many times over in pool size and RAM density.
  If you need the full 384k, keep `MNBT=1024`.

### Open list

- **Coarser KDA interval is the cheapest remaining RAM win.** 14,336 → 28,672 takes the tier to ~1.5M tokens (worst
  mid-prefix recompute ~28k tokens ≈ 5.6 s), 57,344 → ~1.9M, prompt-tail-only → ~2.2M. `OFFLOAD_GIB=64` doubles any of
  them if the host has the RAM (raise `SHM` to at least offload + 16).
- **GPU levers left**, at roughly 1 GiB ≈ 110k pool tokens: drafter off (+130k, decode −30%), K=1 for batch lanes,
  vision-tower skip (~+55k, untested), FP8 requant of lm_head and the dense MLP (~+95k, offline), a 300k window (+35k).
- **Drafter leftovers**: `fc` is 160 MiB replicated → 80 MiB/GPU at FP8, but it is the most acceptance-sensitive tensor
  in the model; the conv kernel projections are another 40 MiB/GPU and need quant-config plumbing.
- **Batch lane shape.** Admission still oscillates (6 → 3 → 5 → 2) when eight ~35k-token answers plus fixed cost exceed
  the pool. The fix is `--max-num-seqs` matched to expected answer length (≈5 for `lavd`-class work), not more
  admission.
- **True W4A4 prefill** needs a checkpoint with calibrated per-expert activation scales — a re-quantization job, not a
  kernel or a flag. Every published W4A4 export of this model is 6.5–16 GiB larger than this one (they leave attention
  BF16) and does not fit two 96 GB cards.
- **B12X PCIe push path** (`B12X_PCIE_TP2_REMOTE_PUSH` / `B12X_PCIE_ONESHOT_PUSH`) has never been tested standalone
  here; the pull path is known broken on this root complex. A FlashInfer B12X MoE with the SwiGLU clamp added to its
  epilogue is the other open thread.

*Everything below is the v1 (2026-08-27/29) write-up — the full launch command, the overlay table, v1's dead ends,
measuring and layout. v2 adds to it; it does not replace it.*

## The launch command
This is what `serve.sh` runs with the defaults (`DRY_RUN=1 ./serve.sh` prints your exact version). Every `-v … :ro` into
`dist-packages/vllm` is one of the overlays below; `$P` = this repo's `patches/`, `$VL` = the image's vLLM package dir.
```bash
docker run --rm --name glm53-flash --gpus all --network host --ipc host --init --shm-size 48gb \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 --ulimit stack=67108864 \
  -v /mnt/raid/models:/root/models:ro \
  -v $CACHE/vllm:/root/.cache/vllm -v $CACHE/flashinfer:/root/.cache/flashinfer -v $CACHE/tmp:/tmp \
  -v $P/glm5next:$VL/models/glm5next:ro \
  -v $P/cstech-indexer/indexer.py:$VL/v1/attention/backends/mla/indexer.py:ro \
  -v $P/dflash2/vllm/model_executor/models/qwen3_dflash2.py:$VL/model_executor/models/qwen3_dflash2.py:ro \
  -v $P/dflash2/vllm/model_executor/models/qwen3_dflash.py:$VL/model_executor/models/qwen3_dflash.py:ro \
  -v $P/dflash2/vllm/model_executor/models/registry.py:$VL/model_executor/models/registry.py:ro \
  -v $P/dflash2/vllm/model_executor/layers/logits_processor.py:$VL/model_executor/layers/logits_processor.py:ro \
  -v $P/dflash2/vllm/config/vllm.py:$VL/config/vllm.py:ro \
  -v $P/dflash2/vllm/v1/worker/gpu/sample/gumbel.py:$VL/v1/worker/gpu/sample/gumbel.py:ro \
  -v $P/dflash2/vllm/v1/worker/gpu/spec_decode/__init__.py:$VL/v1/worker/gpu/spec_decode/__init__.py:ro \
  -v $P/dflash2/vllm/v1/worker/gpu/spec_decode/speculator.py:$VL/v1/worker/gpu/spec_decode/speculator.py:ro \
  -v $P/dflash2/vllm/v1/worker/gpu/spec_decode/dflash2:$VL/v1/worker/gpu/spec_decode/dflash2:ro \
  -v $P/cstech-attn/attention.py:$VL/model_executor/layers/attention/attention.py:ro \
  -v $P/cstech-kv/kv_cache_utils.py:$VL/v1/core/kv_cache_utils.py:ro \
  -v $P/cstech-offload/config.py:$VL/distributed/kv_transfer/kv_connector/v1/offloading/config.py:ro \
  -v $P/cstech-offload/scheduler.py:$VL/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:ro \
  -e VLLM_DISABLED_KERNELS=FlashInferCutedslMxfp8LinearKernel,FlashInferCutlassMxfp8LinearKernel \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=0,1 -e NCCL_P2P_DISABLE=1 -e PYTHONHASHSEED=0 \
  -e INSTANTTENSOR_BACKEND=BUFFERED -e SAFETENSORS_FAST_GPU=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  -e VLLM_LOGGING_LEVEL=INFO \
  glm53-cstech-it:20260828 /root/models/local-inference-lab/GLM-5.3-Flash-NVFP4-4p67 \
  --served-model-name GLM-5.3-Flash --host 0.0.0.0 --port 8000 --tensor-parallel-size 2 \
  --max-num-batched-tokens 1024 --max-num-seqs 8 --max-model-len 393216 --gpu-memory-utilization 0.988 \
  --kv-cache-dtype fp8 --load-format instanttensor --enable-prefix-caching --no-enable-flashinfer-autotune \
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45 --kv-cache-memory 4080218931 \
  --disable-custom-all-reduce \
  --speculative-config '{"method":"dflash","model":"/root/models/incoai/GLM-5.3-Flash-DFlash2-FP8","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"standard","attention_backend":"FLASH_ATTN","kv_cache_dtype":"auto"}' \
  --kv-offloading-size 32 --kv-offloading-backend native
```
`VL=/usr/local/lib/python3.12/dist-packages/vllm` in this image. Without InstantTensor: use the `cstechdev/vllm:…-r1` image and
`--load-format auto` (+~2 min boot). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` is required by the host-RAM offload
(stable CUDA registrations); without offload, `True` is fine.

## Overlays (`patches/`; pristine copy + diff alongside each)
| dir | mounted over | why |
|---|---|---|
| `glm5next/` | `vllm/models/glm5next` | MXFP8 mixed precision per layer (step 2) + `SupportsEagle3` aux capture (step 4) |
| `dflash2/` | 9 files | upstream PR #52816 (DFlash-2), two fuzz hunks hand-fixed — README inside |
| `cstech-attn/attention.py` | `model_executor/layers/attention/attention.py` | sliding-window spec block padding (257 blocks/request) |
| `cstech-kv/kv_cache_utils.py` | `v1/core/kv_cache_utils.py` | drafter SW group co-owns the MLA slot tensors |
| `cstech-indexer/indexer.py` | `v1/attention/backends/mla/indexer.py` | K-gather workspace sized by seqs × ctx (+1.17 GiB KV) |
| `cstech-offload/{config,scheduler}.py` | `kv_connector/v1/offloading/` | inert-in-position KV groups for host-RAM offload |

The ten overlays v2 adds are tabled in the v2 section above.

## Dead ends (so you don't repeat them)
MTP head (2 GiB/GPU) · `--max-model-len -1` ·
`--gpu-memory-utilization` above 0.988 (no headroom under load) · `--kv-cache-memory` ≥ 4.3 GiB at 384k/8 · packing KDA state
pages into MLA sub-slots (181 MiB ceiling — 4 KDA groups × (2+K) block ids is the real cost) · the "one spec block per K" Mamba
shortcut (the KDA backend needs one state column per speculative token; silent corruption) · `--mamba-block-size 14336` for
offload RAM efficiency (works, 1.5× less RAM, but partial hash hits are off on this stack so RAM hits floor at 14,336 tokens).
Two latent offload defects are documented in `cstech-offload/` (hit rounded to the Mamba align before the MLA convergence loop;
`cache_config.block_size` used for the Mamba stride) — unmounted candidates alongside. The refuted / diagnostic overlays from the
build are not in this repo.

## Measuring
`tools/glm_bench.py <label>` (cold prefill 4k/16k/64k + decode C1/C2/C4 → `results/bench_results.jsonl`), `tools/glm_stress.py` /
`tools/glm_stress_heavy.py` (→ `results/stress_results.jsonl`), `tools/pcie_probe.sh` (nvidia-smi dmon rx/tx/SM during prefill),
`tools/glm_smoke.py` (needle + acceptance check). All default to `http://127.0.0.1:8000` and take a base URL as an argument. External:
[`local-inference-lab/llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench)
(`--test-profile estonia`; `--skip-prefill --contexts 0 --concurrency 1,2,4 --max-tokens 4096`; `--prefill-only --prefill-contexts 4k,16k,64k`).
Speculation health: `curl :8000/metrics | grep spec_decode`. KV pool: `docker logs glm53-flash 2>&1 | grep "GPU KV cache size"`.
`results/bench_results.jsonl` carries every sweep row from the build with its config label (image / spec / ctx / seqs / MNBT / util).

## Layout
```
serve.sh                  parameterized launcher (env knobs; DRY_RUN=1 prints the docker command)
Dockerfile.instanttensor  cstechdev image + InstantTensor loader
patches/                  the overlays — see both tables above; *.pristine.py / *.orig / glm5next.pristine = unmodified image files
tools/                    v1: quant_drafter_fp8.py, glm_bench.py, glm_stress*.py, glm_smoke.py, pcie_probe.sh
                          v2: build_embfp8.py, quant_drafter_nvfp4_attnfp8.py, verify_drafter_quant.py
results/                  bench_results.jsonl, stress_results.jsonl
```
Overlays are pinned to `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1` (vLLM 0.1.dev20051). A different image
build needs the diffs re-applied against its own files — that is what the pristine copies are for.

## License
Apache-2.0 (see `LICENSE`, `NOTICE`). `patches/` are modified vLLM files. Model weights are not included: the checkpoint and
drafter carry their own licenses (the DFlash-2 drafter is CC BY-NC-ND).

## Credits
`cstechdev` for the SM120 NoPE sparse-MLA image · `local-inference-lab` (voipmonitor) for the NVFP4/MXFP8 quant, InstantTensor
and llm-inference-bench · `incoai` for the DFlash-2 drafter · vLLM PR #52816 authors. Built 2026-08-27/28 by Claude (Fable 5)
in a Claude Code session with Austin driving; every number above was measured on this box.
