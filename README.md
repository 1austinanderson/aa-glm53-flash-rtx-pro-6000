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
patches/                  the overlays — see table above; *.pristine.py / *.orig / glm5next.pristine = unmodified image files
tools/                    quant_drafter_fp8.py, glm_bench.py, glm_stress*.py, glm_smoke.py, pcie_probe.sh
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
