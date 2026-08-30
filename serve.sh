#!/usr/bin/env bash
# GLM-5.3-Flash (NVFP4/MXFP8) on 2x RTX PRO 6000 Blackwell — vLLM TP2 with DFlash-2 speculative decoding.
# Standalone launcher: every path/knob is an env var. Run from anywhere.
#   MODELS_DIR   host dir holding the checkpoints (see "models" below)
#   REPO_DIR     this repo (patches/ is bind-mounted over the image's vLLM)
#   CACHE_DIR    host dir for vLLM/FlashInfer/TileLang/Triton compile caches and /tmp (persist across boots)
#   IMAGE        glm53-cstech-it:20260828 (build: docker build -f Dockerfile.instanttensor -t glm53-cstech-it:20260828 .)
#                or cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1 with LOAD_FORMAT=auto
# Defaults = the v2 (2026-08-30) recipe: 360k window, MNBT 2048, KV 4.3 GiB, mamba block 14336,
# bf16 KDA state, 5-hash-unit eagle back-off, FP8-embedding checkpoint, quantized DFlash-2 drafter.
# Optional: PORT, GPUS, CTX, SEQS, MNBT, UTIL, KV_MEM_BYTES, K (0 = no speculative decoding),
#           OFFLOAD_GIB (0 = off), SHM, MAMBA_BLOCK (0 = v1 behaviour), EAGLE_DROP_UNITS,
#           KDA_STATE_DTYPE ("" = fp32/v1), EMBED_FP8 (0 = BF16 embedding), DRAFTER_QUANT (0 = FP8 drafter),
#           MODEL, DRAFTER, DRY_RUN=1
# models: EMBED_FP8=1 wants local-inference-lab/GLM-5.3-Flash-NVFP4-4p67-embfp8 (tools/build_embfp8.py derives it
#         from …-4p67); DRAFTER_QUANT=1 wants incoai/GLM-5.3-Flash-DFlash2-NVFP4-attnFP8
#         (tools/quant_drafter_nvfp4_attnfp8.py derives it from the bf16 incoai/GLM-5.3-Flash-DFlash2).
set -euo pipefail
MODELS_DIR="${MODELS_DIR:-/mnt/raid/models}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/cache}"
IMAGE="${IMAGE:-glm53-cstech-it:20260828}"
LOAD_FORMAT="${LOAD_FORMAT:-instanttensor}"
PORT="${PORT:-8000}"; GPUS="${GPUS:-0,1}"; CTX="${CTX:-368640}"; SEQS="${SEQS:-8}"; MNBT="${MNBT:-2048}"; UTIL="${UTIL:-0.988}"
KV_MEM_BYTES="${KV_MEM_BYTES:-4617089843}"   # 4.3 GiB explicit claim; 0 = let vLLM profile from UTIL
K="${K:-3}"
OFFLOAD_GIB="${OFFLOAD_GIB:-32}"; SHM="${SHM:-48gb}"
MAMBA_BLOCK="${MAMBA_BLOCK:-14336}"          # KDA state snapshot interval; 0 = the image default (one per attention block)
EAGLE_DROP_UNITS="${EAGLE_DROP_UNITS-5}"    # prefix-hit back-off in hash units (512 tokens each with bf16 state)
KDA_STATE_DTYPE="${KDA_STATE_DTYPE-bfloat16}"   # "" = the image's hard-coded fp32 temporal state
EMBED_FP8="${EMBED_FP8:-1}"                  # 1 = per-row FP8 embedding table (needs the -embfp8 checkpoint)
DRAFTER_QUANT="${DRAFTER_QUANT:-1}"          # 1 = NVFP4-MLP / FP8-attention drafter (needs the quantized-attention overlay)
if [ "$EMBED_FP8" != "0" ]; then
  MODEL="${MODEL:-/root/models/local-inference-lab/GLM-5.3-Flash-NVFP4-4p67-embfp8}"
else
  MODEL="${MODEL:-/root/models/local-inference-lab/GLM-5.3-Flash-NVFP4-4p67}"
fi
if [ "$DRAFTER_QUANT" != "0" ]; then
  DRAFTER="${DRAFTER:-/root/models/incoai/GLM-5.3-Flash-DFlash2-NVFP4-attnFP8}"; DFLASH_DIR="dflash2-quantattn"
else
  DRAFTER="${DRAFTER:-/root/models/incoai/GLM-5.3-Flash-DFlash2-FP8}"; DFLASH_DIR="dflash2"
fi
VL=/usr/local/lib/python3.12/dist-packages/vllm
P="$REPO_DIR/patches"
mkdir -p "$CACHE_DIR"/{vllm,flashinfer,tmp,tilelang,triton}

MOUNTS=(
  -v "$MODELS_DIR:/root/models:ro"
  -v "$CACHE_DIR/vllm:/root/.cache/vllm" -v "$CACHE_DIR/flashinfer:/root/.cache/flashinfer" -v "$CACHE_DIR/tmp:/tmp"
  # v2 step 1: TileLang + Triton JIT caches — without these, kernels compile DURING the first
  # inference of each new shape after a boot (5-13 s latency spikes)
  -v "$CACHE_DIR/tilelang:/root/.tilelang" -v "$CACHE_DIR/triton:/root/.triton"
  # step 6: sparse-indexer K-gather workspace sized by seqs x ctx
  -v "$P/cstech-indexer/indexer.py:$VL/v1/attention/backends/mla/indexer.py:ro"
)
# step 2: MXFP8 mixed-precision attention (+ SupportsEagle3 for DFlash).
# v2 step 5: EMBED_FP8 swaps in the copy that also passes the quant config to the embedding.
if [ "$EMBED_FP8" != "0" ]; then
  MOUNTS+=(-v "$P/glm5next-fp8embed:$VL/models/glm5next:ro")
  MOUNTS+=(-v "$P/fp8-embed/modelopt.py:$VL/model_executor/layers/quantization/modelopt.py:ro")
else
  MOUNTS+=(-v "$P/glm5next:$VL/models/glm5next:ro")
fi
ARGS=(
  --served-model-name GLM-5.3-Flash --host 0.0.0.0 --port "$PORT" --tensor-parallel-size 2
  --max-num-batched-tokens "$MNBT" --max-num-seqs "$SEQS" --max-model-len "$CTX" --gpu-memory-utilization "$UTIL"
  --kv-cache-dtype fp8 --load-format "$LOAD_FORMAT" --enable-prefix-caching --no-enable-flashinfer-autotune
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45 --disable-custom-all-reduce
)
[ "$KV_MEM_BYTES" != "0" ] && ARGS+=(--kv-cache-memory "$KV_MEM_BYTES")
ALLOC_CONF="expandable_segments:True"
ENVS=()
if [ "$K" != "0" ]; then
  # step 4: DFlash-2 (upstream PR #52816) + the two KV-layout fixes it needs on this image
  for f in model_executor/models/qwen3_dflash2.py model_executor/models/qwen3_dflash.py model_executor/models/registry.py \
           model_executor/layers/logits_processor.py config/vllm.py v1/worker/gpu/sample/gumbel.py \
           v1/worker/gpu/spec_decode/__init__.py v1/worker/gpu/spec_decode/speculator.py v1/worker/gpu/spec_decode/dflash2; do
    MOUNTS+=(-v "$P/$DFLASH_DIR/vllm/$f:$VL/$f:ro")
  done
  MOUNTS+=(-v "$P/cstech-attn/attention.py:$VL/model_executor/layers/attention/attention.py:ro")
  MOUNTS+=(-v "$P/cstech-kv/kv_cache_utils.py:$VL/v1/core/kv_cache_utils.py:ro")
  ARGS+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$K,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"standard\",\"attention_backend\":\"FLASH_ATTN\",\"kv_cache_dtype\":\"auto\"}")
fi
# v2 steps 2-3: coarse KDA snapshot interval + prompt-end partial-tail resume + fine-grained (hash-unit)
# prefix hits. These five files move together: the scheduler stops a prefill split at the partial-tail
# boundary, the coordinator stops the inert group from disabling fine-grained hits, and the manager
# registers the Mamba partial tail early enough to survive the attention groups' eagle drop.
if [ "$MAMBA_BLOCK" != "0" ]; then
  MOUNTS+=(-v "$P/cstech-mamba-interval/interface.py:$VL/platforms/interface.py:ro")
  MOUNTS+=(-v "$P/cstech-mamba-interval/mamba_hybrid.py:$VL/v1/worker/gpu/model_states/mamba_hybrid.py:ro")
  MOUNTS+=(-v "$P/cstech-mamba-interval/sched_scheduler.partial-tail.py:$VL/v1/core/sched/scheduler.py:ro")
  MOUNTS+=(-v "$P/cstech-coord/kv_cache_coordinator.py:$VL/v1/core/kv_cache_coordinator.py:ro")
  MOUNTS+=(-v "$P/cstech-partial-tail/single_type_kv_cache_manager.py:$VL/v1/core/single_type_kv_cache_manager.py:ro")
  ARGS+=(--mamba-block-size "$MAMBA_BLOCK")
fi
[ -n "$EAGLE_DROP_UNITS" ] && ENVS+=(-e VLLM_EAGLE_DROP_UNITS="$EAGLE_DROP_UNITS")
# v2 step 4: bf16 KDA temporal state (--mamba-cache-dtype alone is a no-op; kda_state_dtype hard-codes fp32)
if [ -n "$KDA_STATE_DTYPE" ]; then
  MOUNTS+=(-v "$P/cstech-kda-state/mamba_utils.py:$VL/model_executor/layers/mamba/mamba_utils.py:ro")
  ENVS+=(-e VLLM_KDA_STATE_DTYPE="$KDA_STATE_DTYPE")
fi
if [ "$OFFLOAD_GIB" != "0" ]; then
  # step 7: host-RAM KV offload (needs stable CUDA registrations -> expandable_segments off).
  # v2: the drafter's sliding-window group also goes inert (it was 3.3 of every 7.4 GiB per 60k prompt),
  # and the scheduler learns the mamba re-round + partial-tail hand-off + the eagle cap on RAM hits.
  if [ "$MAMBA_BLOCK" != "0" ]; then
    OFF_CONFIG="config.sw-inert.py"; OFF_SCHED="scheduler.partial-tail.py"
  else
    OFF_CONFIG="config.py"; OFF_SCHED="scheduler.py"
  fi
  MOUNTS+=(-v "$P/cstech-offload/$OFF_CONFIG:$VL/distributed/kv_transfer/kv_connector/v1/offloading/config.py:ro")
  MOUNTS+=(-v "$P/cstech-offload/$OFF_SCHED:$VL/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:ro")
  ARGS+=(--kv-offloading-size "$OFFLOAD_GIB" --kv-offloading-backend native)
  ALLOC_CONF="expandable_segments:False"
fi
CMD=(docker run --rm --name glm53-flash --gpus all --network host --ipc host --init --shm-size "$SHM"
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 --ulimit stack=67108864
  "${MOUNTS[@]}"
  -e VLLM_DISABLED_KERNELS=FlashInferCutedslMxfp8LinearKernel,FlashInferCutlassMxfp8LinearKernel   # step 2(e): Marlin for MXFP8
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="$GPUS" -e NCCL_P2P_DISABLE=1 -e PYTHONHASHSEED=0
  -e INSTANTTENSOR_BACKEND=BUFFERED -e SAFETENSORS_FAST_GPU=1 -e PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF"
  -e VLLM_LOGGING_LEVEL=INFO
  "${ENVS[@]}"
  "$IMAGE" "$MODEL" "${ARGS[@]}")
if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
exec "${CMD[@]}"
