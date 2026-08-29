#!/usr/bin/env bash
# GLM-5.3-Flash (NVFP4/MXFP8) on 2x RTX PRO 6000 Blackwell — vLLM TP2 with DFlash-2 speculative decoding.
# Standalone launcher: every path/knob is an env var. Run from anywhere.
#   MODELS_DIR   host dir containing local-inference-lab/GLM-5.3-Flash-NVFP4-4p67 and incoai/GLM-5.3-Flash-DFlash2-FP8
#   REPO_DIR     this repo (patches/ is bind-mounted over the image's vLLM)
#   CACHE_DIR    host dir for vLLM/FlashInfer compile caches and /tmp (persist across boots)
#   IMAGE        glm53-cstech-it:20260828 (build: docker build -f Dockerfile.instanttensor -t glm53-cstech-it:20260828 .)
#                or cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1 with LOAD_FORMAT=auto
# Optional: PORT, GPUS, CTX, SEQS, MNBT, UTIL, KV_MEM_BYTES, K (0 = no speculative decoding), OFFLOAD_GIB (0 = off), SHM, DRY_RUN=1
set -euo pipefail
MODELS_DIR="${MODELS_DIR:-/mnt/raid/models}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/cache}"
IMAGE="${IMAGE:-glm53-cstech-it:20260828}"
LOAD_FORMAT="${LOAD_FORMAT:-instanttensor}"
PORT="${PORT:-8000}"; GPUS="${GPUS:-0,1}"; CTX="${CTX:-393216}"; SEQS="${SEQS:-8}"; MNBT="${MNBT:-1024}"; UTIL="${UTIL:-0.988}"
KV_MEM_BYTES="${KV_MEM_BYTES:-4080218931}"   # 3.8 GiB explicit claim; 0 = let vLLM profile from UTIL
K="${K:-3}"; DRAFTER="${DRAFTER:-/root/models/incoai/GLM-5.3-Flash-DFlash2-FP8}"
OFFLOAD_GIB="${OFFLOAD_GIB:-32}"; SHM="${SHM:-48gb}"
MODEL="${MODEL:-/root/models/local-inference-lab/GLM-5.3-Flash-NVFP4-4p67}"
VL=/usr/local/lib/python3.12/dist-packages/vllm
P="$REPO_DIR/patches"
mkdir -p "$CACHE_DIR"/{vllm,flashinfer,tmp}

MOUNTS=(
  -v "$MODELS_DIR:/root/models:ro"
  -v "$CACHE_DIR/vllm:/root/.cache/vllm" -v "$CACHE_DIR/flashinfer:/root/.cache/flashinfer" -v "$CACHE_DIR/tmp:/tmp"
  # step 2: MXFP8 mixed-precision attention (+ SupportsEagle3 for DFlash)
  -v "$P/glm5next:$VL/models/glm5next:ro"
  # step 6: sparse-indexer K-gather workspace sized by seqs x ctx
  -v "$P/cstech-indexer/indexer.py:$VL/v1/attention/backends/mla/indexer.py:ro"
)
ARGS=(
  --served-model-name GLM-5.3-Flash --host 0.0.0.0 --port "$PORT" --tensor-parallel-size 2
  --max-num-batched-tokens "$MNBT" --max-num-seqs "$SEQS" --max-model-len "$CTX" --gpu-memory-utilization "$UTIL"
  --kv-cache-dtype fp8 --load-format "$LOAD_FORMAT" --enable-prefix-caching --no-enable-flashinfer-autotune
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45 --disable-custom-all-reduce
)
[ "$KV_MEM_BYTES" != "0" ] && ARGS+=(--kv-cache-memory "$KV_MEM_BYTES")
ALLOC_CONF="expandable_segments:True"
if [ "$K" != "0" ]; then
  # step 4: DFlash-2 (upstream PR #52816) + the two KV-layout fixes it needs on this image
  for f in model_executor/models/qwen3_dflash2.py model_executor/models/qwen3_dflash.py model_executor/models/registry.py \
           model_executor/layers/logits_processor.py config/vllm.py v1/worker/gpu/sample/gumbel.py \
           v1/worker/gpu/spec_decode/__init__.py v1/worker/gpu/spec_decode/speculator.py v1/worker/gpu/spec_decode/dflash2; do
    MOUNTS+=(-v "$P/dflash2/vllm/$f:$VL/$f:ro")
  done
  MOUNTS+=(-v "$P/cstech-attn/attention.py:$VL/model_executor/layers/attention/attention.py:ro")
  MOUNTS+=(-v "$P/cstech-kv/kv_cache_utils.py:$VL/v1/core/kv_cache_utils.py:ro")
  ARGS+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$K,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"standard\",\"attention_backend\":\"FLASH_ATTN\",\"kv_cache_dtype\":\"auto\"}")
fi
if [ "$OFFLOAD_GIB" != "0" ]; then
  # step 7: host-RAM KV offload (needs stable CUDA registrations -> expandable_segments off)
  MOUNTS+=(-v "$P/cstech-offload/config.py:$VL/distributed/kv_transfer/kv_connector/v1/offloading/config.py:ro")
  MOUNTS+=(-v "$P/cstech-offload/scheduler.py:$VL/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:ro")
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
  "$IMAGE" "$MODEL" "${ARGS[@]}")
if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
exec "${CMD[@]}"
