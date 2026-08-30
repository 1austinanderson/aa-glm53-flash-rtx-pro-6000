DFlash 2 overlay + quantized-attention fix (2026-08-30)
=======================================================
Copy of `patches/dflash2/` (upstream vLLM PR #52816, see that dir's history) with ONE
change, in `vllm/model_executor/models/qwen3_dflash.py`:

  `DFlashQwen3Model._build_context_kv_buffers` / `_project_context_kv` used to build the
  fused context K/V projection by row-slicing the RAW `qkv_proj.weight`
  (`a.qkv_proj.weight[a.q_size:]`) and running `F.linear` on it.  That only works while
  attention is unquantized: with a quantized attention checkpoint the parameter is fp8 or
  packed uint8 (and after `process_weights_after_loading` may be transposed/repacked), so
  `F.linear` raises `expected mat1 and mat2 to have the same dtype`.  This is why
  `GLM-5.3-Flash-DFlash2-FP8` had to carry `re:.*self_attn.*` in its ignore list.

  The fix adds `_qkv_weight_is_raw()`, evaluated once at load time.  Unquantized ->
  behaviour is byte-identical to `patches/dflash2` (same fused single GEMM).  Quantized ->
  each layer's own `qkv_proj` forward runs (so its quant method / kernel applies) and the
  K/V *columns* of the output are kept; `F.linear(x, W[q:])` == `F.linear(x, W)[..., q:]`,
  and the per-layer slices concatenate into the identical layer-major
  `[num_ctx, L*2*kv_size]` layout.  Cost of the quantized path: L=5 small GEMMs instead of
  one, and Q is computed and discarded (3x the FLOPs of this projection: 6144 vs 2048
  output columns per layer, on num_ctx rows only).  The branch is a Python bool fixed at
  load time - no host sync, CUDA-graph / torch.compile safe.

Baseline for the diff: `vllm/model_executor/models/qwen3_dflash.py.pristine.py`
(= the live `patches/dflash2` file).  Diff: `../dflash2-quantattn.patch`.
`qwen3_dflash.py.orig` is still the image's pre-PR-52816 file, as in `patches/dflash2`.

Boot with this overlay instead of `patches/dflash2`:
  GLM_DFLASH2_DIR=$REPO/patches/dflash2-quantattn \
  GLM_DFLASH2_MODEL=/root/models/incoai/GLM-5.3-Flash-DFlash2-NVFP4-attnFP8
(`serve.sh` reads `GLM_DFLASH2_DIR`, default `patches/dflash2`.)
