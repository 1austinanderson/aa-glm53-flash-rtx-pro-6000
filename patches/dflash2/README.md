DFlash 2 overlay for cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1 (2026-08-28)
= upstream vLLM PR #52816 "[Spec Decode] DFlash2: local convolution + candidate selector"
(merged 2026-08-21; the image's fork branched before it). Applied with `patch -F3`; two fuzz-3
hunks hand-corrected: registry.py line placed after DFlashDraftModel (patch dropped it inside the
K3DSparkModel tuple) and the DFlash2 dispatch moved under method == "dflash" (patch put it under
"dspark"). Follow-up upstream fixes NOT included (all still open 08-28): #53662 init, #53978
warmup, #52883 selector LM head, #54041 SWA KV groups, #53979 sm12x FA2 prefill.
Files are bind-mounted over the image by ~/glm53-flash-serve.sh when GLM_DFLASH2=1.
