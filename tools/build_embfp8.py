#!/usr/bin/env python3
"""Derive GLM-5.3-Flash-NVFP4-4p67-embfp8 from GLM-5.3-Flash-NVFP4-4p67.

PRODUCES a derived checkpoint directory (~3.4 GiB on disk, not another 175 GiB)
that loads 0.295 GiB/GPU lighter than the source:

  * ONE rewritten shard (model-hf-nonexpert-00001-of-00004.safetensors) in which
    exactly one tensor -- model.language_model.embed_tokens.weight -- goes from
    BF16 [154880, 4096] to float8_e4m3fn with per-row (per-vocab-entry) fp32
    scales stored alongside as ...embed_tokens.weight_scale [154880].  Every
    other tensor in that shard is copied byte-identically (same dtype, same
    values, same key order); lm_head stays BF16.
  * EDITED config.json / hf_quant_config.json / model.safetensors.index.json:
    the quant config gains a `group_fp8_embed` config group and a
    `quantized_layers[...embed_tokens] = FP8_EMBED_ROW` entry, which is what
    the patches/fp8-embed/modelopt.py overlay keys off at load time.
  * RELATIVE SYMLINKS for everything else (the three expert shards, tokenizer,
    ...) back into the source checkpoint -- relative on purpose, so they resolve
    the same on the host and through the container's models bind mount.

Run inside the vLLM image:

  docker run --rm --entrypoint python3 \
    -v <src>:/src:ro -v <dst>:/dst -v <this dir>:/work:ro \
    glm53-cstech-it:20260828 /work/build_embfp8.py
"""

import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/src"
DST = "/dst"
# Directory name of SRC *on the host*, not inside the container: the symlinks
# below are relative and must resolve through the models-dir ->
# /root/models bind mount, so they cannot use the container's /src mount name.
SRC_HOST_NAME = os.environ.get("FP8EMB_SRC_NAME", "GLM-5.3-Flash-NVFP4-4p67")
SHARD = "model-hf-nonexpert-00001-of-00004.safetensors"
EMB = "model.language_model.embed_tokens.weight"
EMB_SCALE = "model.language_model.embed_tokens.weight_scale"
FP8_MAX = 448.0
# Name used in quantized_layers; ModelOptMixedPrecisionConfig.get_quant_method
# in the fp8-embed overlay keys off this string.
ALGO = "FP8_EMBED_ROW"
GROUP = "group_fp8_embed"
EMB_MODULE = "model.language_model.embed_tokens"

# Files we write ourselves; everything else is symlinked.
REWRITTEN = {SHARD, "model.safetensors.index.json", "hf_quant_config.json", "config.json"}


def log(*a):
    print(*a, flush=True)


def quantize_rows(w: torch.Tensor):
    """Per-row symmetric fp8-e4m3 quantization.  Returns (q, scale[N] fp32)."""
    n_rows = w.shape[0]
    q = torch.empty(w.shape, dtype=torch.float8_e4m3fn)
    scale = torch.empty(n_rows, dtype=torch.float32)
    chunk = 8192
    n_zero_rows = 0
    for s in range(0, n_rows, chunk):
        e = min(s + chunk, n_rows)
        blk = w[s:e].to(torch.float32)
        amax = blk.abs().amax(dim=1)
        n_zero_rows += int((amax == 0).sum())
        # A row that is exactly zero has amax 0; scale 1.0 keeps it exactly zero
        # and avoids a divide-by-zero.
        sc = torch.where(amax == 0, torch.ones_like(amax), amax / FP8_MAX)
        q[s:e] = torch.clamp(blk / sc.unsqueeze(1), -FP8_MAX, FP8_MAX).to(
            torch.float8_e4m3fn
        )
        scale[s:e] = sc
        del blk
    return q, scale, n_zero_rows


def main():
    src_shard = os.path.join(SRC, SHARD)
    dst_shard = os.path.join(DST, SHARD)

    log(f"reading {src_shard}")
    with safe_open(src_shard, framework="pt") as f:
        keys = list(f.keys())
        metadata = f.metadata()
        tensors = {k: f.get_tensor(k) for k in keys}
    log(f"  {len(keys)} tensors, metadata={metadata}")

    w = tensors[EMB]
    assert w.dtype == torch.bfloat16, w.dtype
    assert w.dim() == 2, w.shape
    n_rows, n_cols = w.shape
    log(f"  {EMB}: {tuple(w.shape)} {w.dtype}")

    q, scale, n_zero_rows = quantize_rows(w)
    log(f"  quantized; {n_zero_rows} all-zero rows (scale forced to 1.0)")

    # Round-trip error on a sample, just as a build-time sanity check; the real
    # numbers come from verify_fp8_embed.py.
    ref = w[:4096].to(torch.float32)
    deq = q[:4096].to(torch.float32) * scale[:4096].unsqueeze(1)
    log(f"  sample max abs err {float((deq - ref).abs().max()):.6g}")

    tensors[EMB] = q
    # Insert weight_scale directly after weight so the header key order stays
    # readable; safetensors sorts by name internally anyway.
    tensors[EMB_SCALE] = scale

    os.makedirs(DST, exist_ok=True)
    log(f"writing {dst_shard}")
    save_file(tensors, dst_shard, metadata=metadata)
    del tensors, q, w

    # --- round-trip check on the file we just wrote -----------------------
    with safe_open(dst_shard, framework="pt") as f:
        new_keys = list(f.keys())
        assert f.metadata() == metadata, (f.metadata(), metadata)
        qs = f.get_slice(EMB)
        ss = f.get_slice(EMB_SCALE)
        log(f"  reread: {len(new_keys)} tensors, metadata={f.metadata()}")
        log(f"  {EMB}: {qs.get_dtype()} {qs.get_shape()}")
        log(f"  {EMB_SCALE}: {ss.get_dtype()} {ss.get_shape()}")
    assert set(new_keys) == set(keys) | {EMB_SCALE}
    assert qs.get_dtype() == "F8_E4M3" and qs.get_shape() == [n_rows, n_cols]
    assert ss.get_dtype() == "F32" and ss.get_shape() == [n_rows]

    # --- index.json --------------------------------------------------------
    with open(os.path.join(SRC, "model.safetensors.index.json")) as fh:
        index = json.load(fh)
    wm = index["weight_map"]
    assert wm[EMB] == SHARD
    assert EMB_SCALE not in wm
    wm[EMB_SCALE] = SHARD
    old_total = index["metadata"]["total_size"]
    delta = (
        -n_rows * n_cols * 2  # bf16 weight removed
        + n_rows * n_cols * 1  # fp8 weight added
        + n_rows * 4  # fp32 scale added
    )
    index["metadata"]["total_size"] = old_total + delta
    with open(os.path.join(DST, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh)
    log(
        f"index: {len(wm)} entries, total_size {old_total} -> "
        f"{index['metadata']['total_size']} ({delta:+d})"
    )

    # --- quant configs -----------------------------------------------------
    # vLLM reads config.json's quantization_config first (transformers_utils/
    # config.py:760) and only falls back to hf_quant_config.json, so both files
    # get the new group.  They are identical in the source checkpoint.
    group_cfg = {
        "weights": {
            "dynamic": False,
            "num_bits": 8,
            "type": "float",
            "strategy": "channel",
            "symmetric": True,
        },
        "input_activations": None,
        "targets": [EMB_MODULE],
    }

    def extend(qcfg):
        assert GROUP not in qcfg["config_groups"]
        qcfg["config_groups"][GROUP] = group_cfg
        assert EMB_MODULE not in qcfg["quantized_layers"]
        qcfg["quantized_layers"][EMB_MODULE] = {"quant_algo": ALGO}
        return qcfg

    with open(os.path.join(SRC, "hf_quant_config.json")) as fh:
        hq = json.load(fh)
    with open(os.path.join(DST, "hf_quant_config.json"), "w") as fh:
        json.dump(extend(hq), fh, indent=2)

    with open(os.path.join(SRC, "config.json")) as fh:
        cfg = json.load(fh)
    extend(cfg["quantization_config"])
    with open(os.path.join(DST, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    log(
        "quant configs extended: config_groups += %s, quantized_layers += "
        "%s -> %s (%d entries)"
        % (GROUP, EMB_MODULE, ALGO, len(cfg["quantization_config"]["quantized_layers"]))
    )

    # --- relative symlinks for everything else -----------------------------
    # Relative (not absolute) on purpose: the engine sees this tree through the
    # bind mount <models dir> -> /root/models, so an absolute host
    # symlink would dangle inside the container.  A relative link resolves the
    # same on the host and in the container.
    n_links = 0
    for name in sorted(os.listdir(SRC)):
        if name in REWRITTEN:
            continue
        dst = os.path.join(DST, name)
        if os.path.lexists(dst):
            os.unlink(dst)
        os.symlink(os.path.join("..", SRC_HOST_NAME, name), dst)
        n_links += 1
    log(f"symlinked {n_links} entries")

    size = os.path.getsize(dst_shard)
    log(f"new shard size {size} bytes ({size / 2**30:.3f} GiB)")
    log("OK")


if __name__ == "__main__":
    sys.exit(main())
