#!/usr/bin/env python3
# PATCH (local, 2026-08-30): CPU verification for the FP8 input-embedding work.
#
# Checks three things, all on CPU, against the real vLLM code in the image:
#   (a) numerics -- dequantize the derived shard's embedding and compare it to
#       the original BF16 table.
#   (b) plumbing -- build the real VocabParallelEmbedding with the real
#       ModelOptFp8RowEmbeddingMethod for TP world size 2, rank 0 and rank 1;
#       load the derived shard's tensors through the real weight_loader; check
#       that the reassembled TP2 lookup matches the dequantized reference.
#   (c) drafter -- show the DFlash-2 draft model still gets a working embedding.
#
# Run:
#   docker run --rm --entrypoint python3 \
#     -v $MODELS_DIR:/root/models:ro \
#     -v $REPO/patches/fp8-embed/modelopt.py:\
# /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py:ro \
#     -v $REPO/patches/fp8-embed:/work:ro \
#     -e CUDA_VISIBLE_DEVICES= \
#     glm53-cstech-it:20260828 /work/verify_fp8_embed.py
#
# Exits non-zero if any check fails.

import json
import os
import sys

import torch
from safetensors import safe_open

ORIG = os.environ.get(
    "FP8EMB_ORIG", "/root/models/local-inference-lab/GLM-5.3-Flash-NVFP4-4p67"
)
NEW = os.environ.get(
    "FP8EMB_NEW", "/root/models/local-inference-lab/GLM-5.3-Flash-NVFP4-4p67-embfp8"
)
DRAFT = os.environ.get("FP8EMB_DRAFT", "/root/models/incoai/GLM-5.3-Flash-DFlash2-FP8")
SHARD = "model-hf-nonexpert-00001-of-00004.safetensors"
EMB = "model.language_model.embed_tokens.weight"
EMB_SCALE = "model.language_model.embed_tokens.weight_scale"
EMB_PREFIX = "language_model.model.embed_tokens"
FP8_MAX = 448.0
PARAMS_DTYPE = torch.bfloat16

_failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}",
          flush=True)
    if not ok:
        _failures.append(name)
    return ok


def section(title):
    print(f"\n=== {title} ===", flush=True)


# ---------------------------------------------------------------------------
# (a) numerics
# ---------------------------------------------------------------------------
def part_a():
    section("(a) dequantized embedding vs original BF16")

    with safe_open(os.path.join(ORIG, SHARD), framework="pt") as f:
        orig_keys = set(f.keys())
        ref_bf16 = f.get_tensor(EMB)
    with safe_open(os.path.join(NEW, SHARD), framework="pt") as f:
        new_keys = set(f.keys())
        new_meta = f.metadata()
        q = f.get_tensor(EMB)
        scale = f.get_tensor(EMB_SCALE)

    check(
        "shard tensor set == original + weight_scale",
        new_keys == orig_keys | {EMB_SCALE},
        f"{len(new_keys)} tensors",
    )
    check("shard metadata preserved", new_meta == {"format": "pt"}, str(new_meta))
    check(
        "embed weight dtype/shape",
        q.dtype == torch.float8_e4m3fn and tuple(q.shape) == tuple(ref_bf16.shape),
        f"{q.dtype} {tuple(q.shape)}",
    )
    check(
        "weight_scale dtype/shape",
        scale.dtype == torch.float32 and tuple(scale.shape) == (q.shape[0],),
        f"{scale.dtype} {tuple(scale.shape)}",
    )

    # Every other tensor must be byte-identical to the original.
    n_checked, n_same = 0, 0
    with safe_open(os.path.join(ORIG, SHARD), framework="pt") as fo, safe_open(
        os.path.join(NEW, SHARD), framework="pt"
    ) as fn:
        for k in sorted(orig_keys):
            if k == EMB:
                continue
            n_checked += 1
            a, b = fo.get_tensor(k), fn.get_tensor(k)
            if a.dtype == b.dtype and a.shape == b.shape and torch.equal(
                a.view(torch.uint8) if a.dtype.itemsize == 1 else a, b.view(torch.uint8)
                if b.dtype.itemsize == 1
                else b
            ):
                n_same += 1
    check(
        "all other tensors byte-identical",
        n_same == n_checked,
        f"{n_same}/{n_checked}",
    )

    n_rows, n_cols = q.shape
    max_abs = 0.0
    sum_abs_err = 0.0
    sum_abs_ref = 0.0
    min_cos = 2.0
    rows_over_5pct = 0
    max_err_over_amax = 0.0
    chunk = 8192
    for s in range(0, n_rows, chunk):
        e = min(s + chunk, n_rows)
        ref = ref_bf16[s:e].to(torch.float32)
        deq = q[s:e].to(torch.float32) * scale[s:e].unsqueeze(1)
        err = (deq - ref).abs()
        max_abs = max(max_abs, float(err.max()))
        sum_abs_err += float(err.sum())
        sum_abs_ref += float(ref.abs().sum())
        amax = ref.abs().amax(dim=1).clamp_min(1e-30)
        rel_to_amax = err.amax(dim=1) / amax
        max_err_over_amax = max(max_err_over_amax, float(rel_to_amax.max()))
        rows_over_5pct += int((rel_to_amax > 0.05).sum())
        cos = torch.nn.functional.cosine_similarity(deq, ref, dim=1)
        min_cos = min(min_cos, float(cos.min()))
        del ref, deq, err

    mean_rel = sum_abs_err / sum_abs_ref
    print(f"  max abs error                       {max_abs:.6g}")
    print(f"  mean relative error (L1/L1)         {mean_rel:.6g}")
    print(f"  min per-row cosine similarity       {min_cos:.9f}")
    print(f"  worst per-row  max|err| / row amax  {max_err_over_amax:.6g}")
    print(f"  rows with any element err > 5% of row amax   {rows_over_5pct}")

    # Bound: with a per-row scale s = amax/448 the largest e4m3 bin is
    # [256, 448] with ulp 32, so |q*s - x| <= 16 s = amax/28 = 3.5714% of the
    # row amax.  This part dequantizes in fp32, so that is the whole budget.
    check("no row exceeds 5% of its amax", rows_over_5pct == 0)
    check("worst row error within the e4m3 half-ulp bound (3.5715% of amax)",
          max_err_over_amax <= 0.035715, f"{max_err_over_amax:.6g}")
    check("min per-row cosine similarity > 0.999", min_cos > 0.999, f"{min_cos:.9f}")

    return q, scale, ref_bf16


# ---------------------------------------------------------------------------
# (b) real VocabParallelEmbedding under TP2
# ---------------------------------------------------------------------------
def part_b(q, scale, ref_bf16):
    section("(b) real VocabParallelEmbedding + weight_loader, TP world size 2")

    import vllm.model_executor.layers.vocab_parallel_embedding as vpe
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptFp8RowEmbeddingMethod,
        ModelOptMixedPrecisionConfig,
    )

    # Build the quant config straight from the derived checkpoint's config.json
    # so the wiring under test is the real one (quantized_layers lookup), not a
    # hand-made stand-in.
    with open(os.path.join(NEW, "config.json")) as fh:
        qcfg_dict = json.load(fh)["quantization_config"]
    quant_config = ModelOptMixedPrecisionConfig.from_config(qcfg_dict)
    check(
        "checkpoint declares MIXED_PRECISION with an FP8_EMBED_ROW embedding",
        quant_config.quantized_layers.get(
            "model.language_model.embed_tokens", {}
        ).get("quant_algo")
        == "FP8_EMBED_ROW",
    )

    n_rows, n_cols = q.shape
    vocab_size = n_rows

    # Mock the two distributed lookups VocabParallelEmbedding.__init__ makes and
    # the all-reduce in forward(); everything else is the real code path.
    orig_rank = vpe.get_tensor_model_parallel_rank
    orig_ws = vpe.get_tensor_model_parallel_world_size
    orig_ar = vpe.tensor_model_parallel_all_reduce
    vpe.tensor_model_parallel_all_reduce = lambda x: x  # sum the ranks by hand

    layers = {}
    try:
        for rank in (0, 1):
            vpe.get_tensor_model_parallel_rank = lambda r=rank: r
            vpe.get_tensor_model_parallel_world_size = lambda: 2
            layer = vpe.VocabParallelEmbedding(
                vocab_size,
                n_cols,
                params_dtype=PARAMS_DTYPE,
                quant_config=quant_config,
                prefix=EMB_PREFIX,
            )
            layers[rank] = layer
    finally:
        vpe.get_tensor_model_parallel_rank = orig_rank
        vpe.get_tensor_model_parallel_world_size = orig_ws

    check(
        "quant method resolved to ModelOptFp8RowEmbeddingMethod on both ranks",
        all(
            isinstance(layers[r].quant_method, ModelOptFp8RowEmbeddingMethod)
            for r in (0, 1)
        ),
        type(layers[0].quant_method).__name__,
    )
    check(
        "no vocab padding (154880 is divisible by 64 and by TP2)",
        layers[0].num_embeddings_padded == vocab_size
        and layers[0].num_embeddings_per_partition == vocab_size // 2,
        f"padded={layers[0].num_embeddings_padded} "
        f"per_partition={layers[0].num_embeddings_per_partition}",
    )
    check(
        "weight is fp8 and weight_scale is fp32 on the shard's vocab range",
        all(
            layers[r].weight.dtype == torch.float8_e4m3fn
            and tuple(layers[r].weight.shape) == (vocab_size // 2, n_cols)
            and layers[r].weight_scale.dtype == torch.float32
            and tuple(layers[r].weight_scale.shape) == (vocab_size // 2,)
            for r in (0, 1)
        ),
        f"{tuple(layers[0].weight.shape)} / {tuple(layers[0].weight_scale.shape)}",
    )

    # Load through the real weight_loader (the same one the checkpoint loader
    # calls via param.weight_loader).
    for rank in (0, 1):
        layer = layers[rank]
        layer.weight_loader(layer.weight, q)
        layer.weight_loader(layer.weight_scale, scale)
    lo, hi = (
        layers[1].shard_indices.org_vocab_start_index,
        layers[1].shard_indices.org_vocab_end_index,
    )
    check(
        "rank 1 holds vocab rows [77440, 154880)", (lo, hi) == (77440, 154880),
        f"[{lo}, {hi})",
    )
    check(
        "rank 1 fp8 rows match the checkpoint slice bit-for-bit",
        torch.equal(
            layers[1].weight.view(torch.uint8), q[lo:hi].view(torch.uint8)
        ),
    )
    check(
        "rank 1 scales match the checkpoint slice",
        torch.equal(layers[1].weight_scale, scale[lo:hi]),
    )

    # Reference: the dequantized table, in the model dtype, computed exactly the
    # way the quant method computes it.
    def deq_ref(ids):
        rows = q.index_select(0, ids).to(torch.float32)
        sc = scale.index_select(0, ids).to(torch.float32).unsqueeze(-1)
        return (rows * sc).to(PARAMS_DTYPE)

    torch.manual_seed(20260830)
    special = [0, 1, vocab_size - 1, vocab_size - 2, 77439, 77440]
    # From the checkpoint's own tokenizer/generation config: pad, eos set, and
    # the DFlash mask token.
    for path, keys in (
        (os.path.join(ORIG, "config.json"), None),
        (os.path.join(ORIG, "generation_config.json"), None),
        (os.path.join(DRAFT, "config.json"), None),
    ):
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            cfg = json.load(fh)

        def harvest(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    harvest(v)
                elif "token_id" in k or k in ("eos_token_id", "pad_token_id"):
                    for t in v if isinstance(v, list) else [v]:
                        if isinstance(t, int) and 0 <= t < vocab_size:
                            special.append(t)

        harvest(cfg)
    special = sorted(set(special))
    print(f"  special ids under test: {special}")

    rand_ids = torch.randint(0, vocab_size, (2000,), dtype=torch.long)
    ids = torch.cat([rand_ids, torch.tensor(special, dtype=torch.long)])

    out = None
    for rank in (0, 1):
        y = layers[rank](ids)
        out = y if out is None else out + y  # stands in for the all-reduce
    ref_deq = deq_ref(ids)

    check(
        f"TP2 lookup of {len(ids)} ids ({len(rand_ids)} random + "
        f"{len(special)} special) == dequantized reference, exactly",
        torch.equal(out, ref_deq),
        f"max diff {float((out.float() - ref_deq.float()).abs().max()):.6g}",
    )

    # And against the ORIGINAL bf16 table.  The layer returns the model dtype,
    # so the budget is e4m3's half-ulp plus bf16's own rounding of the product:
    #   16 s  +  2^-8 * 448 s  =  17.75 s  =  17.75/448 = 3.9621% of row amax.
    ref_orig = ref_bf16.index_select(0, ids).to(torch.float32)
    amax = ref_orig.abs().amax(dim=1).clamp_min(1e-30)
    err = (out.to(torch.float32) - ref_orig).abs()
    rel_to_amax = err.amax(dim=1) / amax
    worst = float(rel_to_amax.max())
    n_over_5pct = int((rel_to_amax > 0.05).sum())
    check(
        "TP2 lookup within the e4m3 + bf16 rounding bound (3.9621% of row amax)",
        worst <= 0.039621,
        f"worst max|err|/row amax = {worst:.6g}",
    )
    check(
        "no looked-up row exceeds 5% of its amax", n_over_5pct == 0,
        f"{n_over_5pct}/{len(ids)} rows",
    )

    vpe.tensor_model_parallel_all_reduce = orig_ar

    # A BF16 checkpoint (no FP8_EMBED_ROW entry) must keep the stock method.
    bf16_cfg_dict = json.loads(json.dumps(qcfg_dict))
    del bf16_cfg_dict["quantized_layers"]["model.language_model.embed_tokens"]
    bf16_cfg = ModelOptMixedPrecisionConfig.from_config(bf16_cfg_dict)
    try:
        vpe.get_tensor_model_parallel_rank = lambda: 0
        vpe.get_tensor_model_parallel_world_size = lambda: 2
        plain = vpe.VocabParallelEmbedding(
            vocab_size, 8, params_dtype=PARAMS_DTYPE,
            quant_config=bf16_cfg, prefix=EMB_PREFIX,
        )
    finally:
        vpe.get_tensor_model_parallel_rank = orig_rank
        vpe.get_tensor_model_parallel_world_size = orig_ws
    check(
        "checkpoint without the FP8_EMBED_ROW entry keeps "
        "UnquantizedEmbeddingMethod / BF16",
        isinstance(plain.quant_method, vpe.UnquantizedEmbeddingMethod)
        and plain.weight.dtype == PARAMS_DTYPE
        and not hasattr(plain, "weight_scale"),
        type(plain.quant_method).__name__,
    )

    return layers


# ---------------------------------------------------------------------------
# (c) DFlash-2 drafter
# ---------------------------------------------------------------------------
def part_c(layers):
    section("(c) DFlash-2 drafter embedding path")

    import torch.nn as nn
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share

    draft_ckpt = os.path.join(DRAFT, "model.safetensors")
    if os.path.exists(draft_ckpt):
        with safe_open(draft_ckpt, framework="pt") as f:
            draft_keys = list(f.keys())
        has_embed = any("embed_tokens" in k for k in draft_keys)
        check(
            "draft checkpoint ships no embed_tokens "
            "(so has_own_embed_tokens stays False)",
            not has_embed,
            f"{len(draft_keys)} tensors",
        )
    else:
        check("draft checkpoint present", False, draft_ckpt)
        return

    # load_dflash_model does:  if _should_share(...): draft_inner.embed_tokens =
    # target_embed  -- a MODULE assignment, not a weight copy, so the drafter
    # inherits the fp8 weight *and* its scales *and* the quant method, at zero
    # extra VRAM.  _should_share short-circuits on has_own_embed_tokens=False
    # before it would ever touch .weight, so there is no fp8-vs-bf16
    # torch.equal() to trip over.
    target_embed = layers[0]
    dflash_model = nn.Module()
    dflash_model.model = nn.Module()
    draft_inner = dflash_model.model
    draft_inner.embed_tokens = None  # drafter's own table, unloaded

    share = _should_share(
        dflash_model, "has_own_embed_tokens", draft_inner.embed_tokens, target_embed
    )
    check("_should_share() -> True without reading .weight", share is True)

    draft_inner.embed_tokens = target_embed
    check(
        "drafter's embed_tokens IS the target module (shared, not copied)",
        draft_inner.embed_tokens is target_embed,
    )
    check(
        "shared module keeps the fp8 weight + fp32 scales",
        draft_inner.embed_tokens.weight.dtype == torch.float8_e4m3fn
        and draft_inner.embed_tokens.weight_scale.dtype == torch.float32,
    )

    ids = torch.tensor([0, 1, 154820, 154856, 154879], dtype=torch.long)
    import vllm.model_executor.layers.vocab_parallel_embedding as vpe

    orig_ar = vpe.tensor_model_parallel_all_reduce
    vpe.tensor_model_parallel_all_reduce = lambda x: x
    try:
        via_draft = draft_inner.embed_tokens(ids)
        via_target = target_embed(ids)
    finally:
        vpe.tensor_model_parallel_all_reduce = orig_ar
    check(
        "drafter lookup == target lookup (incl. DFlash mask_token_id 154856)",
        torch.equal(via_draft, via_target),
    )
    print(
        "  note: the drafter applies its separate mask_embedding (when a "
        "mask_embedding.pt is shipped -- this one is not) to the *output* of "
        "embed_tokens, never to .weight, so it is unaffected."
    )


def main():
    q, scale, ref_bf16 = part_a()
    layers = part_b(q, scale, ref_bf16)
    part_c(layers)

    section("summary")
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {', '.join(_failures)}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
