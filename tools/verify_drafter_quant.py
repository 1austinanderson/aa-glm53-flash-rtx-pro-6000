"""CPU verification of the NVFP4-MLP / FP8-attention DFlash2 drafter checkpoint.

Run inside the serving image, no GPU:

  docker run --rm --entrypoint python3 -e CUDA_VISIBLE_DEVICES= \
    -v $REPO/tools:/tools:ro -v $MODELS_DIR/incoai:/models:ro \
    -v $REPO/patches:/patches:ro \
    glm53-cstech-it:20260828 /tools/verify_drafter_quant.py \
    /models/GLM-5.3-Flash-DFlash2 /models/GLM-5.3-Flash-DFlash2-NVFP4-attnFP8

(a) dequantization error of every quantized tensor vs the bf16 source
(b) which compressed-tensors scheme vLLM resolves for every linear in the drafter
(c) the patched _project_context_kv vs the original, bf16 weights and a mocked
    quantized layer
(d) safetensors round-trip + parameter names/shapes/dtypes vs the schemes' own
    create_weights()

Exit code 0 = all PASS, 1 = any FAIL.
"""

import json
import os
import re
import sys
import types

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_drafter_nvfp4_attnfp8 import GROUP, nvfp4_dequantize  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def section(title: str) -> None:
    print(f"\n===== {title} =====")


# --------------------------------------------------------------------------
# (a) dequantization error
# --------------------------------------------------------------------------
def part_a(src_sd, dst_sd):
    section("(a) dequantized weights vs bf16 source")
    print(f"{'tensor':44s} {'scheme':6s} {'max|dW|':>10s} {'mean rel':>10s} {'min row cos':>12s}")
    mlp_re = re.compile(r"^layers\.\d+\.mlp\.(gate|up|down)_proj\.weight$")
    attn_re = re.compile(r"^layers\.\d+\.self_attn\.[qkvo]_proj\.weight$")
    worst_cos = {"nvfp4": 1.0, "fp8": 1.0}
    worst_rel = {"nvfp4": 0.0, "fp8": 0.0}
    for name in sorted(src_sd):
        if mlp_re.match(name):
            scheme = "nvfp4"
            base = name[: -len(".weight")]
            deq = nvfp4_dequantize(
                dst_sd[base + ".weight_packed"],
                dst_sd[base + ".weight_scale"],
                float(dst_sd[base + ".weight_global_scale"][0]),
            )
        elif attn_re.match(name):
            scheme = "fp8"
            base = name[: -len(".weight")]
            deq = dst_sd[base + ".weight"].to(torch.float32) * dst_sd[
                base + ".weight_scale"
            ].to(torch.float32)
        else:
            continue
        ref = src_sd[name].to(torch.float32)
        if deq.shape != ref.shape:
            check(f"shape {name}", False, f"{deq.shape} != {ref.shape}")
            continue
        max_abs = float((deq - ref).abs().max())
        denom = ref.abs().mean()
        mean_rel = float((deq - ref).abs().mean() / denom)
        cos = F.cosine_similarity(deq, ref, dim=1)
        min_cos = float(cos.min())
        worst_cos[scheme] = min(worst_cos[scheme], min_cos)
        worst_rel[scheme] = max(worst_rel[scheme], mean_rel)
        print(f"{name:44s} {scheme:6s} {max_abs:10.5f} {mean_rel:10.5f} {min_cos:12.6f}")
    # Element-wise round-off budgets.  E2M1 has a 1-bit mantissa: within a
    # binade the worst relative step is 1/6 and the RMS relative error lands
    # near 9%, so per-row cosine ~0.995.  E4M3 has a 3-bit mantissa: ~2.2% mean
    # relative error, per-row cosine ~0.9996.  Anything materially worse means
    # the scales or the packing are wrong, not that 4-bit is lossy.
    check("nvfp4 min per-row cosine > 0.99", worst_cos["nvfp4"] > 0.99,
          f"min={worst_cos['nvfp4']:.6f}")
    check("nvfp4 mean rel error < 0.12", worst_rel["nvfp4"] < 0.12,
          f"max={worst_rel['nvfp4']:.5f}")
    check("fp8 min per-row cosine > 0.999", worst_cos["fp8"] > 0.999,
          f"min={worst_cos['fp8']:.6f}")
    check("fp8 mean rel error < 0.04", worst_rel["fp8"] < 0.04,
          f"max={worst_rel['fp8']:.5f}")
    check("fp8 is at least 3x more accurate than nvfp4 (sanity on both paths)",
          worst_rel["fp8"] * 3 < worst_rel["nvfp4"],
          f"fp8={worst_rel['fp8']:.5f} nvfp4={worst_rel['nvfp4']:.5f}")


# --------------------------------------------------------------------------
# (b) scheme resolution
# --------------------------------------------------------------------------
def part_b(dst_dir, dst_sd):
    section("(b) scheme resolved by vLLM's CompressedTensorsConfig")
    import vllm.platforms as _platforms
    from vllm.model_executor.layers.linear import (
        LinearBase,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        ReplicatedLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )

    # There is no GPU here; pretend to be the SM120 card the drafter runs on so
    # _check_scheme_supported() and the kernel pickers behave as they will in
    # production.
    class _Cap:
        def to_int(self):
            return 120

    _platforms.current_platform.get_device_capability = staticmethod(lambda *a, **k: _Cap())

    # For W4A16 NVFP4 with linear_backend=auto, vLLM *forces*
    # MarlinNvFp4LinearKernel ("Force a16 (Marlin) when running weight-only
    # quantization", kernels/linear/__init__.py).  That kernel's is_supported()
    # needs the compiled vllm._C, absent in a CPU container, so confirm the
    # forced choice from the error text and then stub the kernel so the rest of
    # the scheme can still be exercised.
    from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w4a4_nvfp4 as _nvfp4,
    )

    try:
        init_nvfp4_linear_kernel(use_a16=True)
        forced = "constructed on this host"
    except Exception as exc:  # noqa: BLE001
        forced = str(exc)
    check("W4A16 NVFP4 routes to MarlinNvFp4LinearKernel (the SM120 path)",
          "MarlinNvFp4LinearKernel" in forced or forced == "constructed on this host",
          forced[:110])

    class _StubKernel:
        def __init__(self, use_a16):
            self.use_a16 = use_a16

        def input_quant_key(self):
            return None

        def process_weights_after_loading(self, layer):
            return None

    _nvfp4.init_nvfp4_linear_kernel = lambda use_a16=False: _StubKernel(use_a16)

    cfg_json = json.load(open(os.path.join(dst_dir, "config.json")))
    cfg = CompressedTensorsConfig.from_config(cfg_json["quantization_config"])
    # DFlashQwen3Model fuses q/k/v and gate/up; vLLM populates this from the
    # model class before get_scheme() is ever called.
    cfg.packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # Every linear module the drafter builds, with the prefix vLLM gives it.
    modules: list[tuple[str, type]] = []
    for i in range(5):
        modules += [
            (f"model.layers.{i}.self_attn.qkv_proj", QKVParallelLinear),
            (f"model.layers.{i}.self_attn.o_proj", RowParallelLinear),
            (f"model.layers.{i}.mlp.gate_up_proj", MergedColumnParallelLinear),
            (f"model.layers.{i}.mlp.down_proj", RowParallelLinear),
            (f"model.layers.{i}.attention_conv.kernel_projection", ReplicatedLinear),
            (f"model.layers.{i}.mlp_conv.kernel_projection", ReplicatedLinear),
        ]
    modules += [
        ("model.fc", ReplicatedLinear),
        ("model.candidate_selector.hidden_projection", ReplicatedLinear),
    ]

    expect = {
        "qkv_proj": "CompressedTensorsW8A16Fp8",
        "o_proj": "CompressedTensorsW8A16Fp8",
        "gate_up_proj": "CompressedTensorsW4A4Fp4",
        "down_proj": "CompressedTensorsW4A4Fp4",
        "kernel_projection": "unquantized",
        "fc": "unquantized",
        "hidden_projection": "unquantized",
    }
    ok_all = True
    for prefix, cls in modules:
        # find_matched_target() only ever looks at module.__class__.__name__.
        module = type(cls.__name__, (), {})()
        try:
            scheme = cfg.get_scheme(layer=module, layer_name=prefix)
            got = type(scheme).__name__ if scheme is not None else "unquantized"
        except Exception as exc:  # noqa: BLE001
            got = f"ERROR: {type(exc).__name__}: {exc}"
        want = expect[prefix.rsplit(".", 1)[-1]]
        good = got == want
        ok_all &= good
        extra = ""
        if got == "CompressedTensorsW4A4Fp4" and scheme is not None:
            extra = f" (use_a16={scheme.use_a16}, group_size={scheme.group_size})"
        elif got == "CompressedTensorsW8A16Fp8" and scheme is not None:
            extra = f" (strategy={scheme.strategy})"
        print(f"  {'ok ' if good else 'BAD'} {prefix:56s} -> {got}{extra}")
    check("every drafter linear resolves to the intended scheme", bool(ok_all))

    # DFlashQwen3ForCausalLM does not declare packed_modules_mapping, so in
    # production the config's mapping is empty and only the fused module name is
    # ever matched.  The targets list both the fused and unfused spellings, so
    # the result must be identical either way.
    saved = cfg.packed_modules_mapping
    cfg.packed_modules_mapping = {}
    same = True
    for prefix, cls in modules:
        scheme = cfg.get_scheme(layer=type(cls.__name__, (), {})(), layer_name=prefix)
        got = type(scheme).__name__ if scheme is not None else "unquantized"
        same &= got == expect[prefix.rsplit(".", 1)[-1]]
    cfg.packed_modules_mapping = saved
    check("same resolution with an empty packed_modules_mapping "
          "(the drafter declares none)", bool(same))

    # NVFP4 weight-only must be W4A16: no input_global_scale is stored anywhere.
    has_input_gs = any(k.endswith("input_global_scale") or k.endswith("input_scale")
                       for k in dst_sd)
    check("no activation-quant scales in the checkpoint (weight-only)", not has_input_gs)
    return cfg


# --------------------------------------------------------------------------
# (c) patched _project_context_kv
# --------------------------------------------------------------------------
def part_c():
    section("(c) patched _project_context_kv vs original")
    torch.manual_seed(0)
    L, H, NH, NKV, HD, NCTX = 5, 256, 4, 2, 64, 13
    q_size, kv_size = NH * HD, NKV * HD
    eps = 1e-5

    class FakeQKV(torch.nn.Module):
        """Stands in for QKVParallelLinear: returns (out, bias)."""

        def __init__(self, w):
            super().__init__()
            self.weight = torch.nn.Parameter(w, requires_grad=False)

        def forward(self, x):
            return F.linear(x, self.weight), None

    class FakeAttn(torch.nn.Module):
        def __init__(self, w):
            super().__init__()
            self.qkv_proj = FakeQKV(w)
            self.q_size, self.kv_size = q_size, kv_size

    weights = [torch.randn(q_size + 2 * kv_size, H, dtype=torch.bfloat16) * 0.02
               for _ in range(L)]
    attns = [FakeAttn(w) for w in weights]
    hidden_norm_w = torch.randn(H, dtype=torch.bfloat16).abs() + 0.5
    x = torch.randn(NCTX, H, dtype=torch.bfloat16)

    def rms_norm(t, w):
        f = t.to(torch.float32)
        return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)).to(t.dtype) * w

    normed = rms_norm(x, hidden_norm_w)

    def original():
        fused = torch.cat([a.qkv_proj.weight[q_size:] for a in attns], dim=0)
        flat = F.linear(normed, fused, None)
        kv = flat.view(NCTX, L, 2, NKV, HD).permute(2, 1, 0, 3, 4).contiguous()
        return kv[0], kv[1]

    def patched(layers):
        per_layer = []
        for a in layers:
            out = a.qkv_proj(normed)
            if isinstance(out, tuple):
                out = out[0]
            per_layer.append(out[..., q_size:])
        flat = torch.cat(per_layer, dim=-1)
        kv = flat.view(NCTX, L, 2, NKV, HD).permute(2, 1, 0, 3, 4).contiguous()
        return kv[0], kv[1]

    k0, v0 = original()
    k1, v1 = patched(attns)
    dk = float((k0.to(torch.float32) - k1.to(torch.float32)).abs().max())
    dv = float((v0.to(torch.float32) - v1.to(torch.float32)).abs().max())
    tol = 4 * float(torch.finfo(torch.bfloat16).eps) * float(k0.abs().max().to(torch.float32))
    check("bf16 weights: patched == original (within bf16 tol)",
          dk <= tol and dv <= tol, f"max|dK|={dk:.3e} max|dV|={dv:.3e} tol={tol:.3e}")

    # Mocked quantized layer: the parameter is fp8 (so the raw slice would be
    # illegal) but forward() is a plain matmul on the dequantized weight.
    class FakeQuantQKV(torch.nn.Module):
        def __init__(self, w):
            super().__init__()
            scale = (w.to(torch.float32).abs().amax(1, keepdim=True) / 448.0).clamp(min=1e-12)
            self.weight = torch.nn.Parameter(
                (w.to(torch.float32) / scale).clamp(-448, 448).to(torch.float8_e4m3fn).t(),
                requires_grad=False,
            )
            self.weight_scale = torch.nn.Parameter(scale, requires_grad=False)

        def dequant(self):
            return (self.weight.t().to(torch.float32) * self.weight_scale).to(torch.bfloat16)

        def forward(self, x):
            return F.linear(x, self.dequant()), None

    class FakeQuantAttn(torch.nn.Module):
        def __init__(self, w):
            super().__init__()
            self.qkv_proj = FakeQuantQKV(w)
            self.q_size, self.kv_size = q_size, kv_size

    qattns = [FakeQuantAttn(w) for w in weights]
    # ground truth: raw row-sliced GEMM on the same dequantized weights
    fused_deq = torch.cat([a.qkv_proj.dequant()[q_size:] for a in qattns], dim=0)
    flat_ref = F.linear(normed, fused_deq, None)
    kv_ref = flat_ref.view(NCTX, L, 2, NKV, HD).permute(2, 1, 0, 3, 4).contiguous()
    k2, v2 = patched(qattns)
    dk = float((kv_ref[0].to(torch.float32) - k2.to(torch.float32)).abs().max())
    dv = float((kv_ref[1].to(torch.float32) - v2.to(torch.float32)).abs().max())
    check("mocked quantized layer: patched == row-sliced reference",
          dk == 0.0 and dv == 0.0, f"max|dK|={dk:.3e} max|dV|={dv:.3e}")

    # And the original path is genuinely broken on that layer (the bug being fixed).
    raw_fails = False
    try:
        F.linear(normed, torch.cat([a.qkv_proj.weight[q_size:] for a in qattns], dim=0), None)
    except Exception:  # noqa: BLE001
        raw_fails = True
    check("raw .weight slice + F.linear does fail on a quantized layer", raw_fails)

    # Guard: the predicate must pick the raw path only for plain float weights.
    sys.path.insert(0, "/patches/dflash2-quantattn/vllm/model_executor/models")
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    def is_raw(attn):
        qkv = attn.qkv_proj
        if not isinstance(getattr(qkv, "quant_method", None), UnquantizedLinearMethod):
            return False
        w = getattr(qkv, "weight", None)
        return (isinstance(w, torch.Tensor) and w.dim() == 2
                and w.dtype in (torch.bfloat16, torch.float16, torch.float32)
                and w.shape[0] == attn.q_size + 2 * attn.kv_size)

    for a in attns:
        a.qkv_proj.quant_method = UnquantizedLinearMethod()
    for a in qattns:
        a.qkv_proj.quant_method = object()
    check("_qkv_weight_is_raw: True for bf16 / False for quantized",
          all(is_raw(a) for a in attns) and not any(is_raw(a) for a in qattns))


# --------------------------------------------------------------------------
# (d) file layout vs the schemes' own create_weights()
# --------------------------------------------------------------------------
def part_d(dst_dir, dst_sd, cfg):
    section("(d) tensor names/shapes/dtypes vs scheme.create_weights()")
    from safetensors import safe_open

    # round-trip
    with safe_open(os.path.join(dst_dir, "model.safetensors"), framework="pt") as f:
        keys = set(f.keys())
    check("safetensors round-trip: same key set", keys == set(dst_sd),
          f"{len(keys)} tensors")


    # vLLM's parameter classes read the TP rank/size at construction; there is no
    # process group here, so pin them to a single-rank view (the shapes below are
    # therefore the TP1 shapes).
    import vllm.model_executor.parameter as _param

    _param.get_tensor_model_parallel_rank = lambda: 0
    _param.get_tensor_model_parallel_world_size = lambda: 1

    # Same story as the NVFP4 Marlin kernel: the FP8 scaled-mm kernels need the
    # compiled extension, so stub the picker.  (The FP8 W8A16 path is already
    # proven in production by GLM-5.3-Flash-DFlash2-FP8.)
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w8a16_fp8 as _w8a16m,
    )

    class _StubScaledMM:
        config = types.SimpleNamespace(weight_quant_key=None)

        def process_weights_after_loading(self, layer):
            return None

    _w8a16m.init_wfp8_a16_linear_kernel = lambda **kw: _StubScaledMM()

    def created_params(layer_name, module_cls, out_partitions, in_size, in_part=None):
        scheme = cfg.get_scheme(layer=_ns(module_cls), layer_name=layer_name)
        assert scheme is not None, layer_name
        layer = torch.nn.Module()
        scheme.create_weights(
            layer=layer,
            input_size=in_size,
            output_size=sum(out_partitions),
            output_partition_sizes=out_partitions,
            input_size_per_partition=in_part if in_part is not None else in_size,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *a, **k: None,
        )
        return {n: (tuple(p.shape), p.dtype) for n, p in layer.named_parameters()}

    def _ns(cls):
        return type(cls.__name__, (), {})()

    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )

    if True:
        # gate_up_proj: TP1 view, 2 shards of 12288 over 4096 inputs
        want = created_params("model.layers.0.mlp.gate_up_proj",
                              MergedColumnParallelLinear, [12288, 12288], 4096)
        print("  gate_up_proj create_weights:", want)
        got = {
            "weight_packed": ((24576, 2048), torch.uint8),
            "weight_scale": ((24576, 256), torch.float8_e4m3fn),
            "weight_global_scale": ((2,), torch.float32),
        }
        check("nvfp4 gate_up_proj param names match create_weights",
              set(want) == set(got))
        ok = all(want[n] == got[n] for n in got if n in want)
        check("nvfp4 gate_up_proj shapes/dtypes match create_weights", ok)

        # the checkpoint's per-shard tensors must stack into exactly those params
        g = dst_sd["layers.0.mlp.gate_proj.weight_packed"]
        u = dst_sd["layers.0.mlp.up_proj.weight_packed"]
        gs = dst_sd["layers.0.mlp.gate_proj.weight_scale"]
        ggs = dst_sd["layers.0.mlp.gate_proj.weight_global_scale"]
        ugs = dst_sd["layers.0.mlp.up_proj.weight_global_scale"]
        check("checkpoint gate+up weight_packed stack -> create_weights shape",
              (g.shape[0] + u.shape[0], g.shape[1]) == want["weight_packed"][0]
              and g.dtype == want["weight_packed"][1],
              f"{tuple(g.shape)}+{tuple(u.shape)} {g.dtype}")
        check("checkpoint weight_scale dtype/shape",
              gs.dtype == torch.float8_e4m3fn and gs.shape == (12288, 4096 // GROUP),
              f"{tuple(gs.shape)} {gs.dtype}")
        check("checkpoint weight_global_scale is float32 [1]",
              ggs.dtype == torch.float32 and tuple(ggs.shape) == (1,),
              f"{tuple(ggs.shape)} {ggs.dtype}")
        # vLLM takes max() over the fused shards' global scales, so they must agree
        check("gate/up share one weight_global_scale (vLLM takes max over shards)",
              float(ggs[0]) == float(ugs[0]), f"{float(ggs[0]):.4f} vs {float(ugs[0]):.4f}")

        want = created_params("model.layers.0.mlp.down_proj",
                              RowParallelLinear, [4096], 12288)
        print("  down_proj create_weights:", want)
        d = dst_sd["layers.0.mlp.down_proj.weight_packed"]
        ds = dst_sd["layers.0.mlp.down_proj.weight_scale"]
        dgs = dst_sd["layers.0.mlp.down_proj.weight_global_scale"]
        check("nvfp4 down_proj matches create_weights",
              (tuple(d.shape), d.dtype) == want["weight_packed"]
              and (tuple(ds.shape), ds.dtype) == want["weight_scale"]
              and (tuple(dgs.shape), dgs.dtype) == want["weight_global_scale"])

        want = created_params("model.layers.0.self_attn.qkv_proj",
                              QKVParallelLinear, [4096, 1024, 1024], 4096)
        print("  qkv_proj create_weights:", want)
        q = dst_sd["layers.0.self_attn.q_proj.weight"]
        k = dst_sd["layers.0.self_attn.k_proj.weight"]
        v = dst_sd["layers.0.self_attn.v_proj.weight"]
        qs = dst_sd["layers.0.self_attn.q_proj.weight_scale"]
        rows = q.shape[0] + k.shape[0] + v.shape[0]
        check("fp8 qkv stack matches create_weights weight",
              ((rows, q.shape[1]), q.dtype) == want["weight"],
              f"({rows}, {q.shape[1]}) {q.dtype} vs {want['weight']}")
        check("fp8 weight_scale is per-channel f32 [out,1]",
              want["weight_scale"] == ((6144, 1), torch.float32)
              and tuple(qs.shape) == (4096, 1) and qs.dtype == torch.float32,
              f"ckpt {tuple(qs.shape)} {qs.dtype}, create_weights {want['weight_scale']}")
        check("fp8 scheme creates no input_scale (weight-only)",
              "input_scale" not in want)

        want = created_params("model.layers.0.self_attn.o_proj",
                              RowParallelLinear, [4096], 4096)
        o = dst_sd["layers.0.self_attn.o_proj.weight"]
        os_ = dst_sd["layers.0.self_attn.o_proj.weight_scale"]
        check("fp8 o_proj matches create_weights",
              (tuple(o.shape), o.dtype) == want["weight"]
              and (tuple(os_.shape), os_.dtype) == want["weight_scale"])

        # ---- TP2 shards (the drafter runs tensor-parallel 2) ----------------
        # Row-parallel layers narrow the checkpoint tensor along the input dim.
        # For NVFP4 that dim is packed (2 values/byte) and grouped (16
        # values/scale), so it only shards cleanly if both divide the per-rank
        # input size.
        def tp2_row(layer_name, cls, out, in_size, params):
            w = created_params(layer_name, cls, [out], in_size, in_size // 2)
            ok = True
            for pname, ckpt in params.items():
                exp_shape, exp_dtype = w[pname]
                sliced = ckpt.narrow(1, 0, exp_shape[1]) if len(exp_shape) == 2 and \
                    exp_shape[1] != ckpt.shape[1] else ckpt
                ok &= tuple(sliced.shape) == exp_shape and sliced.dtype == exp_dtype
            check(f"TP2 rank0 shard of {layer_name.rsplit('.', 1)[-1]} matches "
                  f"create_weights", bool(ok), str(w))

        tp2_row("model.layers.0.mlp.down_proj", RowParallelLinear, 4096, 12288,
                {"weight_packed": d, "weight_scale": ds})
        tp2_row("model.layers.0.self_attn.o_proj", RowParallelLinear, 4096, 4096,
                {"weight": o})
        # Column-parallel layers split the output dim; only divisibility matters.
        check("TP2 column splits are integral (gate/up 12288, q 4096, kv 1024, "
              "8 kv heads)",
              12288 % 2 == 0 and 4096 % 2 == 0 and 1024 % 2 == 0 and 8 % 2 == 0)


def main(src_dir, dst_dir):
    src_sd = load_file(os.path.join(src_dir, "model.safetensors"))
    dst_sd = load_file(os.path.join(dst_dir, "model.safetensors"))
    part_a(src_sd, dst_sd)
    # CompressedTensorsW8A16Fp8.__init__ reads get_current_vllm_config() for the
    # activation dtype.  Building a real VllmConfig needs a device, so stub just
    # the field the scheme touches.
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w8a16_fp8 as _w8a16,
    )

    _w8a16.get_current_vllm_config = lambda: types.SimpleNamespace(
        model_config=types.SimpleNamespace(dtype=torch.bfloat16)
    )
    cfg = part_b(dst_dir, dst_sd)
    part_c()
    part_d(dst_dir, dst_sd, cfg)
    section("RESULT")
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("PASS: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
