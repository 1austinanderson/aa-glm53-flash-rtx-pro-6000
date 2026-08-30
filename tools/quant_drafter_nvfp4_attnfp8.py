"""Mixed NVFP4 (MLP) + FP8 (attention) quantization of the DFlash2 drafter.

MLP  gate/up/down_proj -> compressed-tensors 'nvfp4-pack-quantized' W4A16:
        <name>.weight_packed        uint8            [out, in // 2]   (2 e2m1 nibbles/byte)
        <name>.weight_scale         float8_e4m3fn    [out, in // 16]  (per-group-of-16)
        <name>.weight_global_scale  float32          [1]              (CT divisor: 448*6/amax)
attn q/k/v/o_proj      -> compressed-tensors 'float-quantized'  W8A16 per-channel:
        <name>.weight               float8_e4m3fn    [out, in]
        <name>.weight_scale         float32          [out, 1]
Everything else (attention_conv.*, mlp_conv.*, candidate_selector.*, fc, norms) stays bf16.

Dequantization contracts implemented here are the ones vLLM applies at load time:
  nvfp4: w ~= e2m1(nibble) * float(weight_scale) / weight_global_scale
         (CompressedTensorsW4A4Fp4.process_weights_after_loading stores 1/weight_global_scale)
  fp8  : w ~= float(weight) * weight_scale

Fused-module note: vLLM merges gate_proj+up_proj into gate_up_proj and takes
max() over the two weight_global_scale values, so gate and up MUST share one
global scale or the smaller-amax shard is dequantized with the wrong divisor.
q/k/v are FP8 per-channel, which has no shared global scale, so no such
constraint applies to attention.

Usage: python3 quant_drafter_nvfp4_attnfp8.py <bf16_src_dir> <dst_dir>
"""

import json
import os
import re
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0
FP4_MAX = 6.0
GROUP = 16

MLP_RE = re.compile(r"^layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$")
ATTN_RE = re.compile(r"^layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight$")


def cast_to_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest E2M1 value (verbatim thresholds from vLLM's
    nvfp4_emulation_utils.cast_to_fp4 -> round-half-to-even)."""
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e2m1_to_nibble(v: torch.Tensor) -> torch.Tensor:
    """Map exact E2M1 float values to their 4-bit codes (sign bit = 0x08)."""
    mag = torch.abs(v)
    code = torch.zeros_like(mag, dtype=torch.uint8)
    for i, val in enumerate(_E2M1_VALUES.tolist()):
        code = torch.where(mag == val, torch.full_like(code, i), code)
    neg = (v < 0) | ((v == 0) & (torch.signbit(v)))
    # -0.0 is encoded as +0.0; only nonzero magnitudes carry the sign bit.
    neg = neg & (mag > 0)
    return code | (neg.to(torch.uint8) << 3)


def nvfp4_quantize(w: torch.Tensor, global_scale: float):
    """w: [out, in] float32.  Returns (packed uint8 [out, in//2],
    group scales float8_e4m3fn [out, in//16])."""
    out, inp = w.shape
    assert inp % GROUP == 0
    gs = torch.tensor(global_scale, dtype=torch.float32)
    x = w.to(torch.float32).reshape(out, inp // GROUP, GROUP)
    vec_max = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(gs * (vec_max / FP4_MAX), max=FP8_MAX)
    scale_fp8 = scale.to(torch.float8_e4m3fn)
    scale_f32 = scale_fp8.to(torch.float32)
    # 1/(scale/gs), zero-safe (all-zero groups get scale 0 -> quantize to 0)
    recip = 1.0 / (scale_f32 + (scale_f32 == 0) * 1e8)
    q = cast_to_fp4(torch.clamp(x * (recip * gs), -FP4_MAX, FP4_MAX))
    nib = e2m1_to_nibble(q.reshape(out, inp))
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()
    return packed, scale_fp8.squeeze(-1).contiguous()


def nvfp4_dequantize(packed: torch.Tensor, scale_fp8: torch.Tensor, global_scale: float):
    """Inverse of nvfp4_quantize, matching vLLM's break_fp4_bytes ordering."""
    out, packed_in = packed.shape
    low = (packed & 0x0F).to(torch.long)
    high = ((packed >> 4) & 0x0F).to(torch.long)
    nib = torch.stack((low, high), dim=2).reshape(out, packed_in * 2)
    mag = _E2M1_VALUES[nib & 0x07]
    vals = mag * torch.where((nib & 0x08).bool(), -1.0, 1.0)
    s = scale_fp8.to(torch.float32).unsqueeze(-1) / global_scale
    return (vals.reshape(out, -1, GROUP) * s).reshape(out, packed_in * 2)


def fp8_quantize(w: torch.Tensor):
    f = w.to(torch.float32)
    scale = (f.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
    q = (f / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.contiguous(), scale.to(torch.float32).contiguous()


QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "mixed-precision",
    "quantization_status": "compressed",
    "kv_cache_scheme": None,
    "config_groups": {
        # MLP: NVFP4 weight-only.  vLLM resolves this to
        # CompressedTensorsW4A4Fp4(use_a16=True) via _is_nvfp4_format().
        "group_0_mlp_nvfp4": {
            "format": "nvfp4-pack-quantized",
            "targets": [r"re:.*\.mlp\.(gate_proj|up_proj|down_proj|gate_up_proj)$"],
            "weights": {
                "num_bits": 4,
                "type": "float",
                "strategy": "tensor_group",
                "group_size": 16,
                "symmetric": True,
                "dynamic": False,
            },
            "input_activations": None,
        },
        # Attention projections: FP8 per-channel weight-only ->
        # CompressedTensorsW8A16Fp8.
        "group_1_attn_fp8": {
            "format": "float-quantized",
            "targets": [r"re:.*\.self_attn\.(q_proj|k_proj|v_proj|o_proj|qkv_proj)$"],
            "weights": {
                "num_bits": 8,
                "type": "float",
                "strategy": "channel",
                "symmetric": True,
                "dynamic": False,
            },
            "input_activations": None,
        },
    },
    # Left in bf16: the DFlash conv kernels, the candidate selector, the
    # aux-hidden-state fusion `fc`, all norms, and the (absent) embedding.
    "ignore": [
        "lm_head",
        "re:.*embed_tokens.*",
        "re:.*candidate_selector.*",
        "re:.*attention_conv.*",
        "re:.*mlp_conv.*",
        r"re:.*\.fc$",
        "fc",
        "re:.*norm.*",
    ],
}


def main(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    W = load_file(os.path.join(src, "model.safetensors"))

    # One shared NVFP4 global scale per *fused* module: (layer, "gate_up") and
    # (layer, "down").  See the fused-module note in the docstring.
    amax: dict[tuple[str, str], float] = {}
    for k, v in W.items():
        m = MLP_RE.match(k)
        if m:
            key = (m.group(1), "gate_up" if m.group(2) in ("gate", "up") else "down")
            amax[key] = max(amax.get(key, 0.0), float(v.to(torch.float32).abs().max()))
    gscale = {k: (FP8_MAX * FP4_MAX) / a for k, a in amax.items()}

    out: dict[str, torch.Tensor] = {}
    bytes_mlp = bytes_attn = bytes_other = 0
    n_nvfp4 = n_fp8 = 0
    for k, v in W.items():
        m = MLP_RE.match(k)
        a = ATTN_RE.match(k)
        base = k[: -len(".weight")]
        if m:
            key = (m.group(1), "gate_up" if m.group(2) in ("gate", "up") else "down")
            gs = gscale[key]
            packed, scale = nvfp4_quantize(v.to(torch.float32), gs)
            out[base + ".weight_packed"] = packed
            out[base + ".weight_scale"] = scale
            out[base + ".weight_global_scale"] = torch.tensor([gs], dtype=torch.float32)
            bytes_mlp += packed.numel() + scale.numel() + 4
            n_nvfp4 += 1
        elif a:
            q, scale = fp8_quantize(v)
            out[base + ".weight"] = q
            out[base + ".weight_scale"] = scale
            bytes_attn += q.numel() + scale.numel() * 4
            n_fp8 += 1
        else:
            out[k] = v
            bytes_other += v.numel() * v.element_size()

    save_file(out, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})

    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["quantization_config"] = QUANT_CONFIG
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)

    for f in os.listdir(src):
        if f in ("model.safetensors", "config.json") or f.startswith("."):
            continue
        p = os.path.join(src, f)
        if os.path.isfile(p):
            shutil.copy(p, dst)

    g = 2**30
    print(f"nvfp4 tensors: {n_nvfp4}   fp8 tensors: {n_fp8}   passthrough: {len(W) - n_nvfp4 - n_fp8}")
    print(f"mlp   {bytes_mlp / g:.4f} GiB")
    print(f"attn  {bytes_attn / g:.4f} GiB")
    print(f"other {bytes_other / g:.4f} GiB")
    print(f"total {(bytes_mlp + bytes_attn + bytes_other) / g:.4f} GiB")
    print("global scales:", {f"{k[0]}.{k[1]}": round(v, 3) for k, v in sorted(gscale.items())})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
