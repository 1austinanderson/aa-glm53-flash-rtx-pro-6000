"""Weight-only FP8 (W8A16, per-output-channel scale) quantization of the DFlash-2 drafter.
Compressed-tensors 'float-quantized' layout: <name>.weight (float8_e4m3fn) + <name>.weight_scale (fp32 [out,1]).
Only mlp.{gate,up,down}_proj are quantized (MLP-only); attention projections MUST stay bf16 because
qwen3_dflash.py:_project_context_kv does F.linear on the raw qkv_proj weight. The ignore list below also names
self_attn/qkv_proj/o_proj explicitly: compressed-tensors applies the scheme by `targets: [Linear]`, so an unlisted
attention layer would be built as an fp8 layer even though its tensors are bf16.
A first cut with dynamic per-token A8 activations lost draft acceptance (1.7 -> 1.29 extra tokens/step); W8A16 is lossless.
Usage: python quant_drafter_fp8.py <bf16 drafter dir> <output dir>"""
import json, os, re, shutil, sys, torch
from safetensors.torch import load_file, save_file
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
W = load_file(os.path.join(src, "model.safetensors"))
TARGET = re.compile(r"^layers\.\d+\.mlp\.(gate|up|down)_proj\.weight$")
FMAX = torch.finfo(torch.float8_e4m3fn).max
out, nq, saved = {}, 0, 0
for k, v in W.items():
    if TARGET.match(k):
        w = v.to(torch.float32)
        scale = (w.abs().amax(dim=1, keepdim=True) / FMAX).clamp(min=1e-12)
        q = (w / scale).clamp(-FMAX, FMAX).to(torch.float8_e4m3fn)
        out[k] = q.contiguous(); out[k.replace(".weight", ".weight_scale")] = scale.to(torch.float32).contiguous()
        nq += 1; saved += v.numel()
    else:
        out[k] = v
save_file(out, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
cfg = json.load(open(os.path.join(src, "config.json")))
cfg["quantization_config"] = {
    "quant_method": "compressed-tensors", "format": "float-quantized", "quantization_status": "compressed",
    "kv_cache_scheme": None,
    "config_groups": {"group_0": {
        "targets": ["Linear"],
        "weights": {"num_bits": 8, "type": "float", "strategy": "channel", "symmetric": True, "dynamic": False},
        "input_activations": None}},
    "ignore": ["lm_head", "re:.*embed_tokens.*", "re:.*candidate_selector.*", "re:.*attention_conv.*", "re:.*mlp_conv.*", "re:.*\\.fc$", "fc", "re:.*norm.*",
               "re:.*self_attn.*", "re:.*qkv_proj.*", "re:.*o_proj.*"],
}
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
for f in os.listdir(src):
    if f not in ("model.safetensors", "config.json") and not f.startswith("."):
        p = os.path.join(src, f); (shutil.copy(p, dst) if os.path.isfile(p) else None)
print(f"quantized {nq} tensors, {saved/1e9:.2f}B params -> fp8; bf16 bytes saved ~{saved/2**30:.2f} GiB total")
