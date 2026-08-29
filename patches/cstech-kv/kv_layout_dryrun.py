"""GLM-5.3-Flash + DFlash-2 KV layout dry-run.

Reproduces the REAL specs measured by the KVDIAG unify dump, then cross-checks
the model against three facts observed in the live boot log:

  A) bytes_per_block == 27,163,136
  B) at max_model_len=262144, K=3: needed == 2.56 GiB  (boot error 07:00)
  C) the same boot's binary search reported estimated max len == 114688,
     which pins available memory into a one-block-wide bracket.

Then it prices every candidate design against that model.
"""

import sys
import torch
from types import SimpleNamespace
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec, KpoolTailSpec, MambaSpec, SlidingWindowSpec,
)
import vllm.v1.core.kv_cache_utils as U
from vllm.utils.math_utils import cdiv

FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16
MIB = 2 ** 20
GIB = 2 ** 30

MLA_BLOCK = 3584
MLA_PAGE = 2351104          # KVDIAG: MLAAttentionSpec x11 page=2351104
IDX_PAGE = 118272           # KVDIAG: MLAAttentionSpec x11 page=118272
TAIL_PAGE = 2048            # KVDIAG: KpoolTailSpec x11 block=4 page=2048
SW_BLOCK_RAW = 1136         # KVDIAG: SlidingWindowSpec x5 block=1136
N_MLA = 11
N_KDA = 34
N_SW = 5


def build_spec(with_drafter=True, num_mamba_groups_hint=None):
    spec = {}
    for i in range(N_MLA):
        spec[f"lm.layers.{i}.self_attn.attn"] = MLAAttentionSpec(
            block_size=MLA_BLOCK, num_kv_heads=1, head_size=656, dtype=FP8,
            cache_dtype_str="fp8")
        spec[f"lm.layers.{i}.self_attn.indexer.k_cache"] = MLAAttentionSpec(
            block_size=MLA_BLOCK, num_kv_heads=1, head_size=132, dtype=FP8,
            cache_dtype_str="fp8", compress_ratio=4)
        spec[f"lm.layers.{i}.self_attn.indexer.tail"] = KpoolTailSpec(
            block_size=4, num_kv_heads=1, head_size=128, dtype=BF16,
            sliding_window=2048)
    for i in range(N_KDA):
        # KDA state @ TP2: recurrent 32 heads x 128 x 128 fp32 + conv state.
        # real 2,170,880 B, padded by the platform to the MLA page.
        spec[f"lm.layers.m{i}.mamba"] = MambaSpec(
            block_size=MLA_BLOCK,
            shapes=((32, 128, 128), (12288, 3)),
            dtypes=(torch.float32, BF16),
            mamba_cache_mode="align",
            page_size_padded=MLA_PAGE,
            num_speculative_blocks=K_SPEC_BLOCKS)
    if with_drafter:
        for i in range(N_SW):
            spec[f"drafter.layers.{i}.self_attn.attn"] = SlidingWindowSpec(
                block_size=SW_BLOCK_RAW, num_kv_heads=4, head_size=128,
                dtype=BF16, sliding_window=2048, page_size_padded=MLA_PAGE)
    return spec


def make_cfg(max_model_len, k, mnbt=1024, seqs=2, max_concurrent_batches=2):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=1,
                                        decode_context_parallel_size=1),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None,
                                     block_size=MLA_BLOCK,
                                     enable_prefix_caching=True,
                                     prefix_match_unit=None,
                                     mamba_cache_mode="align"),
        kv_transfer_config=None,
        model_config=SimpleNamespace(max_model_len=max_model_len,
                                     get_total_num_hidden_layers=lambda: 45),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False,
                                         max_num_batched_tokens=mnbt,
                                         max_num_seqs=seqs),
        max_in_flight_tokens=max_concurrent_batches * mnbt,
        speculative_config=(SimpleNamespace(num_speculative_tokens=k)
                            if k else None),
    )


def price(max_model_len, k, with_drafter=True, verbose=False):
    cfg = make_cfg(max_model_len, k)
    spec = build_spec(with_drafter)
    groups = U._get_kv_cache_groups_glm5_next(cfg, spec)
    assert groups is not None, "glm5_next layout rejected"
    bpb = U._pool_bytes_per_block(cfg, groups)
    needed = U._max_memory_usage_bytes_from_groups(cfg, groups)
    per_req = needed // bpb
    # break the per-request block demand down group by group
    lay = U._glm5_next_tensor_layout(groups)
    attn_group, slot_groups, mla_names, idx_names, mla_page, idx_page, tail, _ = lay
    parts = {"attn": attn_group.kv_cache_spec.max_memory_usage_pages(cfg)}
    for g in slot_groups:
        s = g.kv_cache_spec
        kind = type(s).__name__
        n = cdiv(s.max_memory_usage_bytes(cfg), s.page_size_bytes)
        parts[f"{kind}[{len(g.layer_names)}L]"] = parts.get(
            f"{kind}[{len(g.layer_names)}L]", 0) + n
    parts["tail"] = 1 if tail else 0
    if verbose:
        print(f"  groups={len(groups)} bpb={bpb} per_req={per_req} parts={parts}")
    return dict(groups=groups, bpb=bpb, needed=needed, per_req=per_req,
                parts=parts, cfg=cfg, slot_groups=slot_groups,
                n_mla_slots=len(mla_names))


def pool_tokens(num_blocks, per_req, max_model_len):
    return int(num_blocks / per_req * max_model_len)


print("=" * 78)
print("STEP 1 — validate the model against the live boot log")
print("=" * 78)

K_SPEC_BLOCKS = 3
r = price(262144, 3, verbose=True)
print(f"  bytes_per_block = {r['bpb']:,}   (log: 27,163,136)")
assert r["bpb"] == N_MLA * MLA_PAGE + N_MLA * IDX_PAGE == 27163136, r["bpb"]
print(f"  needed @262144  = {r['needed']/GIB:.2f} GiB  (boot error: 2.56 GiB)")
assert f"{r['needed']/GIB:.2f}" == "2.56", r["needed"] / GIB

# (C) the failing boot's binary search said est. max len == 114688; find the
# available-memory bracket that reproduces exactly that.
lo_blocks = price(114688, 3)["per_req"]
hi_blocks = price(114688 + MLA_BLOCK, 3)["per_req"]
lo = lo_blocks * r["bpb"]
hi = hi_blocks * r["bpb"]
print(f"  est_max_len 114688 => avail in [{lo/GIB:.4f}, {hi/GIB:.4f}) GiB "
      f"= [{lo_blocks}, {hi_blocks}) blocks   (log printed 1.49 GiB)")
assert f"{lo/GIB:.2f}" == "1.49"
AVAIL_256K = lo          # tight lower bound on the real available memory
print("  ✓ model reproduces all three observed facts")

print()
print("=" * 78)
print("STEP 2 — where the per-request blocks actually go")
print("=" * 78)
for L in (114688, 131072, 163840, 196608, 262144):
    for k in (0, 1, 2, 3):
        K_SPEC_BLOCKS = k
        rr = price(L, k or None)
        print(f"  L={L:7d} K={k}  per_req={rr['per_req']:4d}  "
              f"needed={rr['needed']/GIB:5.2f} GiB   {rr['parts']}")
    print()

print("=" * 78)
print("STEP 3 — ceiling of design A (sub-slot packing of the KDA/SW pages)")
print("=" * 78)
K_SPEC_BLOCKS = 3
r = price(262144, 3)
slot_pages = 0
for g in r["slot_groups"]:
    s = g.kv_cache_spec
    n = cdiv(s.max_memory_usage_bytes(r["cfg"]), s.page_size_bytes)
    slot_pages += n * len(g.layer_names)
    print(f"  {type(s).__name__:18s} {len(g.layer_names):2d} layers x "
          f"{n} state blocks = {n*len(g.layer_names):3d} pages "
          f"(real {s.real_page_size_bytes/MIB:.2f} MiB, "
          f"padded {s.page_size_bytes/MIB:.2f} MiB)")
today = sum(v for kk, v in r["parts"].items() if kk not in ("attn", "tail"))
ideal = cdiv(slot_pages, r["n_mla_slots"])
print(f"  slot pages required        : {slot_pages}")
print(f"  slots per block id         : {r['n_mla_slots']}")
print(f"  block ids TODAY            : {today}")
print(f"  block ids with PERFECT pack: {ideal}")
print(f"  design-A ceiling           : {today-ideal} blocks "
      f"= {(today-ideal)*r['bpb']/MIB:.0f} MiB / request")
print(f"  per_req {r['per_req']} -> {r['per_req']-(today-ideal)}; "
      f"needed {r['needed']/GIB:.2f} -> "
      f"{(r['per_req']-(today-ideal))*r['bpb']/GIB:.2f} GiB "
      f"vs available {AVAIL_256K/GIB:.2f} GiB")

print()
print("=" * 78)
print("STEP 4 — the hard floor: MLA context alone at 256k")
print("=" * 78)
ctx_blocks = price(262144, 3)["parts"]["attn"]
print(f"  MLA+indexer context for ONE 262144-token request: {ctx_blocks} blocks"
      f" = {ctx_blocks*r['bpb']/GIB:.2f} GiB")
print(f"  available KV memory at 262144 with the drafter   : "
      f"{AVAIL_256K/GIB:.2f} GiB")
print(f"  => 256k is infeasible even with a ZERO-cost KDA layout "
      f"(short by {(ctx_blocks*r['bpb']-AVAIL_256K)/GIB:.2f} GiB)")

print()
print("=" * 78)
print("STEP 5 — pool tokens under each option (avail measured per ctx)")
print("=" * 78)
# available KV memory measured at util 0.988 with the K3 drafter loaded
AVAIL = {114688: 2.22 * GIB, 163840: 2.25 * GIB, 262144: AVAIL_256K}
for L, avail in AVAIL.items():
    nb = int(avail // 27163136)
    row = []
    for label, k, spec_blocks, pack in (
        ("baseline        ", 3, 3, False),
        ("spec_blocks=1   ", 3, 1, False),
        ("design A        ", 3, 3, True),
        ("A + spec_blocks1", 3, 1, True),
    ):
        K_SPEC_BLOCKS = spec_blocks
        rr = price(L, k)
        pr = rr["per_req"]
        if pack:
            sp = 0
            for g in rr["slot_groups"]:
                s = g.kv_cache_spec
                n = cdiv(s.max_memory_usage_bytes(rr["cfg"]), s.page_size_bytes)
                sp += n * len(g.layer_names)
            cur = sum(v for kk, v in rr["parts"].items()
                      if kk not in ("attn", "tail"))
            pr = pr - cur + cdiv(sp, rr["n_mla_slots"])
        ok = pr <= nb
        row.append(f"{label} per_req={pr:3d} "
                   f"{'ADMIT' if ok else 'FAIL '} "
                   f"pool={pool_tokens(nb, pr, L) if ok else 0:7,}")
    print(f"  L={L:7d} avail={avail/GIB:.2f} GiB num_blocks={nb}")
    for x in row:
        print(f"      {x}")

print()
print("=" * 78)
print("STEP 6 — no-drafter regression must stay byte-identical")
print("=" * 78)
K_SPEC_BLOCKS = 0
cfg = make_cfg(262144, None)
spec = build_spec(with_drafter=False)
g2 = U._get_kv_cache_groups_glm5_next(cfg, spec)
print(f"  groups={len(g2)} bytes/block={U._pool_bytes_per_block(cfg, g2):,} "
      f"per_req={U._max_memory_usage_bytes_from_groups(cfg, g2)//27163136}")
kvc2 = U.get_kv_cache_config_from_groups(cfg, g2, int(2.84 * GIB))
s2, h2 = U.resolve_kv_cache_block_sizes(kvc2, cfg)
print(f"  num_blocks={kvc2.num_blocks} sched_block={s2} hash_block={h2} "
      f"pool={pool_tokens(kvc2.num_blocks, U._max_memory_usage_bytes_from_groups(cfg,g2)//27163136, 262144):,}")
print("OK")
