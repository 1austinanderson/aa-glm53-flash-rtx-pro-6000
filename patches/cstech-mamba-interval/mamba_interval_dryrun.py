"""CPU dry-run for the KDA state-checkpoint interval overlay.

Overlay under test:
  vllm/platforms/interface.py  ->  in `mamba_cache_mode == "align"`, honour a
  user-specified `--mamba-block-size` that is a positive multiple of the
  (final) attention block size, instead of unconditionally forcing it to the
  attention block size.

Two independent checks, both CPU-only (no GPU, no engine):

  PART 1 -- the overlay's decision table, executed from the *shipped source
            text* of interface.py (the align branch is sliced out of the file
            and exec'd against a stub cache_config), vs the pristine file.
  PART 2 -- the whole downstream stack with MambaSpec.block_size = 14336:
            GLM-5-Next grouping, resolve_kv_cache_block_sizes (scheduler block
            size / tokens_per_hash), build_offloading_config, CPUOffloadingSpec,
            resolve_mamba_align_size, SchedulerOffloadConfig.from_spec, GPU
            admission bytes, and the host-RAM admission cost per prompt.

Run inside the serving image (CPU only):
  docker run --rm --entrypoint python3 <mounts> <image> /work/mamba_interval_dryrun.py
"""

import re
import sys
import textwrap
from types import SimpleNamespace

import torch

import vllm.v1.core.kv_cache_utils as U
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheTensor,
    KpoolTailSpec,
    MLAAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

FP8, BF16 = torch.float8_e4m3fn, torch.bfloat16
MLA_BLOCK, MLA_PAGE, IDX_PAGE = 3584, 2351104, 118272
SW_BLOCK_RAW = 1136
N_MLA, N_KDA, N_SW = 11, 34, 5
K_SPEC_BLOCKS = 3

HERE = "/work"  # dir the overlay + pristine are mounted at inside the box
BASELINE_MAMBA_BLOCK = MLA_BLOCK
PROPOSED_MAMBA_BLOCK = 14336

FAILURES: list[str] = []


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got!r}" + ("" if ok else f" (want {want!r})"))
    if not ok:
        FAILURES.append(f"{name}: got {got!r} want {want!r}")
    return ok


# --------------------------------------------------------------------------
# PART 1 -- exec the align branch straight out of the shipped source text
# --------------------------------------------------------------------------
ALIGN_START = re.compile(r'^\s*if cache_config\.mamba_cache_mode == "align":\s*$')
ALIGN_END = re.compile(r"^\s*# Pad mamba page size to exactly match attention page size")


def slice_align_branch(path: str) -> str:
    lines = open(path).read().splitlines()
    starts = [i for i, ln in enumerate(lines) if ALIGN_START.match(ln)]
    ends = [i for i, ln in enumerate(lines) if ALIGN_END.match(ln)]
    # the align branch we care about is the LAST such `if` before the padding
    # comment (the earlier one at the top of the function is the "all" test).
    end = ends[-1]
    start = max(i for i in starts if i < end)
    return textwrap.dedent("\n".join(lines[start:end]))


class _StubLogger:
    def __init__(self):
        self.info_msgs: list[str] = []
        self.warning_msgs: list[str] = []

    def info(self, fmt, *a):
        self.info_msgs.append(fmt % a)

    def warning(self, fmt, *a):
        self.warning_msgs.append(fmt % a)


def run_align_branch(src: str, block_size: int, user_mamba_block_size):
    """Execute the sliced branch. `user_mamba_block_size` is the value the
    enclosing function computed: `cache_config.mamba_block_size` if the user
    set it, else None."""
    cache_config = SimpleNamespace(
        mamba_cache_mode="align",
        block_size=block_size,
        mamba_block_size=user_mamba_block_size,
        user_specified_mamba_block_size=user_mamba_block_size is not None,
    )
    log = _StubLogger()
    ns = {"cache_config": cache_config, "mamba_block_size": user_mamba_block_size,
          "logger": log}
    exec(compile(src, "<align-branch>", "exec"), ns, ns)
    return cache_config.mamba_block_size, log


def part1():
    print("=" * 78)
    print("PART 1 -- align-branch decision table (exec'd from the shipped source)")
    pristine = slice_align_branch(f"{HERE}/interface.pristine.py")
    patched = slice_align_branch(f"{HERE}/interface.py")
    print("-" * 78)
    print("pristine branch source:")
    print(textwrap.indent(pristine, "    | "))
    print("-" * 78)

    # (a) default behaviour must be BYTE-IDENTICAL when the user says nothing
    #     (and when the user asks for exactly the attention block size).
    for user_val in (None, MLA_BLOCK):
        p_out, _ = run_align_branch(pristine, MLA_BLOCK, user_val)
        n_out, nlog = run_align_branch(patched, MLA_BLOCK, user_val)
        check(f"default-identical (user={user_val}) pristine==patched",
              (p_out, n_out), (MLA_BLOCK, MLA_BLOCK))
        check(f"default-identical (user={user_val}) no log noise",
              (nlog.info_msgs, nlog.warning_msgs), ([], []))

    # (b) the lever: a positive multiple is honoured
    for mult in (2, 4, 8):
        want = MLA_BLOCK * mult
        n_out, nlog = run_align_branch(patched, MLA_BLOCK, want)
        check(f"honours {mult}x block_size ({want})", n_out, want)
        check(f"honours {mult}x logs once", len(nlog.info_msgs), 1)
        p_out, _ = run_align_branch(pristine, MLA_BLOCK, want)
        check(f"pristine would have clamped {want} -> {MLA_BLOCK}", p_out, MLA_BLOCK)

    # (c) rejects everything that is not a positive multiple
    for bad in (14337, 1792, 3585, 100):
        n_out, nlog = run_align_branch(patched, MLA_BLOCK, bad)
        check(f"rejects non-multiple {bad} -> attention block size", n_out, MLA_BLOCK)
        check(f"rejects non-multiple {bad} warns", len(nlog.warning_msgs), 1)

    # (d) the block_size the branch aligns to is the FINAL one (the function
    #     may have just raised cache_config.block_size to attn_block_size).
    n_out, _ = run_align_branch(patched, 3584, 14336)
    check("aligns against the final (raised) block_size", n_out, 14336)
    n_out, _ = run_align_branch(patched, 3328, 14336)  # K=0 block size
    check("14336 is NOT a multiple of 3328 -> clamped", n_out, 3328)
    n_out, _ = run_align_branch(patched, 3328, 13312)  # 4 x 3328
    check("13312 == 4 x 3328 honoured at K=0 block size", n_out, 13312)


# --------------------------------------------------------------------------
# PART 2 -- the downstream stack (adapted from cstech-offload/kv_offload_dryrun)
# --------------------------------------------------------------------------
def build_layer_specs(mamba_block: int, with_drafter=True):
    spec = {}
    for i in range(N_MLA):
        spec[f"lm.layers.{i}.self_attn.attn"] = MLAAttentionSpec(
            block_size=MLA_BLOCK, num_kv_heads=1, head_size=656, dtype=FP8,
            cache_dtype_str="fp8")
        spec[f"lm.layers.{i}.self_attn.indexer.k_cache"] = MLAAttentionSpec(
            block_size=MLA_BLOCK, num_kv_heads=1, head_size=132, dtype=FP8,
            cache_dtype_str="fp8", compress_ratio=4)
        spec[f"lm.layers.{i}.self_attn.indexer.tail_cache"] = KpoolTailSpec(
            block_size=4, num_kv_heads=1, head_size=128, dtype=BF16,
            sliding_window=2048)
    for i in range(N_KDA):
        # Real KDA state (MambaStateShapeCalculator.kda_state_shape,
        # linear_num_heads=64, linear_head_dim=128, conv_kernel=4, tp=2,
        # num_spec=3): conv (12288, 6) bf16 = 147,456 B + recurrent
        # (32,128,128) fp32 = 2,097,152 B -> real page 2,244,608 B, padded to
        # the 2,351,104 B MLA slot page. The state size does NOT depend on the
        # checkpoint interval -- that is the whole point of the lever.
        spec[f"lm.layers.m{i}.mamba"] = MambaSpec(
            block_size=mamba_block, shapes=((12288, 6), (32, 128, 128)),
            dtypes=(BF16, torch.float32), mamba_cache_mode="align",
            page_size_padded=MLA_PAGE, num_speculative_blocks=K_SPEC_BLOCKS)
    if with_drafter:
        for i in range(N_SW):
            spec[f"drafter.layers.{i}.self_attn.attn"] = SlidingWindowSpec(
                block_size=SW_BLOCK_RAW, num_kv_heads=4, head_size=128,
                dtype=BF16, sliding_window=2048, page_size_padded=MLA_PAGE)
    return spec


def group_worker_bytes(kv_cache_config) -> list[int]:
    from dataclasses import replace as _replace

    from vllm.v1.kv_cache_interface import AttentionSpec as _A
    from vllm.v1.kv_cache_interface import MambaSpec as _M
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs as _U

    out: list[int] = []
    for g in kv_cache_config.kv_cache_groups:
        gs = g.kv_cache_spec
        per_layer = gs.kv_cache_specs if isinstance(gs, _U) else {}
        total = 0
        for ln in g.layer_names:
            s = per_layer.get(ln, gs)
            if isinstance(s, _A):
                total += s.unpadded_page_size_bytes
            elif isinstance(s, _M):
                total += _replace(s, page_size_padded=None).page_size_bytes
            else:
                raise NotImplementedError(type(s).__name__)
        out.append(total)
    return out


def make_vllm_config(mamba_block: int, offload_gib=32.0, max_model_len=393216):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, decode_context_parallel_size=1,
            prefill_context_parallel_size=1, tensor_parallel_size=2,
            world_size=2, rank=0, data_parallel_index=0, data_parallel_size=1,
            data_parallel_rank_local=None, distributed_executor_backend="mp",
            nnodes_within_dp=1),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None, block_size=MLA_BLOCK,
            enable_prefix_caching=True, prefix_match_unit=None,
            mamba_cache_mode="align", cache_dtype="fp8",
            mamba_block_size=mamba_block),
        kv_transfer_config=SimpleNamespace(
            engine_id="dryrun",
            kv_connector_extra_config={"cpu_bytes_to_use": int(offload_gib * 2**30)}),
        kv_events_config=None,
        model_config=SimpleNamespace(
            max_model_len=max_model_len, model="glm53", dtype=BF16, use_mla=True,
            get_total_num_hidden_layers=lambda: 45,
            get_total_num_kv_heads=lambda: 1),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
            max_num_batched_tokens=1024, max_num_seqs=8),
        max_in_flight_tokens=8192,
        speculative_config=SimpleNamespace(
            num_speculative_tokens=K_SPEC_BLOCKS,
            use_eagle=lambda: True, method="dflash"),
        use_v2_model_runner=False,
    )


def scheduler_view(groups):
    import copy
    out = copy.deepcopy(groups)
    for g in out:
        if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs):
            g.kv_cache_spec = next(iter(g.kv_cache_spec.kv_cache_specs.values()))
    return out


def round_up(x, m):
    return ((x + m - 1) // m) * m


def analyse(mamba_block: int, with_drafter=True, verbose=True):
    """Return a dict of everything that matters for one mamba_block_size."""
    cfg = make_vllm_config(mamba_block)
    layer_specs = build_layer_specs(mamba_block, with_drafter)
    groups = U._get_kv_cache_groups_glm5_next(cfg, layer_specs)
    assert groups is not None, "glm5_next layout rejected"

    num_blocks = 130
    tensors = []
    lay = U._glm5_next_tensor_layout(groups)
    _, _, mla_names, idx_names, mla_page, idx_page, tail_names, _ = lay
    for n in mla_names:
        tensors.append(KVCacheTensor(size=num_blocks * mla_page, shared_by=[n]))
    for n in idx_names:
        tensors.append(KVCacheTensor(size=num_blocks * idx_page, shared_by=[n]))
    worker_cfg = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=tensors,
                               kv_cache_groups=groups)
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=tensors,
        kv_cache_groups=scheduler_view(groups))

    sched_bs, tph = U.resolve_kv_cache_block_sizes(kv_cache_config, cfg)

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
        build_offloading_config,
    )
    ocfg = build_offloading_config(cfg, kv_cache_config)
    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
    spec = CPUOffloadingSpec(ocfg)
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        SchedulerOffloadConfig, resolve_mamba_align_size,
    )
    align = resolve_mamba_align_size(spec, kv_cache_config)
    sc = SchedulerOffloadConfig.from_spec(spec, cfg, kv_cache_config)

    # GPU-side admission (must be identical across the two settings)
    pool_bpb = U._pool_bytes_per_block(cfg, groups)
    admission = U._max_memory_usage_bytes_from_groups(cfg, groups)
    mamba_specs = [g.kv_cache_spec for g in kv_cache_config.kv_cache_groups
                   if isinstance(g.kv_cache_spec, MambaSpec)]
    mamba_admit = mamba_specs[0].max_memory_usage_bytes(cfg)
    mamba_rowlen = mamba_specs[0].max_num_blocks_per_req(
        cfg, cfg.model_config.max_model_len)

    real = group_worker_bytes(worker_cfg)
    nw = sc.num_workers
    charged = spec.kv_bytes_per_chunk

    if verbose:
        print(f"KV cache groups (scheduler view), mamba_block_size={mamba_block}")
        for i, g in enumerate(kv_cache_config.kv_cache_groups):
            s = g.kv_cache_spec
            print(f"  [{i}] {type(s).__name__:<20} block={s.block_size:<6} "
                  f"layers={len(g.layer_names):<3} "
                  f"prefix_caching={s.participates_in_prefix_caching}")
        print(f"  scheduler_block_size={sched_bs}  tokens_per_hash={tph}")
        print(f"  worker_kv_bytes_per_block={ocfg.worker_kv_bytes_per_block:,}")
        print(f"  CPUOffloadingSpec: num_blocks={spec.num_blocks} "
              f"kv_bytes_per_chunk={spec.kv_bytes_per_chunk:,} "
              f"({spec.kv_bytes_per_chunk / 2**20:.1f} MiB)")
        print(f"  resolve_mamba_align_size -> {align}")
        print("  SchedulerOffloadConfig.kv_group_configs:")
        for i, c in enumerate(sc.kv_group_configs):
            print(f"    [{i}] tpb={c.tokens_per_block:<6} tpc={c.tokens_per_chunk:<6} "
                  f"hpc={c.hashes_per_chunk:<3} sw_chunks={c.sliding_window_size_in_chunks} "
                  f"align_chunks={c.alignment_chunk_count} eagle={c.is_eagle_group} "
                  f"cow={c.requires_cow_source} inert={c.is_inert} "
                  f"real_bytes/worker={0 if c.is_inert else real[i]:,}")
        print(f"  blocks_per_chunk={sc.blocks_per_chunk} "
              f"tokens_per_hash={sc.tokens_per_hash} num_workers={sc.num_workers} "
              f"supports_partial_tail={sc.supports_partial_tail}")
        print(f"  GPU: pool_bytes_per_block={pool_bpb:,}  "
              f"admission@{cfg.model_config.max_model_len}={admission/2**30:.3f} GiB")
        print(f"  GPU: mamba max_memory_usage_bytes={mamba_admit:,} "
              f"({mamba_admit // MLA_PAGE} pages)  "
              f"mamba block-table row length={mamba_rowlen}")

    # host-RAM admission per prompt
    def ram(prompt_tokens):
        n = 0
        b = 0
        per_group = []
        for i, c in enumerate(sc.kv_group_configs):
            if c.is_inert:
                per_group.append(0)
                continue
            k = prompt_tokens // c.tokens_per_chunk
            per_group.append(k)
            n += k
            b += k * charged
        return n, b, per_group

    return dict(
        mamba_block=mamba_block, sched_bs=sched_bs, tph=tph, align=align,
        spec=spec, sc=sc, real=real, nw=nw, charged=charged,
        pool_bpb=pool_bpb, admission=admission, mamba_admit=mamba_admit,
        mamba_rowlen=mamba_rowlen, ram=ram,
        group_kinds=[type(g.kv_cache_spec).__name__
                     for g in kv_cache_config.kv_cache_groups],
    )


def part2(with_drafter=True):
    print("=" * 78)
    print(f"PART 2 -- downstream stack, drafter={with_drafter}")
    print("-" * 78)
    base = analyse(BASELINE_MAMBA_BLOCK, with_drafter)
    print("-" * 78)
    new = analyse(PROPOSED_MAMBA_BLOCK, with_drafter)
    print("-" * 78)

    print("INVARIANTS that must NOT move:")
    check("tokens_per_hash", new["tph"], base["tph"])
    check("GPU pool bytes per block", new["pool_bpb"], base["pool_bpb"])
    check("GPU admission bytes @ max_model_len", new["admission"], base["admission"])
    check("mamba GPU max_memory_usage_bytes", new["mamba_admit"], base["mamba_admit"])
    check("CPU chunk bytes charged", new["charged"], base["charged"])
    check("CPU pool chunk count", new["spec"].num_blocks, base["spec"].num_blocks)
    check("group kinds/order", new["group_kinds"], base["group_kinds"])
    check("inert set", [c.is_inert for c in new["sc"].kv_group_configs],
          [c.is_inert for c in base["sc"].kv_group_configs])
    check("SW group alignment_chunk_count",
          [c.alignment_chunk_count for c in new["sc"].kv_group_configs],
          [c.alignment_chunk_count for c in base["sc"].kv_group_configs])

    print("DIVISIBILITY / well-formedness at 14336:")
    check("14336 % attention block 3584", PROPOSED_MAMBA_BLOCK % MLA_BLOCK, 0)
    check("14336 % tokens_per_hash", PROPOSED_MAMBA_BLOCK % new["tph"], 0)
    check("14336 % DFlash SW block 896", PROPOSED_MAMBA_BLOCK % 896, 0)
    for c in new["sc"].kv_group_configs:
        check(f"group {c.group_idx} hashes_per_chunk integral & >0",
              c.hashes_per_chunk > 0
              and c.tokens_per_chunk % new["sc"].tokens_per_hash == 0, True)

    print("THINGS THAT MOVE (expected):")
    print(f"  scheduler_block_size      {base['sched_bs']:>7} -> {new['sched_bs']:>7}"
          f"   (LCM over groups; prefix hits / chunk alignment coarsen)")
    print(f"  mamba offload align size  {base['align']:>7} -> {new['align']:>7}"
          f"   (external hit rounded down to this)")
    print(f"  mamba block-table row     {base['mamba_rowlen']:>7} -> "
          f"{new['mamba_rowlen']:>7}   (per mamba group, per request)")

    print("-" * 78)
    print("HOST RAM admitted per prompt (all workers, fixed 54 MB CPU chunk):")
    hdr = f"  {'prompt':>7} | {'chunks now':>10} {'GiB now':>8} | " \
          f"{'chunks new':>10} {'GiB new':>8} | {'saved':>7} {'ratio':>6}"
    print(hdr)
    for P in (14336, 30000, 57000, 60000, 120000):
        n0, b0, g0 = base["ram"](P)
        n1, b1, g1 = new["ram"](P)
        print(f"  {P:>7} | {n0:>10} {b0/2**30:>8.2f} | {n1:>10} {b1/2**30:>8.2f} | "
              f"{(b0-b1)/2**30:>7.2f} {b0/max(b1,1):>6.2f}x")
    print("  per-group chunk counts @60000 tokens:")
    n0, b0, g0 = base["ram"](60000)
    n1, b1, g1 = new["ram"](60000)
    for i, c in enumerate(base["sc"].kv_group_configs):
        kind = base["group_kinds"][i]
        print(f"    [{i}] {kind:<20} {g0[i]:>4} -> {g1[i]:>4} chunks "
              f"({g0[i]*base['charged']/2**30:.2f} -> "
              f"{g1[i]*new['charged']/2**30:.2f} GiB)"
              + ("  INERT" if c.is_inert else ""))
    print(f"  pool depth @ {new['spec'].num_blocks} chunks: "
          f"{base['spec'].num_blocks/max(n0,1):.2f} -> "
          f"{new['spec'].num_blocks/max(n1,1):.2f} x 60k prompts")

    print("-" * 78)
    print("Worst-case KDA recompute after a prefix hit (tokens of linear-attn "
          "prefill re-run because the state checkpoint is coarser):")
    print(f"  before: < {BASELINE_MAMBA_BLOCK} tokens   after: < "
          f"{PROPOSED_MAMBA_BLOCK} tokens")

    print("-" * 78)
    print("Transfer bytes (un-padded pages, all workers) for a 53,760-token hit:")
    for tag, d in (("now", base), ("new", new)):
        H = 53760
        sc, real, nw = d["sc"], d["real"], d["nw"]
        store = load = 0
        for i, c in enumerate(sc.kv_group_configs):
            if c.is_inert:
                continue
            k = H // c.tokens_per_chunk
            store += k * real[i] * nw
            kl = k if c.sliding_window_size_in_chunks is None \
                else min(k, c.sliding_window_size_in_chunks)
            load += kl * real[i] * nw
        print(f"  {tag}: store {store/1e9:6.3f} GB   RAM-hit load {load/1e9:6.3f} GB")


# --------------------------------------------------------------------------
# PART 3 -- the align state-slot seed (mamba_hybrid.py) must use the MAMBA
#           block size, or it indexes past the end of the block-table row.
# --------------------------------------------------------------------------
SEED_RE = re.compile(
    r"self\._mamba_state_idx_gpu\[req_index\]\.fill_\(\s*"
    r"\(new_req_data\.num_computed_tokens - 1\)\s*//\s*([A-Za-z_.]+)\s*\)",
    re.S,
)


def part3():
    print("=" * 78)
    print("PART 3 -- align state-slot seed vs the block-table row it indexes")
    pri = SEED_RE.search(open(f"{HERE}/mamba_hybrid.pristine.py").read())
    pat = SEED_RE.search(open(f"{HERE}/mamba_hybrid.py").read())
    check("pristine seed divisor", pri.group(1) if pri else None,
          "self.cache_config.block_size")
    check("patched seed divisor", pat.group(1) if pat else None, "mamba_block_size")

    # The kernel's own slot formula (mamba_utils.preprocess_mamba_align_fused_kernel):
    #   new_state_idx = cdiv(computed_after, MAMBA_BLOCK_SIZE) - 1
    # with MAMBA_BLOCK_SIZE = mamba_spec.block_size. The seed must agree.
    def kernel_slot(n, mamba_block):
        return -(-n // mamba_block) - 1

    def seed_pristine(n, attn_block, mamba_block):
        return (n - 1) // attn_block

    def seed_patched(n, attn_block, mamba_block):
        return (n - 1) // mamba_block

    max_len = 393216
    for mamba_block in (BASELINE_MAMBA_BLOCK, PROPOSED_MAMBA_BLOCK):
        row = -(-max_len // mamba_block) + K_SPEC_BLOCKS
        print(f"  mamba_block_size={mamba_block}  block-table row width={row}")
        bad_pri = bad_pat = 0
        for n in (3584, 14336, 28672, 57344, 100352, 200704, 393216):
            kp = kernel_slot(n, mamba_block)
            sp = seed_pristine(n, MLA_BLOCK, mamba_block)
            sn = seed_patched(n, MLA_BLOCK, mamba_block)
            oob_p = sp >= row
            bad_pri += (sp != kp)
            bad_pat += (sn != kp)
            print(f"    resumed@{n:>6}  kernel_slot={kp:>3}  "
                  f"pristine_seed={sp:>3}{' OOB!' if oob_p else '':<5}  "
                  f"patched_seed={sn:>3}")
        if mamba_block == BASELINE_MAMBA_BLOCK:
            check("baseline: pristine seed agrees with kernel", bad_pri, 0)
        else:
            check("14336: pristine seed DISAGREES with kernel (the bug)",
                  bad_pri > 0, True)
        check(f"mamba_block={mamba_block}: patched seed agrees with kernel",
              bad_pat, 0)


# --------------------------------------------------------------------------
# PART 4 -- the prefill splitter (v1/core/sched/scheduler.py) must stop on
#           MAMBA boundaries. Executes the real method from both files.
# --------------------------------------------------------------------------
SPLIT_START = re.compile(r"^    def _mamba_block_aligned_split\(")
SPLIT_END = re.compile(r"^    def _get_local_prefix_cache_hit\(")


def slice_split_fn(path: str):
    lines = open(path).read().splitlines()
    a = next(i for i, ln in enumerate(lines) if SPLIT_START.match(ln))
    b = next(i for i, ln in enumerate(lines) if SPLIT_END.match(ln))
    src = textwrap.dedent("\n".join(lines[a:b]))
    ns = {"max": max, "min": min, "Request": object}
    exec(compile(src, path, "exec"), ns, ns)
    return ns["_mamba_block_aligned_split"]


class _Req:
    def __init__(self, n_prompt):
        self.num_prompt_tokens = n_prompt
        self.num_tokens = n_prompt
        self.num_computed_tokens = 0
        self.shared_prefix_boundary = 0


# NOTE (verified live 2026-08-28): in the ENGINE-CORE process
# `cache_config.block_size` is rewritten to min(participating group block
# sizes) = 896 by vllm/v1/engine/core.py before the Scheduler is built, so the
# splitter's upstream value is 896, not the 3584 attention block. In the WORKER
# processes it stays 3584. Test both.
ENGINE_CORE_BLOCK_SIZE = 896


def _sched_stub(mamba_block, mnbt=1024, hash_block=896, use_eagle=True,
                cc_block_size=ENGINE_CORE_BLOCK_SIZE, user_specified=True):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=cc_block_size,
            mamba_block_size=mamba_block,
            user_specified_mamba_block_size=user_specified),
        use_eagle=use_eagle,
        max_num_scheduled_tokens=mnbt,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        hash_block_size=hash_block,
        mamba_partial_cache_hit=hash_block < mamba_block,
    )


def run_prefill(split_fn, sched, prompt_tokens, mnbt=1024):
    """Walk a full chunked prefill and return every chunk-end position."""
    req = _Req(prompt_tokens)
    ends = []
    guard = 0
    while req.num_computed_tokens < prompt_tokens:
        guard += 1
        assert guard < 100000, "prefill did not terminate"
        want = min(mnbt, prompt_tokens - req.num_computed_tokens)
        got = split_fn(sched, req, want)
        assert got > 0, f"split returned {got} at {req.num_computed_tokens}"
        req.num_computed_tokens += got
        ends.append(req.num_computed_tokens)
    return ends


def part4():
    print("=" * 78)
    print("PART 4 -- prefill splitter: does every mamba boundary get a chunk end?")
    pri = slice_split_fn(f"{HERE}/sched_scheduler.pristine.py")
    pat = slice_split_fn(f"{HERE}/sched_scheduler.py")
    P = 60000
    # (a) DEFAULT (no --mamba-block-size): patched must be byte-identical to
    #     upstream for BOTH possible cache_config.block_size values.
    for cc in (ENGINE_CORE_BLOCK_SIZE, MLA_BLOCK):
        for mnbt in (1024, 8192):
            e_pri = run_prefill(pri, _sched_stub(BASELINE_MAMBA_BLOCK, mnbt=mnbt,
                                                 cc_block_size=cc,
                                                 user_specified=False), P, mnbt)
            e_pat = run_prefill(pat, _sched_stub(BASELINE_MAMBA_BLOCK, mnbt=mnbt,
                                                 cc_block_size=cc,
                                                 user_specified=False), P, mnbt)
            check(f"DEFAULT (no flag) cc_block={cc} mnbt={mnbt}: patched == "
                  f"pristine ({len(e_pat)} chunks)", e_pat == e_pri, True)

    # (b) with --mamba-block-size: chunk ends must hit every mamba boundary.
    for mnbt in (1024, 8192):
        for mamba_block in (BASELINE_MAMBA_BLOCK, PROPOSED_MAMBA_BLOCK):
            res = {}
            for tag, fn in (("pristine", pri), ("patched", pat)):
                sched = _sched_stub(mamba_block, mnbt=mnbt)
                ends = run_prefill(fn, sched, P, mnbt=mnbt)
                bounds = list(range(mamba_block, P + 1, mamba_block))
                missed = [b for b in bounds if b not in ends]
                res[tag] = (ends, missed)
                print(f"  mnbt={mnbt:<5} mamba_block={mamba_block:<6} {tag:<9} "
                      f"chunks={len(ends):<4} missed mamba boundaries={missed}")
            if mamba_block == BASELINE_MAMBA_BLOCK:
                # explicit --mamba-block-size 3584: the overlay aligns to the
                # real mamba stride instead of the finer engine-core 896. A
                # chunk WIDER than one mamba block can still skip boundaries
                # (upstream hole, see note) -- require only "no worse".
                check(f"mnbt={mnbt} mb={mamba_block} patched no worse than "
                      f"pristine ({len(res['patched'][1])} vs "
                      f"{len(res['pristine'][1])} missed)",
                      len(res["patched"][1]) <= len(res["pristine"][1]), True)
                if res["pristine"][1]:
                    print(f"      note: UPSTREAM already misses these at "
                          f"mnbt={mnbt} -- pre-existing, unchanged by the "
                          f"overlay (chunks larger than one mamba block skip "
                          f"boundaries).")
            else:
                check(f"mnbt={mnbt} mb={mamba_block} patched: no missed boundary",
                      res["patched"][1], [])
                check(f"mnbt={mnbt} mb={mamba_block} patched no worse than "
                      f"pristine", len(res["patched"][1])
                      <= len(res["pristine"][1]), True)
    # the perf angle: chunk count at the live config
    n0 = len(run_prefill(pri, _sched_stub(BASELINE_MAMBA_BLOCK, mnbt=1024,
                                          user_specified=False), P, 1024))
    n1 = len(run_prefill(pat, _sched_stub(PROPOSED_MAMBA_BLOCK, mnbt=1024),
                         P, 1024))
    print(f"  60k prefill chunks @mnbt1024: live baseline (cc_block=896) {n0} "
          f"-> overlay @14336 {n1}  (ideal ceil(60000/1024)={-(-P // 1024)})")


def main():
    with_drafter = "--no-drafter" not in sys.argv
    part1()
    print()
    part3()
    print()
    part4()
    print()
    part2(with_drafter)
    print("=" * 78)
    if FAILURES:
        print(f"DRY RUN FAILED ({len(FAILURES)} checks):")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("DRY RUN OK")


if __name__ == "__main__":
    main()
