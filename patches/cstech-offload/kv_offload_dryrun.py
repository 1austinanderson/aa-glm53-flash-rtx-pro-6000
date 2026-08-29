"""CPU dry-run for the GLM-5.3-Flash + DFlash-2 native KV offload overlays.

Builds the REAL group layout via `_get_kv_cache_groups_glm5_next` (the same
specs the KVDIAG unify dump reports), then exercises the two overlays:

  vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py
      -> build_offloading_config: every KV group kept IN POSITION, unhashable
         ones marked inert (layer_names=()).
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
      -> resolve_mamba_align_size + SchedulerOffloadConfig.from_spec.

Run inside the serving image (CPU only, no GPU needed):
  docker run --rm <mounts> <image> python3 /work/kv_offload_dryrun.py
"""

import sys
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


def build_layer_specs(with_drafter=True):
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
        # Real KDA state, from MambaStateShapeCalculator.kda_state_shape with
        # linear_num_heads=64, linear_head_dim=128, conv_kernel=4, tp=2,
        # num_spec=num_speculative_tokens=3:
        #   conv      (3*64*128/2, 4-1+3) = (12288, 6) bf16   =   147,456 B
        #   recurrent (64/2, 128, 128)         float32        = 2,097,152 B
        #   real page                                         = 2,244,608 B
        # padded to the 2,351,104 B MLA slot page (4.5% slack).
        spec[f"lm.layers.m{i}.mamba"] = MambaSpec(
            block_size=MLA_BLOCK, shapes=((12288, 6), (32, 128, 128)),
            dtypes=(BF16, torch.float32), mamba_cache_mode="align",
            page_size_padded=MLA_PAGE, num_speculative_blocks=K_SPEC_BLOCKS)
    if with_drafter:
        for i in range(N_SW):
            spec[f"drafter.layers.{i}.self_attn.attn"] = SlidingWindowSpec(
                block_size=SW_BLOCK_RAW, num_kv_heads=4, head_size=128,
                dtype=BF16, sliding_window=2048, page_size_padded=MLA_PAGE)
    return spec


def group_worker_bytes(kv_cache_config) -> list[int]:
    """Real (un-padded) CPU bytes ONE worker must hold for one chunk of each
    KV cache group.

    Mirrors exactly what OffloadingConnectorWorker.register_kv_caches puts in
    CanonicalKVCacheRef.page_size_bytes (and therefore what the copy
    descriptors already move):
      * AttentionSpec  -> spec.unpadded_page_size_bytes
      * MambaSpec      -> replace(spec, page_size_padded=None).page_size_bytes
    Summed over the group's layers, because each layer of a group owns its own
    page in a different MLA slot tensor.
    """
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


def round_up(x, m):
    return ((x + m - 1) // m) * m


def make_vllm_config(offload_gib=16.0):
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
            mamba_cache_mode="align", cache_dtype="fp8"),
        kv_transfer_config=SimpleNamespace(
            engine_id="dryrun",
            kv_connector_extra_config={"cpu_bytes_to_use": int(offload_gib * 2**30)}),
        kv_events_config=None,
        model_config=SimpleNamespace(
            max_model_len=262144, model="glm53", dtype=BF16, use_mla=True,
            get_total_num_hidden_layers=lambda: 45,
            get_total_num_kv_heads=lambda: 1),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
            max_num_batched_tokens=1024, max_num_seqs=2),
        max_in_flight_tokens=2048,
        speculative_config=SimpleNamespace(
            num_speculative_tokens=K_SPEC_BLOCKS,
            use_eagle=lambda: True, method="dflash"),
        use_v2_model_runner=False,
    )


def scheduler_view(groups):
    """Mirror generate_scheduler_kv_cache_config: unwrap UniformTypeKVCacheSpecs."""
    import copy
    out = copy.deepcopy(groups)
    for g in out:
        if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs):
            g.kv_cache_spec = next(iter(g.kv_cache_spec.kv_cache_specs.values()))
    return out


def main(with_drafter=True):
    cfg = make_vllm_config()
    layer_specs = build_layer_specs(with_drafter)
    groups = U._get_kv_cache_groups_glm5_next(cfg, layer_specs)
    assert groups is not None, "glm5_next layout rejected"

    num_blocks = 130
    tensors = []
    lay = U._glm5_next_tensor_layout(groups)
    _, _, mla_names, idx_names, mla_page, idx_page, tail_names, _ = lay
    for n in mla_names:
        shared = [n] + [m for g in groups[1:] for m in g.layer_names
                        if mla_names.index(n) == g.layer_names.index(m)] \
            if False else [n]
        tensors.append(KVCacheTensor(size=num_blocks * mla_page, shared_by=[n]))
    for n in idx_names:
        tensors.append(KVCacheTensor(size=num_blocks * idx_page, shared_by=[n]))
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=tensors,
        kv_cache_groups=scheduler_view(groups))

    print("=" * 78)
    print(f"KV cache groups (scheduler view), drafter={with_drafter}")
    for i, g in enumerate(kv_cache_config.kv_cache_groups):
        s = g.kv_cache_spec
        print(f"  [{i}] {type(s).__name__:<20} block={s.block_size:<6} "
              f"layers={len(g.layer_names):<3} "
              f"prefix_caching={s.participates_in_prefix_caching}")
    sched_bs, tph = U.resolve_kv_cache_block_sizes(kv_cache_config, cfg)
    print(f"  scheduler_block_size={sched_bs}  tokens_per_hash={tph}")

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
        build_offloading_config,
    )
    ocfg = build_offloading_config(cfg, kv_cache_config)
    print("-" * 78)
    print("OffloadingConfig.groups (must be one per KV group, IN POSITION):")
    for i, g in enumerate(ocfg.groups):
        print(f"  [{i}] tokens_per_block={g.tokens_per_block:<6} "
              f"layers={len(g.layer_names):<3} "
              f"{'INERT' if not g.layer_names else ''}")
    assert len(ocfg.groups) == len(kv_cache_config.kv_cache_groups), "position lost!"
    print(f"  worker_kv_bytes_per_block={ocfg.worker_kv_bytes_per_block:,} "
          f"tokens_per_hash={ocfg.cache.tokens_per_hash}")

    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
    spec = CPUOffloadingSpec(ocfg)
    print(f"  CPUOffloadingSpec: num_blocks={spec.num_blocks} "
          f"kv_bytes_per_chunk={spec.kv_bytes_per_chunk:,} "
          f"({spec.kv_bytes_per_chunk / 2**20:.1f} MiB) "
          f"replicated_layout={spec.replicated_layout}")

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        SchedulerOffloadConfig, resolve_mamba_align_size,
    )
    align = resolve_mamba_align_size(spec, kv_cache_config)
    print("-" * 78)
    print(f"resolve_mamba_align_size -> {align}  (expect {MLA_BLOCK})")
    assert align == MLA_BLOCK, f"mamba align {align} != {MLA_BLOCK}"

    sc = SchedulerOffloadConfig.from_spec(spec, cfg, kv_cache_config)
    print("SchedulerOffloadConfig.kv_group_configs:")
    for c in sc.kv_group_configs:
        print(f"  [{c.group_idx}] tpb={c.tokens_per_block:<6} tpc={c.tokens_per_chunk:<6} "
              f"hpc={c.hashes_per_chunk:<3} sw_chunks={c.sliding_window_size_in_chunks} "
              f"align_chunks={c.alignment_chunk_count} eagle={c.is_eagle_group} "
              f"cow={c.requires_cow_source} inert={c.is_inert}")
    assert len(sc.kv_group_configs) == len(kv_cache_config.kv_cache_groups)
    for i, c in enumerate(sc.kv_group_configs):
        assert c.group_idx == i, "group_idx must equal the KV cache group index"
    print(f"  blocks_per_chunk={sc.blocks_per_chunk} tokens_per_hash={sc.tokens_per_hash} "
          f"num_workers={sc.num_workers} offload_prompt_only={sc.offload_prompt_only} "
          f"supports_partial_tail={sc.supports_partial_tail}")

    # ---------------------------------------------------------------- audit
    # Exact byte accounting: what a CPU chunk is CHARGED vs what it actually
    # HOLDS, per KV cache group.
    # The WORKER sees the un-unwrapped groups (UniformTypeKVCacheSpecs still
    # carries the per-layer MLA / indexer page sizes); the scheduler view above
    # collapses them to the first inner spec.
    worker_kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=groups)
    real = group_worker_bytes(worker_kv_cache_config)
    nw = sc.num_workers
    charged = spec.kv_bytes_per_chunk           # per chunk, ALL workers
    print("-" * 78)
    print("Per-group chunk occupancy (per worker | x%d workers | charged):" % nw)
    for i, c in enumerate(sc.kv_group_configs):
        if c.is_inert:
            print(f"  [{i}] INERT")
            continue
        print(f"  [{i}] tpc={c.tokens_per_chunk:<5} real={real[i]:>11,} "
              f"| {real[i]*nw:>11,} | {charged:>11,}  "
              f"fill={real[i]*nw/charged:6.1%}")

    def _chunks(P, c, cap=None):
        n = P // c.tokens_per_chunk
        return n if cap is None else min(n, cap)

    print("-" * 78)
    print("RAM per prompt (all workers):")
    for P in (30000, 60000):
        n_now = n_fix = 0
        b_now = b_fix = 0
        for i, c in enumerate(sc.kv_group_configs):
            if c.is_inert:
                continue
            n = _chunks(P, c)
            n_now += n
            b_now += n * charged
            # right-sized: each group's chunk costs only what it holds,
            # rounded up to the mmap page alignment
            b_fix += n * round_up(real[i] * nw, 4096)
            n_fix += n
        print(f"  {P:>6} tokens -> {n_now:>4} CPU chunks | "
              f"now {b_now/2**30:5.2f} GiB ({spec.num_blocks/max(n_now,1):.2f} prompts) | "
              f"right-sized {b_fix/2**30:5.2f} GiB "
              f"({(int(spec.kv_bytes_per_chunk)*spec.num_blocks)/max(b_fix,1):.2f} prompts) | "
              f"{b_now/max(b_fix,1):.2f}x")

    print("-" * 78)
    print("Transfer bytes (all workers) -- pages are ALREADY un-padded:")
    for H in (53760,):
        store = sum(
            _chunks(H, c) * real[i] * nw
            for i, c in enumerate(sc.kv_group_configs) if not c.is_inert)
        load = 0
        for i, c in enumerate(sc.kv_group_configs):
            if c.is_inert:
                continue
            load += _chunks(H, c, c.sliding_window_size_in_chunks) * real[i] * nw
        print(f"  {H:>6}-token prefix: store {store/1e9:5.3f} GB, "
              f"RAM-hit load {load/1e9:5.3f} GB")
        for i, c in enumerate(sc.kv_group_configs):
            if c.is_inert:
                continue
            l = _chunks(H, c, c.sliding_window_size_in_chunks) * real[i] * nw
            print(f"      [{i}] load {l/1e6:8.1f} MB "
                  f"({_chunks(H, c, c.sliding_window_size_in_chunks)} chunks"
                  f"{'' if c.sliding_window_size_in_chunks is None else ' capped by window'})")
    print("=" * 78)
    print("DRY RUN OK")


if __name__ == "__main__":
    main(with_drafter="--no-drafter" not in sys.argv)
