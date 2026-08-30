"""CPU dry-run for the RAM-tier PARTIAL-TAIL hand-off overlays.

Overlays under test (mount all four together -- they are one change):

  vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
      <- patches/cstech-offload/scheduler.partial-tail.py
         (inert groups skipped in the boundary scan / store / touch;
          per-group boundary block index; eagle drop narrowed to real
          draft-model groups; eagle back-off of the hand-off boundary)
  vllm/v1/core/single_type_kv_cache_manager.py
      <- patches/cstech-partial-tail/single_type_kv_cache_manager.py
         (MambaManager registers its partial tail one hash unit earlier
          under eagle)
  vllm/v1/core/sched/scheduler.py
      <- patches/cstech-mamba-interval/sched_scheduler.partial-tail.py
         (prefill split stops at the same backed-off boundary)
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py
      <- patches/cstech-offload/config.sw-inert.py  (UNCHANGED, already live)

Everything runs on the CPU: the REAL GLM-5.3-Flash group layout is built with
`_get_kv_cache_groups_glm5_next` (11 MLA @3,584 + 11 kpool tail @4 + 34 KDA
@14,336 + 5 DFlash-2 sliding-window @896), the real `build_offloading_config` /
`SchedulerOffloadConfig.from_spec` run on it, and an `OffloadingConnectorScheduler`
is driven with a stub OffloadingManager whose lookup/prepare_store/prepare_load
we control.

  docker run --rm --entrypoint python3 <mounts> <image> /work/partial_tail_dryrun.py

PART 7 additionally diffs the prefill splitter against its base; for that, mount
patches/cstech-mamba-interval at /base (it is skipped if /base is absent).
"""

import os
import re
import sys
import textwrap
from types import SimpleNamespace

import torch

import vllm.v1.core.kv_cache_utils as U
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheTensor,
    KpoolTailSpec,
    MLAAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    OffloadingSpec,
    OffloadKey,
    PrepareStoreOutput,
    RequestOffloadingContext,
    make_offload_key,
)

FP8, BF16 = torch.float8_e4m3fn, torch.bfloat16
MLA_BLOCK, MLA_PAGE = 3584, 2351104
SW_BLOCK_RAW = 1136
N_MLA, N_KDA, N_SW = 11, 34, 5
K_SPEC_BLOCKS = 3
MAMBA_BLOCK = 14336          # --mamba-block-size 14336
HASH = 896                   # tokens_per_hash on this layout
PROMPT = 47699               # the prompt measured on the GPU tonight
SHORT_PROMPT = 5000

FAILURES: list[str] = []


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got!r}"
          + ("" if ok else f"   (want {want!r})"))
    if not ok:
        FAILURES.append(f"{name}: got {got!r} want {want!r}")
    return ok


# --------------------------------------------------------------------------
# Real GLM-5.3-Flash layout
# --------------------------------------------------------------------------
def build_layer_specs(mamba_block: int):
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
        spec[f"lm.layers.m{i}.mamba"] = MambaSpec(
            block_size=mamba_block, shapes=((12288, 6), (32, 128, 128)),
            dtypes=(BF16, torch.float32), mamba_cache_mode="align",
            page_size_padded=MLA_PAGE, num_speculative_blocks=K_SPEC_BLOCKS)
    for i in range(N_SW):
        spec[f"drafter.layers.{i}.self_attn.attn"] = SlidingWindowSpec(
            block_size=SW_BLOCK_RAW, num_kv_heads=4, head_size=128,
            dtype=BF16, sliding_window=2048, page_size_padded=MLA_PAGE)
    return spec


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


def build_kv_cache_config(mamba_block: int):
    cfg = make_vllm_config(mamba_block)
    groups = U._get_kv_cache_groups_glm5_next(cfg, build_layer_specs(mamba_block))
    assert groups is not None, "glm5_next layout rejected"
    num_blocks = 130
    _, _, mla_names, idx_names, mla_page, idx_page, _, _ = U._glm5_next_tensor_layout(
        groups
    )
    tensors = [KVCacheTensor(size=num_blocks * mla_page, shared_by=[n])
               for n in mla_names]
    tensors += [KVCacheTensor(size=num_blocks * idx_page, shared_by=[n])
                for n in idx_names]
    return cfg, KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=tensors,
                              kv_cache_groups=scheduler_view(groups))


# --------------------------------------------------------------------------
# Stub offloading spec / manager / request
# --------------------------------------------------------------------------
class _StubStoreSpec(LoadStoreSpec):
    def __init__(self, keys):
        self.keys = list(keys)


class StubManager:
    """OffloadingManager stand-in with a plain set as the RAM pool."""

    def __init__(self):
        self.store: set[OffloadKey] = set()
        self.lookups: list[OffloadKey] = []
        self.touched: list[OffloadKey] = []
        self.loaded: list[OffloadKey] = []

    def on_new_request(self, req_context):
        return RequestOffloadingContext()

    def on_request_finished(self, req_context):
        pass

    def on_schedule_end(self, ctx):
        pass

    def lookup(self, key, req_context):
        self.lookups.append(key)
        return LookupResult.HIT if key in self.store else LookupResult.MISS

    def touch(self, keys, req_context):
        self.touched.extend(keys)

    def prepare_load(self, keys, req_context):
        self.loaded = list(keys)
        return _StubStoreSpec(keys)

    def prepare_store(self, keys, req_context):
        # Accept everything, in the order given (a real manager filters keys
        # already resident; the order-preservation is asserted by the overlay).
        return PrepareStoreOutput(keys_to_store=list(keys),
                                  store_spec=_StubStoreSpec(keys),
                                  evicted_keys=[])

    def complete_store(self, keys, req_context):
        self.store.update(keys)

    def complete_load(self, keys, req_context):
        pass

    def take_events(self):
        return ()

    def get_stats(self):
        return None

    def has_pending_work(self):
        return False

    def reset_cache(self):
        self.store.clear()

    def shutdown(self):
        pass


class StubSpec(OffloadingSpec):
    def __init__(self, config, manager):
        super().__init__(config)
        self._manager = manager

    def get_manager(self):
        return self._manager

    def get_worker(self, kv_caches):
        raise NotImplementedError


class StubRequest:
    def __init__(self, req_id: str, num_prompt_tokens: int):
        self.request_id = req_id
        self.num_prompt_tokens = num_prompt_tokens
        self.num_tokens = num_prompt_tokens
        self.num_computed_tokens = 0
        self.kv_transfer_params = None
        self.skip_reading_prefix_cache = False
        self.status = None
        # One block hash per complete `HASH`-token unit of the prompt, exactly
        # what `Request.block_hashes` holds once the prompt is hashed.
        self.block_hashes = [
            f"blockhash-{i:08d}".encode().ljust(32, b".")
            for i in range(num_prompt_tokens // HASH)
        ]

    def is_finished(self):
        return False


class _Blk:
    __slots__ = ("block_id", "is_null", "block_hash")

    def __init__(self, block_id, is_null=False, block_hash=None):
        self.block_id = block_id
        self.is_null = is_null
        self.block_hash = block_hash


# --------------------------------------------------------------------------
def build_scheduler(kv_cache_config, cfg, manager):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
        build_offloading_config,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    ocfg = build_offloading_config(cfg, kv_cache_config)
    spec = StubSpec(ocfg, manager)
    return OffloadingConnectorScheduler(spec, cfg, kv_cache_config), spec


def group_key(sched, req, group_idx: int, boundary: int) -> OffloadKey:
    return make_offload_key(req.block_hashes[boundary // HASH - 1], group_idx)


def populate_regular_chunks(sched, manager, req, skip=()):
    """Store every regular chunk key a producer would have written for `req`."""
    for gc in sched.config.kv_group_configs:
        if gc.is_inert:
            continue
        n = req.num_prompt_tokens // gc.tokens_per_chunk
        for chunk in range(n):
            end = (chunk + 1) * gc.tokens_per_chunk
            if (gc.group_idx, chunk) in skip:
                continue
            manager.store.add(group_key(sched, req, gc.group_idx, end))


def populate_boundary(sched, manager, req, boundary: int):
    for gc in sched.config.kv_group_configs:
        if gc.is_inert:
            continue
        manager.store.add(group_key(sched, req, gc.group_idx, boundary))


def eagle_boundary(num_prompt_tokens: int) -> int:
    """Where the producer registers its partial tail under EAGLE."""
    return num_prompt_tokens // HASH * HASH - HASH


# --------------------------------------------------------------------------
def part0(sched, kv_cache_config):
    print("=" * 78)
    print("PART 0 -- layout and SchedulerOffloadConfig")
    for i, (g, c) in enumerate(zip(kv_cache_config.kv_cache_groups,
                                   sched.config.kv_group_configs)):
        print(f"  [{i}] {type(g.kv_cache_spec).__name__:<18} "
              f"kv_block={g.kv_cache_spec.block_size:<6} "
              f"offload_tpb={c.tokens_per_block:<6} sw={c.sliding_window_size_in_chunks} "
              f"cow={c.requires_cow_source} eagle_store={c.is_eagle_group} "
              f"draft={c.is_draft_group} inert={c.is_inert}")
    inert = [c.group_idx for c in sched.config.kv_group_configs if c.is_inert]
    check("inert groups (kpool tail + DFlash-2 sliding window)", inert, [1, 6])
    check("tokens_per_hash", sched.config.tokens_per_hash, HASH)
    check("blocks_per_chunk", sched.config.blocks_per_chunk, 1)
    check("mamba align size", sched._mamba_align_size, MAMBA_BLOCK)
    check("(1) supports_partial_tail with the live layout",
          sched.config.supports_partial_tail, True)
    check("partial-tail search window = KDA block (not MLA)",
          sched._partial_tail_block_size, MAMBA_BLOCK)
    check("eagle tail back-off armed", sched._eagle_tail_backoff, True)
    check("store-side eagle guard still covers every active group",
          all(c.is_eagle_group for c in sched.config.kv_group_configs
              if not c.is_inert), True)
    check("load-side drop applies to no active group (none annotated)",
          [c.group_idx for c in sched.config.kv_group_configs if c.is_draft_group],
          [])
    check("cow-source groups = the 4 KDA groups",
          sorted(sched._cow_source_groups), [2, 3, 4, 5])


def fresh(kv_cache_config, cfg):
    manager = StubManager()
    sched, _ = build_scheduler(kv_cache_config, cfg, manager)
    return sched, manager


def run_lookup(sched, manager, req):
    sched.on_new_request(req)
    hit, _async = sched.get_num_new_matched_tokens(req, 0)
    return hit


def part1(kv_cache_config, cfg):
    print("=" * 78)
    print(f"PART 1 -- {PROMPT}-token prompt, everything in RAM")
    boundary = eagle_boundary(PROMPT)
    check("eagle-adjusted hand-off boundary", boundary, 46592)
    check("  = last prompt hash boundary (47,488) - one hash unit",
          PROMPT // HASH * HASH - HASH, 46592)
    check("  boundary is NOT a whole KDA block", boundary % MAMBA_BLOCK, 3584)
    check("  boundary IS a whole MLA block (key aliases chunk 12)",
          boundary % MLA_BLOCK, 0)

    sched, manager = fresh(kv_cache_config, cfg)
    req = StubRequest("r1", PROMPT)
    populate_regular_chunks(sched, manager, req)
    populate_boundary(sched, manager, req, boundary)
    hit = run_lookup(sched, manager, req)
    check("(2) _lookup returns the eagle-adjusted boundary", hit, 46592)
    check("     partial_tail_boundary recorded",
          sched._req_status["r1"].partial_tail_boundary, 46592)
    return sched, manager, req


def part2(sched, manager, req):
    print("=" * 78)
    print("PART 2 -- load job (update_state_after_alloc)")
    boundary = eagle_boundary(PROMPT)
    ncfg = len(sched.config.kv_group_configs)
    blocks = []
    expect_block_indices = []
    for c in sched.config.kv_group_configs:
        if c.is_inert:
            blocks.append([_Blk(9000 + c.group_idx)])
            expect_block_indices.append(0)
            continue
        n = cdiv(boundary, c.tokens_per_block)
        if c.requires_cow_source:
            # mamba "align": leading state slots are null placeholders, the
            # running-state block is the last one.
            row = [_Blk(0, is_null=True) for _ in range(n - 1)]
            row.append(_Blk(100 * c.group_idx + n))
            expect_block_indices.append(n - 1)
        else:
            row = [_Blk(100 * c.group_idx + i) for i in range(n)]
            expect_block_indices.append(0)
        blocks.append(row)
    kv_blocks = SimpleNamespace(blocks=tuple(blocks))

    sched.update_state_after_alloc(req, kv_blocks, boundary)
    job = next(iter(sched._current_batch_load_jobs.values()))
    dst = job.dst_spec
    print(f"  group_sizes   = {list(dst.group_sizes)}")
    print(f"  block_indices = {list(dst.block_indices)}")
    check("(2) load group_sizes", list(dst.group_sizes), [13, 0, 1, 1, 1, 1, 0])
    check("(2) load block_indices", list(dst.block_indices),
          [0, 0, 3, 3, 3, 3, 0])
    check("     KDA boundary state lands in block boundary // 14336",
          list(dst.block_indices)[2], boundary // MAMBA_BLOCK)
    check("     agrees with mamba_hybrid seed (n-1)//mamba_block",
          (boundary - 1) // MAMBA_BLOCK, boundary // MAMBA_BLOCK)

    loaded = manager.loaded
    check("     total blocks loaded == sum(group_sizes)",
          len(loaded), sum(dst.group_sizes))
    check("     no key loaded twice", len(set(loaded)), len(loaded))
    mla_keys = [group_key(sched, req, 0, (i + 1) * MLA_BLOCK) for i in range(13)]
    check("(2) MLA: chunks 0..12 (chunk 12's key IS the boundary key)",
          loaded[:13], mla_keys)
    check("     MLA boundary key == chunk 12 key",
          group_key(sched, req, 0, boundary), mla_keys[12])
    for slot, gidx in enumerate((2, 3, 4, 5)):
        check(f"(2) KDA group {gidx}: only the boundary key",
              loaded[13 + slot], group_key(sched, req, gidx, boundary))
    check("     inert groups contribute zero-sized entries",
          [dst.group_sizes[1], dst.group_sizes[6]], [0, 0])
    check("     len(group_sizes) == number of KV cache groups",
          len(dst.group_sizes), ncfg)


def part3(kv_cache_config, cfg):
    print("=" * 78)
    print("PART 3 -- fallback when the boundary keys are absent")
    sched, manager = fresh(kv_cache_config, cfg)
    req = StubRequest("r3a", PROMPT)
    populate_regular_chunks(sched, manager, req)      # no boundary keys
    hit = run_lookup(sched, manager, req)
    check("(3) no boundary keys -> mamba-aligned complete hit", hit, 43008)
    check("     = 3 x 14,336", 3 * MAMBA_BLOCK, 43008)
    check("     partial_tail_boundary cleared",
          sched._req_status["r3a"].partial_tail_boundary, None)

    sched, manager = fresh(kv_cache_config, cfg)
    req = StubRequest("r3b", PROMPT)
    # KDA chunk index 2 (28,672 -> 43,008) missing from every KDA group.
    populate_regular_chunks(sched, manager, req,
                            skip={(g, 2) for g in (2, 3, 4, 5)})
    hit = run_lookup(sched, manager, req)
    check("(3) KDA chunk 2 missing -> complete hit drops one KDA chunk",
          hit, 28672)

    # Reproduce the PRE-PATCH behaviour: eagle drop on every group, partial
    # tail disabled. This is the 28,672 measured on the GPU tonight for a RAM
    # hit with every chunk present -- proof the loss came from the lookup-side
    # eagle drop, not from a missing store.
    sched, manager = fresh(kv_cache_config, cfg)
    req = StubRequest("r3c", PROMPT)
    populate_regular_chunks(sched, manager, req)
    populate_boundary(sched, manager, req, eagle_boundary(PROMPT))
    sched.config = sched.config._replace(
        supports_partial_tail=False,
        kv_group_configs=tuple(c._replace(is_draft_group=True)
                               for c in sched.config.kv_group_configs))
    hit = run_lookup(sched, manager, req)
    check("(B) pre-patch simulation (eagle drop on ALL groups) reproduces "
          "tonight's measured RAM hit", hit, 28672)


def part4(kv_cache_config, cfg):
    print("=" * 78)
    print(f"PART 4 -- {SHORT_PROMPT}-token prompt")
    last_hash = SHORT_PROMPT // HASH * HASH
    boundary = eagle_boundary(SHORT_PROMPT)
    check("last prompt hash boundary", last_hash, 4480)
    check("eagle back-off -> hand-off boundary", boundary, 3584)
    check("  complete-chunk hit is 0 (round_down(4,999, 14,336))",
          round_down(SHORT_PROMPT - 1, MAMBA_BLOCK), 0)

    sched, manager = fresh(kv_cache_config, cfg)
    req = StubRequest("r4", SHORT_PROMPT)
    populate_regular_chunks(sched, manager, req)
    populate_boundary(sched, manager, req, boundary)
    hit = run_lookup(sched, manager, req)
    check("(4) 5,000-token prompt: RAM hit", hit, 3584)
    check("     3,584 = 1 MLA chunk, and the KDA state at 3,584 "
          "(partial tail, not a whole 14,336 block)",
          boundary % MAMBA_BLOCK, 3584)

    sched, manager = fresh(kv_cache_config, cfg)
    req = StubRequest("r4b", SHORT_PROMPT)
    populate_regular_chunks(sched, manager, req)
    hit = run_lookup(sched, manager, req)
    check("(4) 5,000-token prompt without the boundary keys: no hit", hit, 0)


def part5(kv_cache_config, cfg):
    print("=" * 78)
    print("PART 5 -- partial-tail STORE job")
    sched, manager = fresh(kv_cache_config, cfg)
    req = StubRequest("r5", PROMPT)
    sched.on_new_request(req)
    st = sched._req_status["r5"]
    st.update_offload_keys()
    boundary = eagle_boundary(PROMPT)
    # Block tables as the connector tracks them: MLA is a dense real row, the
    # KDA rows are handed off through the CoW block instead.
    for c in sched.config.kv_group_configs:
        if c.is_inert or c.requires_cow_source:
            continue
        st.group_states[c.group_idx].block_ids = [
            100 * c.group_idx + i for i in range(cdiv(PROMPT, c.tokens_per_block))
        ]
    handoff = {"r5": [(g, 7000 + g, boundary) for g in (2, 3, 4, 5)]}
    scheduler_output = SimpleNamespace(partial_tail_offloads=handoff)
    jobs = sched._build_partial_tail_store_jobs(scheduler_output)
    check("(5) one store job emitted", len(jobs), 1)
    job = next(iter(jobs.values()))
    src = job.src_spec
    print(f"  group_sizes   = {list(src.group_sizes)}")
    print(f"  block_indices = {list(src.block_indices)}")
    print(f"  source blocks = {list(src.block_ids)}")
    check("(5) one block per ACTIVE group, inert groups zero-sized",
          list(src.group_sizes), [1, 0, 1, 1, 1, 1, 0])
    check("(5) per-group boundary block index "
          "(MLA cdiv(46592,3584)-1=12, KDA cdiv(46592,14336)-1=3)",
          list(src.block_indices), [12, 0, 3, 3, 3, 3, 0])
    check("(5) MLA source is its own block table entry; KDA sources are the "
          "CoW blocks", list(src.block_ids), [12, 7002, 7003, 7004, 7005])
    stored = set(job.dst_spec.keys)
    check("(5) MLA boundary key aliases its regular chunk 12 key",
          group_key(sched, req, 0, boundary) in stored, True)
    check("(5) keys are one per active group", len(job.dst_spec.keys), 5)


def part6(kv_cache_config, cfg):
    print("=" * 78)
    print("PART 6 -- eagle trailing-chunk rule on the STORE path")
    sched, manager = fresh(kv_cache_config, cfg)
    req = StubRequest("r6", PROMPT)
    sched.on_new_request(req)
    st = sched._req_status["r6"]
    st.update_offload_keys()
    kda = sched.config.kv_group_configs[2]
    kda_state = st.group_states[2]
    kda_state.block_ids = [200 + i for i in range(4)]
    mla = sched.config.kv_group_configs[0]
    mla_state = st.group_states[0]
    mla_state.block_ids = [300 + i for i in range(14)]

    # Last prefill step: num_offloadable == num_prompt_tokens, NOT decoding.
    check("(6) KDA chunks storable at the 43,008 prefill step",
          st.storable_chunks(kda, kda_state, 43008), 3)
    check("(6) KDA chunks storable at end of prefill "
          f"({PROMPT} tokens, still prefill)",
          st.storable_chunks(kda, kda_state, PROMPT), 3)
    check("(6) MLA chunks storable at end of prefill",
          st.storable_chunks(mla, mla_state, PROMPT), 13)
    # First decode step: the optimistic 1 + K draft tokens push
    # num_offloadable past the prompt, so the trailing chunk is withheld.
    check("(6) KDA chunks storable once decoding (trailing chunk withheld)",
          st.storable_chunks(kda, kda_state, PROMPT + 4), 2)
    st.advance_stored_idx(PROMPT)
    check("(6) next_stored_chunk_idx after the last prefill step",
          kda_state.next_stored_chunk_idx, 3)
    st.advance_stored_idx(PROMPT + 4)
    check("(6) decode exclusion cannot walk the index back over the "
          "prefill-completed KDA chunk", kda_state.next_stored_chunk_idx, 3)


# --------------------------------------------------------------------------
# PART 7 -- the prefill splitter, sliced out of the shipped source text
# --------------------------------------------------------------------------
SPLIT_START = re.compile(r"^    def _mamba_block_aligned_split\(")
SPLIT_END = re.compile(r"^    def _get_local_prefix_cache_hit\(")
LIVE_SPLIT = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
BASE_SPLIT = "/base/sched_scheduler.py"


def slice_split_fn(path: str):
    lines = open(path).read().splitlines()
    a = next(i for i, ln in enumerate(lines) if SPLIT_START.match(ln))
    b = next(i for i, ln in enumerate(lines) if SPLIT_END.match(ln))
    ns = {"max": max, "min": min, "Request": object}
    exec(compile(textwrap.dedent("\n".join(lines[a:b])), path, "exec"), ns, ns)
    return ns["_mamba_block_aligned_split"]


class _SplitReq:
    def __init__(self, n_prompt):
        self.num_prompt_tokens = n_prompt
        self.num_tokens = n_prompt
        self.num_computed_tokens = 0
        self.shared_prefix_boundary = 0


def _split_stub(mamba_block, mnbt, cc_block_size, use_eagle, user_specified=True):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=cc_block_size,
            mamba_block_size=mamba_block,
            user_specified_mamba_block_size=user_specified),
        use_eagle=use_eagle,
        max_num_scheduled_tokens=mnbt,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        hash_block_size=HASH,
        mamba_partial_cache_hit=HASH < mamba_block,
    )


def run_prefill(split_fn, sched, prompt_tokens, mnbt):
    req = _SplitReq(prompt_tokens)
    ends = []
    while req.num_computed_tokens < prompt_tokens:
        want = min(mnbt, prompt_tokens - req.num_computed_tokens)
        got = split_fn(sched, req, want)
        assert got > 0, f"split returned {got} at {req.num_computed_tokens}"
        req.num_computed_tokens += got
        ends.append(req.num_computed_tokens)
    return ends


def part7():
    print("=" * 78)
    print("PART 7 -- prefill splitter (sched_scheduler.partial-tail.py)")
    if not os.path.exists(BASE_SPLIT):
        print("  SKIP  /base/sched_scheduler.py not mounted")
        return
    base = slice_split_fn(BASE_SPLIT)
    new = slice_split_fn(LIVE_SPLIT)

    # (a) No eagle -> byte-identical to the base file everywhere.
    same = True
    for mamba_block in (MLA_BLOCK, MAMBA_BLOCK):
        for cc in (HASH, MLA_BLOCK):
            for mnbt in (1024, 8192):
                for P in (PROMPT, SHORT_PROMPT, 60000):
                    a = run_prefill(base, _split_stub(mamba_block, mnbt, cc, False),
                                    P, mnbt)
                    b = run_prefill(new, _split_stub(mamba_block, mnbt, cc, False),
                                    P, mnbt)
                    same &= a == b
    check("(7) use_eagle=False: identical to the base splitter (24 configs)",
          same, True)

    # (b) With eagle, the partial-tail stop moves back by exactly one hash unit
    #     -- to the position MambaManager._cache_partial_tail_block now
    #     registers -- and disappears entirely when that backed-off position is
    #     a whole KDA state block (nothing partial left to register; the same
    #     `boundary % block_size == 0` guard the manager applies).
    b_off = eagle_boundary(PROMPT)
    last_hash = PROMPT // HASH * HASH
    for mamba_block in (MLA_BLOCK, MAMBA_BLOCK):
        registers = b_off % mamba_block != 0
        for mnbt in (1024, 8192):
            a = run_prefill(base, _split_stub(mamba_block, mnbt, MLA_BLOCK, True),
                            PROMPT, mnbt)
            b = run_prefill(new, _split_stub(mamba_block, mnbt, MLA_BLOCK, True),
                            PROMPT, mnbt)
            tag = f"mamba_block={mamba_block} mnbt={mnbt}"
            check(f"(7) eagle, {tag}: chunk ends below the hand-off boundary "
                  f"unchanged", [e for e in a if e < b_off],
                  [e for e in b if e < b_off])
            check(f"(7) eagle, {tag}: base stops at the un-backed-off boundary "
                  f"{last_hash}", last_hash in a, True)
            check(f"(7) eagle, {tag}: overlay no longer stops there",
                  last_hash in b, False)
            if registers:
                # The tail stop is mandatory, so a chunk always ends there.
                check(f"(7) eagle, {tag}: a chunk ends exactly on the hand-off "
                      f"boundary {b_off}", b_off in b, True)
            else:
                # The backed-off position is already a whole state block: the
                # manager registers no partial tail (same `% block_size == 0`
                # guard) and none is needed -- the regular full-block path
                # caches that state. (Whether the splitter happens to stop
                # there is the pre-existing upstream behaviour for chunks wider
                # than one state block; not this overlay's business.)
                check(f"(7) eagle, {tag}: no hand-off needed -- {b_off} is a "
                      f"whole KDA state block", b_off % mamba_block, 0)

    # (c) The moved stop must not cost a mamba state checkpoint.
    for mnbt in (1024, 8192):
        ends = run_prefill(new, _split_stub(MAMBA_BLOCK, mnbt, MLA_BLOCK, True),
                           60000, mnbt)
        missed = [b for b in range(MAMBA_BLOCK, 60001, MAMBA_BLOCK)
                  if b not in ends]
        check(f"(7) mnbt={mnbt}: no KDA state boundary missed at 60k", missed, [])

    # (d) The backed-off stop is exactly where the KDA partial tail is now
    #     registered, for both prompts.
    for P in (PROMPT, SHORT_PROMPT):
        ends = run_prefill(new, _split_stub(MAMBA_BLOCK, 1024, MLA_BLOCK, True),
                           P, 1024)
        check(f"(7) {P}-token prompt: a chunk ends exactly on the hand-off "
              f"boundary", eagle_boundary(P) in ends, True)


def main():
    cfg, kv_cache_config = build_kv_cache_config(MAMBA_BLOCK)
    sched, manager = fresh(kv_cache_config, cfg)
    part0(sched, kv_cache_config)
    s1, m1, r1 = part1(kv_cache_config, cfg)
    part2(s1, m1, r1)
    part3(kv_cache_config, cfg)
    part4(kv_cache_config, cfg)
    part5(kv_cache_config, cfg)
    part6(kv_cache_config, cfg)
    part7()
    print("=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("PARTIAL-TAIL DRY RUN OK -- all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
