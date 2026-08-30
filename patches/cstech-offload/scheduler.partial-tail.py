# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import chain, islice
from typing import Any, NamedTuple

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventGroupSpec,
    OffloadingEventsTracker,
    get_offloading_event_group_spec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    _ConnectorMetricName,
    _TransferMetricName,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    Locality,
    LookupResult,
    Medium,
    OffloadingManager,
    OffloadingSpec,
    OffloadKey,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    TierFilter,
    TierMatcher,
    make_offload_key,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)

# PATCH (local, 2026-08-30): EAGLE/DFlash hit back-off in hash units. Upstream
# drops ONE unit (896 tokens here) so the drafter gets hidden states for the
# tail; DFlash-2's drafter attends over a 2,048-token sliding window whose KV
# is NOT restored on a hit (draft group inert for offload / superseded local
# tail), so <2,048 recomputed tokens leave it drafting from stale KV -- measured
# 0/367 drafts accepted after a hit. 3 units (2,688-3,583 recomputed) keeps it
# alive. Default 1 = upstream byte-for-byte.
import os as _os
_EAGLE_DROP_UNITS = max(1, int(_os.environ.get("VLLM_EAGLE_DROP_UNITS", "1")))

KV_LOAD_TIERS_KEY = "kv_load_tiers"
MATCHER_MEDIUM_KEY = "medium"
MATCHER_LOCALITY_KEY = "locality"


@dataclass(slots=True)
class TransferJobStatus:
    """Tracks scheduler-side state for a single transfer job."""

    req_id: ReqId
    # Number of workers still pending. Starts at num_workers,
    # decremented as each worker reports completion. Job is done at 0.
    pending_count: int
    # Offload keys this job covers; passed to manager.complete_*().
    keys: set[OffloadKey]
    is_store: bool
    # Store source blocks fenced after the request finishes.
    deferred_fence_block_ids: list[int] | None = None
    # Store source blocks fenced when the transfer is created.
    fenced_block_ids: list[int] | None = None


class GroupOffloadConfig(NamedTuple):
    group_idx: int
    tokens_per_block: int
    tokens_per_chunk: int
    hashes_per_chunk: int
    # KV cache spec metadata propagated onto emitted BlockStored events so
    # KV-aware consumers can classify and filter the group.
    kv_event_group_spec: OffloadingEventGroupSpec
    # None below means full attention
    sliding_window_size_in_chunks: int | None
    # Partial-tail data for this group comes from the scheduler's CoW hand-off
    # rather than the request block table.
    requires_cow_source: bool = False
    # Number of this group's offloaded chunks per full-attention alignment
    # segment. Used to skip storing SWA chunks that can never serve a load
    # hit (e.g. DeepSeek V4 where SWA groups have much smaller block sizes
    # than the MLA full-attention group).
    # None for full-attention groups or when the optimization doesn't apply.
    alignment_chunk_count: int | None = None
    # True for EAGLE/MTP draft-model attention groups. The trailing chunk
    # of these groups is volatile and lacks a stable hash, so it must
    # be excluded from store and load scheduling.
    is_eagle_group: bool = False
    # PATCH (local, 2026-08-28): True for a group that exists in this tuple
    # only to keep offload-group index == KV-cache-group index. It is never
    # hashed, stored, loaded or looked up; it still contributes a zero-sized
    # entry to GPULoadStoreSpec.group_sizes so the worker's per-group arrays
    # (built from kv_cache_config.kv_cache_groups) stay aligned. Marked by
    # build_offloading_config as an OffloadingGroupConfig with layer_names=().
    is_inert: bool = False
    # PATCH (local, 2026-08-30): True only for a group that really holds a
    # DRAFT model's KV, i.e. one the platform annotated with
    # `KVCacheGroupSpec.is_eagle_group`. `is_eagle_group` above keeps its
    # conservative meaning ("spec decoding is running, the trailing chunk of
    # the offloadable range may still be rewritten") and still guards the
    # STORE path; this flag guards the LOOKUP path. See the rationale block in
    # SchedulerOffloadConfig.from_spec.
    is_draft_group: bool = False


def get_sliding_window_size_in_chunks(
    kv_cache_spec: KVCacheSpec, tokens_per_chunk: int
) -> int | None:
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        assert kv_cache_spec.sliding_window > 0
        return cdiv(kv_cache_spec.sliding_window, tokens_per_chunk)

    if isinstance(kv_cache_spec, ChunkedLocalAttentionSpec):
        # Attention never reaches back past one chunk
        assert kv_cache_spec.attention_chunk_size > 0
        return cdiv(kv_cache_spec.attention_chunk_size, tokens_per_chunk)

    if isinstance(kv_cache_spec, MambaSpec):
        # Mamba depends on a single state
        return 1

    assert isinstance(kv_cache_spec, FullAttentionSpec)
    return None


def is_store_reachable_swa_chunk(
    absolute_chunk_index: int,
    storable_chunk_count: int,
    alignment_chunk_count: int | None,
    sliding_window_chunks: int | None,
    is_eagle_group: bool,
) -> bool:
    """Return whether an SWA chunk can participate in an external-cache hit."""
    if alignment_chunk_count is None:
        return True
    assert sliding_window_chunks is not None
    position_in_segment = absolute_chunk_index % alignment_chunk_count
    segment_start = absolute_chunk_index - position_in_segment
    actual_segment_length = min(
        alignment_chunk_count, storable_chunk_count - segment_start
    )
    reachable_tail = sliding_window_chunks + int(is_eagle_group)
    return position_in_segment >= actual_segment_length - reachable_tail


def resolve_mamba_align_size(
    spec: "OffloadingSpec", kv_cache_config: KVCacheConfig
) -> int | None:
    """Scan all KV cache groups in *spec* and return the single mamba alignment
    size, or None if no group requires mamba alignment.

    For MambaSpec groups in "align" or "all" cache mode the hit window must be
    rounded down to a multiple of the offloaded chunk size. Asserts that all
    such groups agree on the same value.
    """
    mamba_align_size: int | None = None
    for idx, tokens_per_block in enumerate(spec.tokens_per_block):
        if not spec.config.groups[idx].layer_names:
            # PATCH: inert group -- it carries a placeholder tokens_per_block
            # and never sees offload traffic, so it must not set (or clash
            # with) the mamba alignment size.
            continue
        kv_spec = kv_cache_config.kv_cache_groups[idx].kv_cache_spec
        if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode in (
            "align",
            "all",
        ):
            tokens_per_chunk = tokens_per_block * spec.blocks_per_chunk
            assert mamba_align_size is None or mamba_align_size == tokens_per_chunk
            mamba_align_size = tokens_per_chunk
    return mamba_align_size


class SchedulerOffloadConfig(NamedTuple):
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    blocks_per_chunk: int
    tokens_per_hash: int
    num_workers: int
    offload_prompt_only: bool
    supports_partial_tail: bool

    @classmethod
    def from_spec(
        cls,
        spec: OffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> "SchedulerOffloadConfig":
        # Determine the alignment token count from the full-attention group(s).
        # This is the tokens_per_chunk of the full-attention group; load
        # hits are always aligned to this boundary, so SWA blocks earlier in
        # each segment can never serve a load hit. Relevant for hybrid
        # architectures like DeepSeek V4 (MLA + SWA groups).
        # PATCH: groups kept in position but never offloaded (layer_names=()).
        inert_groups = {
            idx for idx, group in enumerate(spec.config.groups) if not group.layer_names
        }

        full_attn_tokens_per_chunk: set[int] = set()
        for idx, tokens_per_block in enumerate(spec.tokens_per_block):
            if idx in inert_groups:
                continue
            kv_spec = kv_cache_config.kv_cache_groups[idx].kv_cache_spec
            sw = get_sliding_window_size_in_chunks(
                kv_spec, tokens_per_block * spec.blocks_per_chunk
            )
            if sw is None:
                full_attn_tokens_per_chunk.add(tokens_per_block * spec.blocks_per_chunk)

        # Only apply the optimization if there's a single consistent
        # full-attention alignment size.
        alignment_tokens: int | None = None
        if len(full_attn_tokens_per_chunk) == 1:
            alignment_tokens = full_attn_tokens_per_chunk.pop()

        def _alignment_chunk_count(
            tokens_per_chunk: int,
            sliding_window_size_in_chunks: int | None,
        ) -> int | None:
            if alignment_tokens is None or sliding_window_size_in_chunks is None:
                return None
            if alignment_tokens <= tokens_per_chunk:
                return None
            per_segment = alignment_tokens // tokens_per_chunk
            if sliding_window_size_in_chunks >= per_segment:
                return None
            return per_segment

        eagle_groups = {
            idx
            for idx, g in enumerate(kv_cache_config.kv_cache_groups)
            if g.is_eagle_group
        }

        use_eagle = (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.use_eagle()
        )
        # PATCH (local, 2026-08-30): split the single eagle flag in two.
        #
        # `draft_groups` -- groups the platform ANNOTATED as holding a draft
        # model's KV. `KVCacheGroupSpec.is_eagle_group` has exactly one producer
        # in the tree, `_annotate_eagle_groups_deepseek_v4`
        # (v1/core/kv_cache_utils.py), and it flags precisely the group that
        # contains the MTP/draft attention layer. So "annotated" == "draft
        # model group", exactly; an empty set means "we cannot name the draft
        # group", NOT "every group is a draft group".
        #
        # `eagle_groups` -- unchanged conservative fallback (every group when
        # spec decoding is on but nothing is annotated). It still drives the
        # STORE path, where the guard is real and cheap:
        # `_build_store_jobs` derives `num_offloadable_tokens` from
        # `req.num_computed_tokens + num_scheduled_tokens`, and with spec
        # decoding `num_scheduled_tokens` counts 1 + K OPTIMISTIC draft tokens
        # that a rejection can roll back -- so a chunk that "completes" on
        # optimistic tokens must not be stored. `storable_chunks` only applies
        # it while decoding (`num_offloadable_tokens > num_prompt_tokens`), so
        # a chunk completed during PREFILL is stored before the exclusion ever
        # engages, and `advance_stored_idx`'s max() keeps the index from
        # walking back over it.
        #
        # The LOOKUP-side drop (`_lookup_complete_chunks`) is restricted to
        # `draft_groups`, i.e. to nothing at all on this box (GLM-5.3-Flash's
        # only draft group is the DFlash-2 SlidingWindowSpec group, which
        # config.sw-inert.py already marks inert). Reasons it is redundant for
        # TARGET groups:
        #   * Nothing volatile can be in the cache to begin with: the store
        #     guard above is what keeps it out, and a poisoned key would be hit
        #     as an INTERIOR chunk by any longer request, so dropping the
        #     trailing chunk on lookup could not repair it anyway.
        #   * The recompute EAGLE actually needs (hidden states for the last
        #     accepted position) is one TOKEN, not one chunk, and is guaranteed
        #     by the `num_tokens - 1` cap in `_lookup_complete_chunks`.
        # Measured cost of the conservative fallback on this box (47,699-token
        # prompt, KDA block 14,336, MLA chunk 3,584): the MLA group's hit is
        # cut 43,008 -> 39,424 and the mamba align round-down then floors it to
        # 28,672 -- one whole KDA chunk lost on every RAM hit.
        draft_groups = set(eagle_groups)
        if use_eagle and not eagle_groups:
            eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))

        if eagle_groups:
            logger.info(
                "KV offloading: spec decoding active; the trailing chunk of "
                "groups %s is excluded from STORE while decoding (volatile "
                "under draft-token rejection). Annotated draft-model groups "
                "(load-side trailing-chunk drop): %s.",
                sorted(eagle_groups),
                sorted(draft_groups) or "none",
            )

        kv_group_configs_list: list[GroupOffloadConfig] = []
        for idx, tokens_per_block in enumerate(spec.tokens_per_block):
            kv_cache_group = kv_cache_config.kv_cache_groups[idx]
            kv_spec = kv_cache_group.kv_cache_spec
            if idx in inert_groups:
                # PATCH: placeholder keeping group_idx == KV cache group index.
                # tokens_per_chunk / hashes_per_chunk are well-formed (never 0)
                # so the shared per-group bookkeeping loops stay safe, but every
                # store / load / lookup path skips it on is_inert.
                kv_group_configs_list.append(
                    GroupOffloadConfig(
                        group_idx=idx,
                        tokens_per_block=tokens_per_block,
                        tokens_per_chunk=tokens_per_block * spec.blocks_per_chunk,
                        hashes_per_chunk=max(
                            1,
                            (tokens_per_block * spec.blocks_per_chunk)
                            // spec.tokens_per_hash,
                        ),
                        sliding_window_size_in_chunks=None,
                        alignment_chunk_count=None,
                        kv_event_group_spec=get_offloading_event_group_spec(
                            kv_cache_group
                        ),
                        is_eagle_group=False,
                        requires_cow_source=False,
                        is_inert=True,
                    )
                )
                continue
            sw = get_sliding_window_size_in_chunks(
                kv_spec, tokens_per_block * spec.blocks_per_chunk
            )
            kv_group_configs_list.append(
                GroupOffloadConfig(
                    group_idx=idx,
                    tokens_per_block=tokens_per_block,
                    tokens_per_chunk=tokens_per_block * spec.blocks_per_chunk,
                    hashes_per_chunk=(
                        (tokens_per_block * spec.blocks_per_chunk)
                        // spec.tokens_per_hash
                    ),
                    sliding_window_size_in_chunks=sw,
                    alignment_chunk_count=_alignment_chunk_count(
                        tokens_per_block * spec.blocks_per_chunk, sw
                    ),
                    kv_event_group_spec=get_offloading_event_group_spec(kv_cache_group),
                    is_eagle_group=idx in eagle_groups,
                    is_draft_group=idx in draft_groups,
                    requires_cow_source=(
                        isinstance(kv_spec, MambaSpec)
                        and kv_spec.mamba_cache_mode == "align"
                    ),
                )
            )
        kv_group_configs = tuple(kv_group_configs_list)
        # PATCH: shape decisions look at the groups that actually offload.
        active_group_configs = tuple(
            config for config in kv_group_configs if not config.is_inert
        )
        has_partial_recurrent_group = any(
            config.requires_cow_source
            and config.tokens_per_block > spec.tokens_per_hash
            for config in active_group_configs
        )
        # PATCH (local, 2026-08-30): generalise the partial-tail gate.
        #
        # Dropped `not inert_groups`: an inert group has no offloaded blocks,
        # but it also needs none -- every boundary-key loop, store job, load
        # slice and touch below now skips it and still emits its zero-sized
        # GPULoadStoreSpec entry, exactly as the regular chunk path already
        # does. Requiring zero inert groups disabled the optimization outright
        # on GLM-5.3-Flash (kpool tail + DFlash-2 sliding window are inert).
        #
        # Dropped `len(group_block_sizes) == 1`: uniformity was only needed
        # because upstream indexed every group's block table with one shared
        # `boundary // tokens_per_block`. Each group now uses its own
        # `cdiv(boundary, tokens_per_block) - 1` (MLA 3,584 vs KDA 14,336).
        # What IS still required is a single copy-on-write block size, because
        # that is the window the boundary must fall inside.
        #
        # `is_eagle_group` -> `is_draft_group`: the volatile trailing chunk of
        # a DRAFT model's group cannot be handed off, but the conservative
        # "every group is an eagle group" fallback must not veto the feature
        # for the target model. See the rationale above.
        cow_block_sizes = {
            config.tokens_per_block
            for config in active_group_configs
            if config.requires_cow_source
        }
        supports_partial_tail = (
            spec.blocks_per_chunk == 1
            and len(cow_block_sizes) == 1
            and has_partial_recurrent_group
            and all(
                config.sliding_window_size_in_chunks is None
                or config.requires_cow_source
                for config in active_group_configs
            )
            and all(
                config.tokens_per_block % spec.tokens_per_hash == 0
                for config in active_group_configs
            )
            and not any(config.is_draft_group for config in active_group_configs)
            and vllm_config.parallel_config.decode_context_parallel_size == 1
        )

        return cls(
            num_workers=vllm_config.parallel_config.world_size,
            kv_group_configs=kv_group_configs,
            blocks_per_chunk=spec.blocks_per_chunk,
            tokens_per_hash=spec.tokens_per_hash,
            offload_prompt_only=spec.offload_prompt_only,
            supports_partial_tail=supports_partial_tail,
        )


@dataclass
class RequestGroupState:
    offload_keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    # Index of the next chunk to offload.
    next_stored_chunk_idx: int = 0
    # Number of offloaded chunks hit (including GPU prefix cache)
    # when the request first started
    num_hit_chunks: int = 0


@dataclass(slots=True)
class RequestOffloadState:
    config: SchedulerOffloadConfig
    req: Request
    req_context: ReqContext
    offloading_context: RequestOffloadingContext
    group_states: tuple[RequestGroupState, ...] = field(init=False)
    # upper bound on tokens to offload for this request; None means no cap
    max_offload_tokens: int | None = None
    # number of hits in the GPU cache
    num_locally_computed_tokens: int = 0
    # In-flight job IDs. Per the connector's invariant, at any given time
    # this contains either a single load job, or one or more store jobs.
    transfer_jobs: set[int] = field(default_factory=set)
    # time.monotonic() of this request's first deferred offload lookup;
    # None once consumed (observed) or while no lookup is pending.
    deferred_lookup_start_time: float | None = None
    # Fine-grained token boundary selected beyond the last complete offload
    # chunk. It is consumed when the corresponding load is scheduled.
    partial_tail_boundary: int | None = None
    # True once on_request_finished has been signaled to the manager.
    finished_signaled: bool = False

    def __post_init__(self) -> None:
        self.group_states = tuple(
            RequestGroupState() for _ in self.config.kv_group_configs
        )
        params = self.req.kv_transfer_params

        # NOTE: This field is experimental and subject to change in the future.
        raw = params.get("max_offload_tokens") if params else None
        if type(raw) is int and raw >= 0:
            self.max_offload_tokens = raw
            logger.debug(
                "Request %s: max_offload_tokens set to %d",
                self.req.request_id,
                raw,
            )
        elif raw is not None:
            logger.warning(
                "max_offload_tokens must be a non-negative int, got %r; ignoring", raw
            )

    def update_offload_keys(self) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            if group_config.is_inert:
                # PATCH: never hashed, so it never gets offload keys.
                continue
            for req_block_hash in islice(
                self.req.block_hashes,
                group_config.hashes_per_chunk * len(group_state.offload_keys)
                + group_config.hashes_per_chunk
                - 1,
                None,
                group_config.hashes_per_chunk,
            ):
                group_state.offload_keys.append(
                    make_offload_key(req_block_hash, group_config.group_idx)
                )

    def update_block_id_groups(
        self, new_block_id_groups: tuple[list[int], ...] | None
    ) -> None:
        if new_block_id_groups is None:
            return

        assert len(new_block_id_groups) == len(self.group_states)
        for group_state, new_blocks in zip(self.group_states, new_block_id_groups):
            group_state.block_ids.extend(new_blocks)

    def storable_chunks(
        self,
        group_config: "GroupOffloadConfig",
        group_state: RequestGroupState,
        num_offloadable_tokens: int,
    ) -> int:
        """Number of allocated leading offloaded chunks eligible for store.

        For eagle/MTP groups the volatile trailing chunk of the offloadable
        range is excluded while decoding: the draft-layer KV of the last
        accepted position may be rewritten after spec-token rejection. During
        prefill the trailing chunk is stable (the draft input for a chunk's
        last position is the next prompt token), so it is stored immediately.
        The exclusion must be applied consistently everywhere
        ``next_stored_chunk_idx`` is derived: otherwise the trailing chunk of
        each step is skipped on collection but jumped over by
        ``next_stored_chunk_idx``, so it is never re-considered and a
        permanent hole breaks prefix-reuse lookup.
        """
        num_chunks = num_offloadable_tokens // group_config.tokens_per_chunk
        is_decoding = num_offloadable_tokens > self.req.num_prompt_tokens
        if group_config.is_eagle_group and is_decoding:
            num_chunks = max(0, num_chunks - 1)
        num_allocated_chunks = (
            len(group_state.block_ids) // self.config.blocks_per_chunk
        )
        return min(num_chunks, num_allocated_chunks)

    def advance_stored_idx(self, num_offloadable_tokens: int) -> None:
        # max(): at the prefill->decode transition of a chunk-aligned prompt,
        # storable_chunks drops by one (the eagle exclusion kicks in), and the
        # index must not move backwards past already-stored chunks.
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.next_stored_chunk_idx = max(
                group_state.next_stored_chunk_idx,
                self.storable_chunks(group_config, group_state, num_offloadable_tokens),
            )

    def update_num_hit_chunks(self, num_cached_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.num_hit_chunks = (
                num_cached_tokens // group_config.tokens_per_chunk
            )


def _parse_tier_filter(raw: Any) -> TierFilter:
    """Parse raw kv_transfer_params tier matchers into a TierFilter."""
    if not isinstance(raw, list):
        logger.warning(
            "_parse_tier_filter: expected list, got %s; ignoring",
            type(raw).__name__,
        )
        return TierFilter.ALL
    matchers: list[TierMatcher] = []
    for entry in raw:
        if not isinstance(entry, dict):
            logger.warning("_parse_tier_filter: entry is not a dict; skipping")
            continue
        medium: Medium | None = None
        locality: Locality | None = None
        raw_medium = entry.get(MATCHER_MEDIUM_KEY)
        if raw_medium is not None:
            try:
                medium = Medium(raw_medium.upper())
            except (ValueError, AttributeError):
                logger.warning(
                    "_parse_tier_filter: unknown medium %r; skipping entry",
                    raw_medium,
                )
                continue
        raw_locality = entry.get(MATCHER_LOCALITY_KEY)
        if raw_locality is not None:
            try:
                locality = Locality(raw_locality.upper())
            except (ValueError, AttributeError):
                logger.warning(
                    "_parse_tier_filter: unknown locality %r; skipping entry",
                    raw_locality,
                )
                continue
        matchers.append(TierMatcher(medium=medium, locality=locality))
    if not matchers:
        if not raw:  # input was [] — user explicitly wants nothing
            return TierFilter(matchers=())
        # all entries were invalid — fall back to ALL
        return TierFilter.ALL
    return TierFilter(matchers=tuple(matchers))


def _create_req_context(req: Request) -> ReqContext:
    params = req.kv_transfer_params
    load_filter = TierFilter.ALL
    if params:
        raw = params.get(KV_LOAD_TIERS_KEY)
        if raw is not None:
            load_filter = _parse_tier_filter(raw)
    return ReqContext(
        req_id=req.request_id,
        kv_transfer_params=params,
        load_tier_filter=load_filter,
    )


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(
        self,
        spec: OffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        self.config = SchedulerOffloadConfig.from_spec(
            spec, vllm_config, kv_cache_config
        )
        self.manager: OffloadingManager = spec.get_manager()
        self._connector_stats = OffloadingConnectorStats()

        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in self.config.kv_group_configs:
            if group_config.is_inert:
                # PATCH: never looked up.
                continue
            if group_config.sliding_window_size_in_chunks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)

        # sort sliding window groups by window size in decreasing order
        def _sliding_window_sort_key(i: int) -> int:
            val = self.config.kv_group_configs[i].sliding_window_size_in_chunks
            assert val is not None
            return val

        sliding_window_groups.sort(key=_sliding_window_sort_key, reverse=True)

        # used by _lookup
        self._sliding_window_groups: tuple[int, ...] = tuple(sliding_window_groups)
        self._lookup_groups = tuple(full_attention_groups) + self._sliding_window_groups
        self._mamba_align_size: int | None = resolve_mamba_align_size(
            spec, kv_cache_config
        )
        self._cow_source_groups = frozenset(
            config.group_idx
            for config in self.config.kv_group_configs
            if config.requires_cow_source and not config.is_inert
        )
        # PATCH (local, 2026-08-30): the hand-off boundary lives INSIDE one
        # block of the copy-on-write (mamba "align") group, so the search
        # window is THAT group's block, not kv_group_configs[0]'s -- upstream
        # could use group 0 only because it also required one uniform block
        # size across groups. Here group 0 is MLA (3,584) while the recurrent
        # state block is 14,336, and using 3,584 would cap the scan a whole
        # KDA block short. `from_spec` guarantees exactly one such size.
        self._partial_tail_block_size = 0
        if self.config.supports_partial_tail:
            _cow_block_sizes = {
                config.tokens_per_block
                for config in self.config.kv_group_configs
                if config.requires_cow_source and not config.is_inert
            }
            assert len(_cow_block_sizes) == 1
            self._partial_tail_block_size = _cow_block_sizes.pop()
        # PATCH (local, 2026-08-30): mirror the GPU-side eagle back-off.
        # MambaManager._cache_partial_tail_block (overlay
        # patches/cstech-partial-tail/single_type_kv_cache_manager.py)
        # registers the tail at `last_prompt_hash_boundary - hash_block_size`
        # exactly when its own `use_eagle` is set, and the coordinator sets
        # that from `KVCacheGroupSpec.is_eagle_group` with the same
        # "flag everything when nothing is annotated" fallback
        # (v1/core/kv_cache_coordinator.py). Reproduce the predicate here so
        # the scan starts at the boundary that is actually stored instead of
        # burning one lookup round per group on the un-backed-off one.
        self._eagle_tail_backoff = (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.use_eagle()
            and not any(g.is_eagle_group for g in kv_cache_config.kv_cache_groups)
        )

        self._req_status: dict[ReqId, RequestOffloadState] = {}
        self._current_batch_load_jobs: dict[int, TransferJob] = {}
        self._current_batch_jobs_to_flush: set[int] = set()
        # GPU block IDs allocated in the current engine step
        self._current_batch_allocated_block_ids: set[int] = set()
        # if GPU prefix caching is enabled,
        # Track loaded chunks to avoid redundant loads.
        self._chunks_being_loaded: set[OffloadKey] | None = (
            set() if vllm_config.cache_config.enable_prefix_caching else None
        )

        # Job ID counter shared by loads and stores.
        self._job_counter: int = 0
        # Threshold value for stale jobs. All job ids >= _stale_job_threshold are
        # active jobs.
        self._stale_job_threshold: int = 0
        self._jobs: dict[int, TransferJobStatus] = {}

        # block_id -> pending store job_ids. Used to track jobs that needs
        # flushing in case a block is re-allocated by the KV cache manager.
        # Populated only for finished requests (running-request blocks are
        # protected by their ref_cnt) and for sliding window blocks (which can
        # be freed before a request finishes).
        self._block_id_to_pending_jobs: dict[int, set[int]] = {}

        self._events_tracker = OffloadingEventsTracker(spec.kv_events_config)

    def _maybe_observe_lookup_async_delay(
        self, req_status: RequestOffloadState
    ) -> None:
        start_time = req_status.deferred_lookup_start_time
        if start_time is None:
            return
        req_status.deferred_lookup_start_time = None
        self._connector_stats.observe_histogram(
            _ConnectorMetricName.LOOKUP_ASYNC_DELAY,
            time.monotonic() - start_time,
        )

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter += 1
        return job_id

    def _remove_pending_job(self, job_id: int, block_ids: list[int] | None) -> None:
        for bid in block_ids or ():
            pending = self._block_id_to_pending_jobs[bid]
            pending.remove(job_id)
            if not pending:
                del self._block_id_to_pending_jobs[bid]

    def _calc_num_offloadable_tokens(
        self, req_status: RequestOffloadState, num_computed_tokens: int
    ) -> int:
        num = min(num_computed_tokens, req_status.req.num_tokens)
        max_offload_tokens = req_status.max_offload_tokens
        if max_offload_tokens is not None:
            num = min(num, max_offload_tokens)
        if self.config.offload_prompt_only:
            num = min(num, req_status.req.num_prompt_tokens)
        return num

    def _maximal_prefix_lookup(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext,
        req: Request,
        group_config: GroupOffloadConfig,
        start_chunk_idx: int,
    ) -> int | None:
        """Return the number of consecutive offloaded chunks from the start,
        or None if the backend deferred a lookup."""
        hit_count = 0
        defer_lookup = False
        for local_idx, key in enumerate(keys):
            result = self.manager.lookup(key, req_context)
            match result:
                case LookupResult.HIT:
                    self._events_tracker.record_lookup(
                        req,
                        group_config,
                        start_chunk_idx + local_idx,
                        key,
                    )
                    hit_count += 1
                case LookupResult.HIT_PENDING:
                    defer_lookup = True
                    hit_count += 1
                case LookupResult.RETRY:
                    # Don't break: keep scanning to let manager kick off
                    # async lookups (until a miss is detected).
                    defer_lookup = True
                case LookupResult.MISS:
                    break
        return hit_count if not defer_lookup else None

    def _sliding_window_lookup(
        self,
        keys: Sequence[OffloadKey],
        sliding_window_size: int,
        req_context: ReqContext,
    ) -> int | None:
        """Return the end index (in `keys`) of the last run of
        `sliding_window_size` consecutive hits, scanning from the end.
        Returns 0 on miss, None if the backend deferred a lookup."""
        defer_lookup = False
        consecutive_hits = 0
        for idx in range(len(keys) - 1, -1, -1):
            match self.manager.lookup(keys[idx], req_context):
                case LookupResult.HIT:
                    consecutive_hits += 1
                case LookupResult.HIT_PENDING:
                    # Block is in cache, just not readable yet — counts
                    # as hit for the consecutive streak. Don't break:
                    # keep scanning to let manager kick off async lookups.
                    defer_lookup = True
                    consecutive_hits += 1
                case LookupResult.RETRY:
                    # Block location uncertain — does not count as hit.
                    # Don't break: keep scanning to let manager kick off
                    # async lookups.
                    defer_lookup = True
                    consecutive_hits = 0
                case LookupResult.MISS:
                    consecutive_hits = 0
            if consecutive_hits == sliding_window_size:
                return idx + sliding_window_size if not defer_lookup else None
        return consecutive_hits if not defer_lookup else None

    def _touch(self, req_status: RequestOffloadState):
        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            if group_config.is_inert:
                # PATCH: no keys to touch.
                continue
            if group_config.sliding_window_size_in_chunks is None:
                self.manager.touch(group_state.offload_keys, req_status.req_context)
            else:
                # Keep only chunks needed to hit the original request, plus
                # decoded chunks.
                chunks_to_skip = max(
                    0,
                    group_state.num_hit_chunks
                    - group_config.sliding_window_size_in_chunks,
                )
                self.manager.touch(
                    group_state.offload_keys[chunks_to_skip:],
                    req_status.req_context,
                )
        if req_status.partial_tail_boundary is not None:
            self.manager.touch(
                tuple(
                    self._make_boundary_key(
                        req_status.req,
                        group.group_idx,
                        req_status.partial_tail_boundary,
                    )
                    # PATCH (local, 2026-08-30): an inert group has no boundary
                    # key -- touching one would be a lookup against a key that
                    # was never stored.
                    for group in self.config.kv_group_configs
                    if not group.is_inert
                ),
                req_status.req_context,
            )

    def _lookup_complete_chunks(self, req_status: RequestOffloadState) -> int | None:
        """
        Find how many tokens beyond num_locally_computed_tokens can be loaded.

        Iterates full-attention groups first (prefix lookup), then sliding-window
        groups (suffix lookup). Each group may tighten max_hit_size_tokens, which
        can invalidate an earlier group's result, so the loop re-runs when that
        happens until num_hit_tokens converges.
        """
        num_computed_tokens = req_status.num_locally_computed_tokens
        max_hit_size_tokens: int = req_status.req.num_tokens
        if self._sliding_window_groups or self._eagle_tail_backoff:
            # the last prompt token has to be recomputed to get the logprobs
            # for sliding window attention, we must reduce by 1 to make sure
            # we still have a hit after reduction
            # PATCH (local, 2026-08-30): also apply the -1 when the eagle
            # trailing-chunk drop below has been narrowed to draft-model groups
            # (see from_spec). EAGLE needs the hidden states of the last
            # accepted position, which means at least one token must be
            # recomputed -- one TOKEN, which this guarantees, not one chunk.
            max_hit_size_tokens -= 1
        if self._eagle_tail_backoff:
            # PATCH (local, 2026-08-30): same N-unit back-off as the GPU path,
            # so a complete-chunk RAM hit never lands within the drafter's
            # window of the prompt end either.
            max_hit_size_tokens = min(
                max_hit_size_tokens,
                req_status.req.num_tokens - self.config.tokens_per_hash * _EAGLE_DROP_UNITS,
            )
        if self._mamba_align_size is not None:
            # Constrain hit-window to the mamba block size.
            # PATCH (local, 2026-08-30): unconditional, not only with SW groups.
            max_hit_size_tokens = round_down(
                max_hit_size_tokens, self._mamba_align_size
            )

        num_hit_tokens: int = 0
        defer_lookup = False
        lookup_groups = self._lookup_groups

        # Tracks which eagle groups have already popped their volatile trailing chunk
        # in the current convergence iteration. Reset when a non-eagle group
        # tightens the hit boundary, requiring a fresh pop.
        eagle_verified: set[int] = set()
        while lookup_groups:
            looked_up_sliding_window: bool = False
            groups_iter = iter(lookup_groups)
            lookup_groups = ()
            for group_idx in groups_iter:
                group_config: GroupOffloadConfig = self.config.kv_group_configs[
                    group_idx
                ]
                group_state: RequestGroupState = req_status.group_states[group_idx]
                tokens_per_chunk = group_config.tokens_per_chunk
                offload_keys = group_state.offload_keys

                assert (
                    len(offload_keys) >= req_status.req.num_tokens // tokens_per_chunk
                )

                # PATCH (local, 2026-08-30): only a real draft-model group drops
                # its trailing chunk on lookup (see from_spec). With the
                # conservative "every group" fallback this cost the MLA group
                # 3,584 tokens, which the mamba round-down below then amplified
                # to a whole 14,336-token KDA chunk.
                is_eagle_unverified = (
                    group_config.is_draft_group and group_idx not in eagle_verified
                )

                # Constrain to a chunk-aligned boundary for this group.
                max_hit_size_tokens = min(
                    max_hit_size_tokens, len(offload_keys) * tokens_per_chunk
                )
                if max_hit_size_tokens - num_computed_tokens < tokens_per_chunk:
                    # We can only load less than a chunk, so skip.
                    return 0

                sliding_window_size_in_chunks = (
                    group_config.sliding_window_size_in_chunks
                )

                # For eagle groups, query one extra chunk that will be popped.
                # We only need to increase the query size for sliding window groups.
                query_max = max_hit_size_tokens
                if is_eagle_unverified and sliding_window_size_in_chunks is not None:
                    query_max = min(
                        max_hit_size_tokens + tokens_per_chunk,
                        len(offload_keys) * tokens_per_chunk,
                    )

                num_chunks = min(cdiv(query_max, tokens_per_chunk), len(offload_keys))
                start_chunk_idx = num_computed_tokens // tokens_per_chunk
                offload_keys = offload_keys[start_chunk_idx:num_chunks]

                # end index (in the sliced offload_keys) up to which we
                # have backend-confirmed hits
                num_hit_chunks: int | None
                if sliding_window_size_in_chunks is None:
                    num_hit_chunks = self._maximal_prefix_lookup(
                        offload_keys,
                        req_status.req_context,
                        req_status.req,
                        group_config,
                        start_chunk_idx,
                    )
                else:
                    required_window = sliding_window_size_in_chunks
                    if is_eagle_unverified:
                        required_window += 1
                    num_hit_chunks = self._sliding_window_lookup(
                        offload_keys,
                        required_window,
                        req_status.req_context,
                    )
                if num_hit_chunks == 0:
                    return 0

                if num_hit_chunks is None:
                    defer_lookup = True
                else:
                    if is_eagle_unverified:
                        num_hit_chunks -= 1
                        eagle_verified.add(group_idx)

                    max_hit_size_tokens = min(
                        max_hit_size_tokens,
                        tokens_per_chunk * (start_chunk_idx + num_hit_chunks),
                    )
                    # PATCH (local, 2026-08-30): keep the candidate hit on the
                    # mamba state-checkpoint grid INSIDE the convergence loop.
                    # A group whose chunk is finer than the mamba align size
                    # (MLA 3,584 / draft SW 896 vs KDA 14,336 with
                    # --mamba-block-size 14336) can tighten the bound to a
                    # position where no recurrent state was ever stored; the
                    # KDA group's own lookup would then be asked for
                    # cdiv(bound, 14336) chunks and the load would fetch the
                    # state of a LATER token position (silent wrong output).
                    # Re-rounding here (not after the loop) makes the shrink
                    # visible to the existing re-verification logic, so the
                    # sliding-window groups are re-checked at the new boundary.
                    # Inert whenever every group's chunk is a multiple of the
                    # align size (the default 3,584 config).
                    if self._mamba_align_size is not None:
                        max_hit_size_tokens = round_down(
                            max_hit_size_tokens, self._mamba_align_size
                        )

                new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens
                if new_num_hit_tokens < tokens_per_chunk:
                    # We can only load less than a chunk, so skip.
                    return 0

                if new_num_hit_tokens < num_hit_tokens:
                    # PATCH (local, 2026-08-30): pairs with is_draft_group above.
                    if not group_config.is_draft_group:
                        eagle_verified.clear()
                    if defer_lookup:
                        # make another iteration on all groups to check
                        # if we still need to defer lookup
                        defer_lookup = False
                        lookup_groups = self._lookup_groups
                    elif looked_up_sliding_window and not lookup_groups:
                        # we need another iteration to confirm previously looked up
                        # sliding window works with the new_num_hit_tokens
                        lookup_groups = self._sliding_window_groups

                looked_up_sliding_window |= sliding_window_size_in_chunks is not None
                num_hit_tokens = new_num_hit_tokens

        if defer_lookup:
            logger.debug(
                "Offloading manager delayed request %s as backend requested",
                req_status.req.request_id,
            )
            return None

        # Possibly delay the request if any hit chunk is already being loaded.
        if self._chunks_being_loaded:
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                tokens_per_chunk = group_config.tokens_per_chunk
                sliding_window_size_in_chunks = (
                    group_config.sliding_window_size_in_chunks
                )
                offload_keys = group_state.offload_keys
                num_chunks = cdiv(
                    num_computed_tokens + num_hit_tokens, tokens_per_chunk
                )
                start_chunk_idx = num_computed_tokens // tokens_per_chunk
                offload_keys = offload_keys[start_chunk_idx:num_chunks]
                if sliding_window_size_in_chunks is not None:
                    offload_keys = offload_keys[-sliding_window_size_in_chunks:]
                if any(key in self._chunks_being_loaded for key in offload_keys):
                    # Hit chunks are being loaded, so delay the request.
                    logger.debug(
                        "Delaying request %s since some of its"
                        " chunks are already being loaded",
                        req_status.req.request_id,
                    )
                    return None

        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            req_status.req.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )

        return num_hit_tokens

    def _make_boundary_key(
        self, request: Request, group_idx: int, boundary_tokens: int
    ) -> OffloadKey:
        hash_idx = boundary_tokens // self.config.tokens_per_hash - 1
        return make_offload_key(request.block_hashes[hash_idx], group_idx)

    def _boundary_block_idx(
        self, group_config: GroupOffloadConfig, boundary: int
    ) -> int:
        """Index of the block that must hold this group's KV/state at `boundary`.

        PATCH (local, 2026-08-30): per-group, because block sizes are not
        uniform here (MLA 3,584, KDA 14,336). ``cdiv - 1`` collapses to the last
        REGULAR chunk when the boundary is an exact multiple of the group's
        block (46,592 = 13 x 3,584 for MLA) and to the partial block otherwise
        (46,592 = 3 x 14,336 + 3,584 for KDA). In the first case
        ``_make_boundary_key(boundary)`` resolves to hash index
        ``boundary // tokens_per_hash - 1``, which is byte-for-byte the key
        ``update_offload_keys`` already assigned to that chunk
        (``block_hashes[hashes_per_chunk * chunk + hashes_per_chunk - 1]``), so
        the boundary key and the chunk key are the SAME key and nothing is
        loaded or stored twice.
        """
        return cdiv(boundary, group_config.tokens_per_block) - 1

    def _boundary_probe_keys(
        self,
        req_status: RequestOffloadState,
        group_config: GroupOffloadConfig,
        complete_boundary: int,
        boundary: int,
    ) -> list[OffloadKey] | None:
        """Keys this group needs before `boundary` is a valid resume point.

        PATCH (local, 2026-08-30): upstream probed exactly one key per group,
        which is only sufficient when the complete-chunk hit already verified
        everything below the boundary. It does not here: `_lookup_complete_chunks`
        rounds the complete hit DOWN to the mamba align size (14,336), so a
        finer group's regular chunks between the complete hit and the candidate
        boundary (MLA 3,584) were never looked up. Probe them:
          * full-attention group: every regular chunk in
            [complete_boundary, boundary) plus the boundary block;
          * windowed group (mamba w == 1, SWA w > 1): only the last w blocks
            ending at the boundary -- earlier chunks are unreachable for it and
            `_sliding_window_lookup` never asks for them.
        Returns None when a needed key does not exist yet for this request.
        """
        offload_keys = req_status.group_states[group_config.group_idx].offload_keys
        end_idx = self._boundary_block_idx(group_config, boundary)
        if end_idx < 0 or end_idx > len(offload_keys):
            return None
        hash_idx = boundary // self.config.tokens_per_hash - 1
        if hash_idx < 0 or hash_idx >= len(req_status.req.block_hashes):
            return None
        window = group_config.sliding_window_size_in_chunks
        if window is None:
            first_idx = complete_boundary // group_config.tokens_per_block
        else:
            first_idx = max(0, end_idx - (window - 1))
        first_idx = min(first_idx, end_idx)
        keys = list(offload_keys[first_idx:end_idx])
        keys.append(
            self._make_boundary_key(req_status.req, group_config.group_idx, boundary)
        )
        return keys

    def _lookup(self, req_status: RequestOffloadState) -> int | None:
        complete_hit = self._lookup_complete_chunks(req_status)
        req_status.partial_tail_boundary = None
        if complete_hit is None or not self.config.supports_partial_tail:
            return complete_hit

        local_tokens = req_status.num_locally_computed_tokens
        complete_boundary = local_tokens + complete_hit
        tokens_per_hash = self.config.tokens_per_hash
        block_end = complete_boundary + self._partial_tail_block_size
        max_boundary = round_down(
            min(req_status.req.num_prompt_tokens - 1, block_end - 1), tokens_per_hash
        )
        if self._eagle_tail_backoff:
            # PATCH (local, 2026-08-30): under EAGLE/MTP the producer registers
            # its partial tail one hash unit BEFORE the prompt's last hash
            # boundary (MambaManager._cache_partial_tail_block overlay), because
            # the attention groups' eagle drop lands there. Start the scan at
            # that boundary: anything above it was never stored, and the drafter
            # gets its hidden states from the recomputed tail.
            max_boundary = min(
                max_boundary,
                round_down(req_status.req.num_prompt_tokens, tokens_per_hash)
                - tokens_per_hash,
            )
        if max_boundary <= complete_boundary:
            return complete_hit

        pending = False
        for boundary in range(max_boundary, complete_boundary, -tokens_per_hash):
            boundary_pending = False
            boundary_missed = False
            # (group_config, boundary key) for the event tracker; probe_keys
            # additionally carries the intermediate regular chunks.
            boundary_keys: list[tuple[GroupOffloadConfig, OffloadKey]] = []
            probe_keys: list[OffloadKey] = []
            for group_config in self.config.kv_group_configs:
                if group_config.is_inert:
                    # PATCH (local, 2026-08-30): never stored, so it can never
                    # hit; probing it would veto every boundary. Its GPU blocks
                    # stay freshly allocated, exactly as after a GPU prefix-cache
                    # hit (it opts out of prefix caching).
                    continue
                keys = self._boundary_probe_keys(
                    req_status, group_config, complete_boundary, boundary
                )
                if keys is None:
                    boundary_missed = True
                    break
                boundary_keys.append((group_config, keys[-1]))
                probe_keys.extend(keys)

            if not boundary_missed:
                for key in probe_keys:
                    result = self.manager.lookup(key, req_status.req_context)
                    if result is LookupResult.MISS:
                        boundary_missed = True
                        break
                    if result in (LookupResult.HIT_PENDING, LookupResult.RETRY):
                        boundary_pending = True

            pending |= boundary_pending
            if not boundary_missed and not boundary_pending:
                for group_config, key in boundary_keys:
                    self._events_tracker.record_partial_lookup(
                        req_status.req, group_config, boundary, key
                    )
                req_status.partial_tail_boundary = boundary
                return boundary - local_tokens

        if pending and complete_hit == 0:
            return None
        return complete_hit

    def on_new_request(self, request: Request) -> None:
        """Called when a new request is added to the scheduler."""
        req_context = _create_req_context(request)
        offloading_context = self.manager.on_new_request(req_context)
        req_status = RequestOffloadState(
            config=self.config,
            req=request,
            req_context=req_context,
            offloading_context=offloading_context,
        )
        self._req_status[request.request_id] = req_status

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        req_status = self._req_status[request.request_id]
        for group_state in req_status.group_states:
            group_state.block_ids.clear()

        if req_status.transfer_jobs:
            logger.debug(
                "Delaying request %s since it still has in-flight transfers",
                request.request_id,
            )
            return None, False

        req_status.update_offload_keys()
        req_status.num_locally_computed_tokens = num_computed_tokens

        num_hit_tokens: int | None
        if request.skip_reading_prefix_cache:
            num_hit_tokens = 0
        else:
            lookup_start = time.monotonic()
            num_hit_tokens = self._lookup(req_status)
            self._connector_stats.observe_histogram(
                _ConnectorMetricName.LOOKUP_SYNC_DELAY,
                time.monotonic() - lookup_start,
            )
            if num_hit_tokens is None:
                if req_status.deferred_lookup_start_time is None:
                    req_status.deferred_lookup_start_time = lookup_start
            else:
                self._maybe_observe_lookup_async_delay(req_status)
        req_status.update_num_hit_chunks(num_computed_tokens + (num_hit_tokens or 0))

        self._touch(req_status)

        return num_hit_tokens, bool(num_hit_tokens)

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if num_external_tokens == 0:
            return

        req_status = self._req_status[request.request_id]

        num_locally_computed_tokens = req_status.num_locally_computed_tokens
        num_cached_tokens = num_locally_computed_tokens + num_external_tokens
        partial_tail_boundary = req_status.partial_tail_boundary
        if partial_tail_boundary is not None:
            assert partial_tail_boundary == num_cached_tokens

        keys_to_load: list[OffloadKey] = []
        dst_block_ids: list[int] = []
        # per group
        group_sizes: list[int] = []
        block_indices: list[int] = []
        for group_config, group_state, group_blocks in zip(
            self.config.kv_group_configs,
            req_status.group_states,
            blocks.blocks,
        ):
            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

            if group_config.is_inert:
                # PATCH: nothing offloaded for this group -- contribute an
                # empty slice so the worker's per-group arrays stay aligned.
                # An offload hit therefore leaves this group's freshly
                # allocated GPU blocks untouched, exactly as a GPU prefix
                # cache hit does (it opts out of prefix caching).
                group_sizes.append(0)
                block_indices.append(0)
                continue

            tokens_per_block = group_config.tokens_per_block
            tokens_per_chunk = group_config.tokens_per_chunk
            offload_keys = group_state.offload_keys
            num_gpu_blocks = cdiv(num_cached_tokens, tokens_per_block)

            assert len(group_blocks) >= num_gpu_blocks
            num_locally_computed_gpu_blocks = num_gpu_blocks
            # Skip null placeholder blocks (used for sliding window or mamba padding).
            for i, block in enumerate(group_blocks[:num_gpu_blocks]):
                if not block.is_null and block.block_hash is None:
                    num_locally_computed_gpu_blocks = i
                    break

            assert (
                num_locally_computed_tokens
                <= num_locally_computed_gpu_blocks * tokens_per_block
            )
            num_pending_gpu_blocks = num_gpu_blocks - num_locally_computed_gpu_blocks

            if group_config.sliding_window_size_in_chunks is not None:
                assert (
                    num_pending_gpu_blocks
                    <= group_config.sliding_window_size_in_chunks
                    * self.config.blocks_per_chunk
                    + 1
                )

            num_chunks = cdiv(num_cached_tokens, tokens_per_chunk)
            if num_pending_gpu_blocks:
                start_chunk_idx = (
                    num_locally_computed_gpu_blocks // self.config.blocks_per_chunk
                )
                end_chunk_idx = num_chunks - (partial_tail_boundary is not None)
                # PATCH (local, 2026-08-30): this is exactly "[chunks fully
                # below the boundary block] + [the boundary block]", per group,
                # with NON-uniform block sizes. With blocks_per_chunk == 1
                # (required by supports_partial_tail) tokens_per_chunk ==
                # tokens_per_block, so num_chunks - 1 == cdiv(boundary,
                # tokens_per_block) - 1 == this group's boundary block index.
                # When the boundary is an exact multiple of the group's block
                # (MLA: 46,592 = 13 x 3,584) that index IS the last regular
                # chunk and `_make_boundary_key` returns that chunk's own key,
                # so the slice + append still names each chunk exactly once --
                # no double load. The key count stays equal to
                # num_pending_gpu_blocks in both cases:
                #   (num_chunks - 1 - start) + 1 == num_chunks - start
                #                               == num_gpu_blocks - start.
                assert (
                    partial_tail_boundary is None
                    or end_chunk_idx
                    == self._boundary_block_idx(group_config, partial_tail_boundary)
                )
                assert len(offload_keys) >= end_chunk_idx
                keys_to_load.extend(offload_keys[start_chunk_idx:end_chunk_idx])
                if partial_tail_boundary is not None:
                    keys_to_load.append(
                        self._make_boundary_key(
                            request, group_config.group_idx, partial_tail_boundary
                        )
                    )

            dst_block_ids.extend(
                block.block_id
                for block in group_blocks[
                    num_locally_computed_gpu_blocks:num_gpu_blocks
                ]
            )
            group_sizes.append(num_pending_gpu_blocks)
            # PATCH (local, 2026-08-30): for the mamba "align" group with no
            # local GPU hit this is `boundary // tokens_per_block` (the leading
            # placeholders are null blocks, which the scan above steps over), so
            # the boundary state is written into block index
            # `boundary // mamba_block_size`. That is the same slot the runner
            # reads: mamba_hybrid.py seeds `_mamba_state_idx` with
            # `(num_computed_tokens - 1) // mamba_block_size`, and for a
            # boundary that is not a multiple of mamba_block_size the two
            # expressions agree (e.g. 46,592 // 14,336 == 3 ==
            # (46,592 - 1) // 14,336).
            block_indices.append(num_locally_computed_gpu_blocks)

            # Skip prefix-hit chunks for block-level policy; for
            # request-level, next_stored_chunk_idx stays at 0 so all
            # chunks (including hits) are offloaded.
            if req_status.offloading_context.policy == OffloadPolicy.BLOCK_LEVEL:
                group_state.next_stored_chunk_idx = num_chunks

        src_spec = self.manager.prepare_load(keys_to_load, req_status.req_context)
        dst_spec = GPULoadStoreSpec(
            dst_block_ids, group_sizes=group_sizes, block_indices=block_indices
        )

        load_job_id = self._generate_job_id()
        self._current_batch_load_jobs[load_job_id] = TransferJob(
            req_id=request.request_id,
            src_spec=src_spec,
            dst_spec=dst_spec,
        )
        # a load can only be issued when no other jobs are pending.
        assert not req_status.transfer_jobs
        req_status.transfer_jobs.add(load_job_id)
        self._jobs[load_job_id] = TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=set(keys_to_load),
            is_store=False,
        )

        if self._chunks_being_loaded is not None:
            self._chunks_being_loaded.update(keys_to_load)
        req_status.partial_tail_boundary = None

    def _update_req_states(self, scheduler_output: SchedulerOutput) -> None:
        """
        Update request states from the Scheduler's output.
        """

        # new_block_ids_end[req_id][i] = end of pre-existing block_ids for
        # the i-th sliding window group (before this step's extend).
        # Used to detect sliding window blocks that got re-allocated.
        new_block_ids_end: dict[str, tuple[int, ...]] = {}

        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            req_status = self._req_status[req_id]
            req_status.update_offload_keys()

            if preempted:
                for group_state in req_status.group_states:
                    group_state.block_ids.clear()

            if new_block_id_groups:
                if self._sliding_window_groups:
                    new_block_ids_end[req_id] = tuple(
                        len(req_status.group_states[grp_idx].block_ids)
                        for grp_idx in self._sliding_window_groups
                    )
                req_status.update_block_id_groups(new_block_id_groups)
                for new_blocks in new_block_id_groups:
                    for bid in new_blocks:
                        if bid != 0:
                            self._current_batch_allocated_block_ids.add(bid)

        # Zero out stale block_ids in sliding window groups' pending-store
        # positions. Only sliding window groups can have stale entries (blocks
        # freed by remove_skipped_blocks then reallocated). Only positions in
        # [next_stored_chunk_idx * bsf, end) need checking where end is the
        # pre-extend length: earlier positions were already offloaded, later
        # ones are fresh allocations from this step.
        if self._sliding_window_groups and self._current_batch_allocated_block_ids:
            blocks_per_chunk = self.config.blocks_per_chunk
            for req_id, req_status in self._req_status.items():
                ends = new_block_ids_end.get(req_id)
                for i, grp_idx in enumerate(self._sliding_window_groups):
                    group_state = req_status.group_states[grp_idx]
                    start = group_state.next_stored_chunk_idx * blocks_per_chunk
                    end = ends[i] if ends is not None else len(group_state.block_ids)
                    for j in range(start, end):
                        if (
                            group_state.block_ids[j]
                            in self._current_batch_allocated_block_ids
                        ):
                            group_state.block_ids[j] = 0

    def _build_partial_tail_store_jobs(
        self, scheduler_output: SchedulerOutput
    ) -> dict[int, TransferJob]:
        handoffs = scheduler_output.partial_tail_offloads
        if not self.config.supports_partial_tail or not handoffs:
            return {}

        store_jobs: dict[int, TransferJob] = {}
        for req_id, entries in handoffs.items():
            req_status = self._req_status.get(req_id)
            assert req_status is not None
            assert entries
            boundaries = {boundary for _, _, boundary in entries}
            assert len(boundaries) == 1
            boundary = boundaries.pop()
            req = req_status.req
            max_boundary = min(
                req.num_prompt_tokens,
                req_status.max_offload_tokens or req.num_prompt_tokens,
            )
            assert boundary > 0
            assert boundary % self.config.tokens_per_hash == 0
            assert boundary <= max_boundary

            cow_blocks = {group_idx: block_id for group_idx, block_id, _ in entries}
            assert self._cow_source_groups.issubset(cow_blocks)

            # PATCH (local, 2026-08-30): per-group block index, inert groups
            # skipped. The boundary sits inside ONE block of the copy-on-write
            # (mamba "align") group -- that is what `_partial_tail_block_size`
            # bounds -- but every other group indexes its own block table with
            # its own block size: `cdiv(boundary, tokens_per_block) - 1`. For
            # MLA (3,584) and boundary 46,592 that is block 12, which is also
            # the group's last REGULAR chunk, and `_make_boundary_key` returns
            # that chunk's own key; `prepare_store` then filters it out if the
            # regular store path already put it in the pool, so no byte moves
            # twice. Inert groups (kpool tail, DFlash-2 sliding window) have no
            # block table entry to hand off and still get their zero-sized
            # GPULoadStoreSpec slot below.
            assert boundary % self._partial_tail_block_size != 0
            active_groups = [
                group for group in self.config.kv_group_configs if not group.is_inert
            ]
            block_idx_by_group = {
                group.group_idx: self._boundary_block_idx(group, boundary)
                for group in active_groups
            }
            if any(
                group.group_idx not in self._cow_source_groups
                and block_idx_by_group[group.group_idx]
                >= len(req_status.group_states[group.group_idx].block_ids)
                for group in active_groups
            ):
                continue
            keys = [
                self._make_boundary_key(req, group.group_idx, boundary)
                for group in active_groups
            ]
            block_ids = [
                cow_blocks[group.group_idx]
                if group.group_idx in self._cow_source_groups
                else req_status.group_states[group.group_idx].block_ids[
                    block_idx_by_group[group.group_idx]
                ]
                for group in active_groups
            ]
            assert all(block_id != 0 for block_id in block_ids)

            store_output = self.manager.prepare_store(keys, req_status.req_context)
            if store_output is None:
                self._connector_stats.increase_counter(
                    _ConnectorMetricName.ALLOCATION_FAILURE
                )
                continue
            if not store_output.keys_to_store:
                continue

            for group_config, key in zip(active_groups, keys):
                if key in store_output.keys_to_store:
                    self._events_tracker.record_partial_store(
                        req, group_config, boundary, key
                    )

            group_by_key = {
                key: group.group_idx for key, group in zip(keys, active_groups)
            }
            accepted_groups = [group_by_key[key] for key in store_output.keys_to_store]
            # PATCH (local, 2026-08-30): the worker walks groups in index order
            # and consumes `block_ids` sequentially, while `dst_spec` is built
            # from `keys_to_store` in the manager's order -- so the two only
            # line up if `keys_to_store` preserved the group order we passed in.
            # Upstream relies on that silently; refuse the hand-off instead of
            # writing a mis-paired transfer.
            if accepted_groups != sorted(accepted_groups):
                logger.warning(
                    "Request %s: offloading manager reordered the partial-tail "
                    "store keys (%s); skipping the hand-off.",
                    req_id,
                    accepted_groups,
                )
                continue
            block_id_by_group = dict(
                zip((group.group_idx for group in active_groups), block_ids)
            )
            group_sizes = [0] * len(self.config.kv_group_configs)
            block_indices = [0] * len(self.config.kv_group_configs)
            for group_idx in accepted_groups:
                group_sizes[group_idx] = 1
                block_indices[group_idx] = block_idx_by_group[group_idx]
            source_blocks = [block_id_by_group[group_idx] for group_idx in accepted_groups]

            job_id = self._generate_job_id()
            req_status.transfer_jobs.add(job_id)
            for block_id in source_blocks:
                self._block_id_to_pending_jobs.setdefault(block_id, set()).add(job_id)
            self._jobs[job_id] = TransferJobStatus(
                req_id=req_id,
                pending_count=self.config.num_workers,
                keys=set(store_output.keys_to_store),
                is_store=True,
                fenced_block_ids=source_blocks,
            )
            store_jobs[job_id] = TransferJob(
                req_id=req_id,
                src_spec=GPULoadStoreSpec(
                    source_blocks,
                    group_sizes=group_sizes,
                    block_indices=block_indices,
                ),
                dst_spec=store_output.store_spec,
            )

        return store_jobs

    def _build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[int, TransferJob]:
        blocks_per_chunk = self.config.blocks_per_chunk
        store_jobs: dict[int, TransferJob] = {}
        for req_id in chain(
            scheduler_output.num_scheduled_tokens,
            scheduler_output.finished_req_ids or (),
        ):
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            req = req_status.req

            if req.status is RequestStatus.FINISHED_ABORTED:
                num_tokens_after_batch = req.num_computed_tokens
            elif req.is_finished():
                num_tokens_after_batch = req.num_tokens
            else:
                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
                num_tokens_after_batch = req.num_computed_tokens + num_scheduled_tokens

            num_offloadable_tokens = self._calc_num_offloadable_tokens(
                req_status, num_tokens_after_batch
            )

            # Filter out chunks skipped due to sliding window attention / SSM
            # or unreachable by the load path's alignment constraints.
            new_offload_keys: list[OffloadKey] = []
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                if group_config.is_inert:
                    # PATCH: never stored.
                    continue
                num_chunks = req_status.storable_chunks(
                    group_config, group_state, num_offloadable_tokens
                )

                start_chunk_idx = group_state.next_stored_chunk_idx
                if num_chunks <= start_chunk_idx:
                    continue
                offload_keys = group_state.offload_keys[start_chunk_idx:num_chunks]
                # For each chunk, take the last corresponding GPU block. For
                # blocks_per_chunk=3 and GPU block IDs 1 5 6 7 2 4 9 3 8,
                # this selects GPU blocks 6 4 8.
                # A block_id of 0 means either a sliding window / SSM skip
                # or a stale entry that was zeroed out — skip it either way.
                offload_block_ids = group_state.block_ids[
                    start_chunk_idx * blocks_per_chunk
                    + blocks_per_chunk
                    - 1 : num_chunks * blocks_per_chunk : blocks_per_chunk
                ]
                assert len(offload_keys) == len(offload_block_ids)

                for key_idx, (offload_key, block_id) in enumerate(
                    zip(offload_keys, offload_block_ids)
                ):
                    if block_id == 0:
                        continue
                    # Skip SWA chunks that can never serve a load hit:
                    # within each full-attention alignment segment, only the
                    # trailing chunks queried by _sliding_window_lookup are
                    # reachable. EAGLE/MTP requires one additional chunk that
                    # lookup later drops as its volatile draft tail.
                    abs_chunk_idx = start_chunk_idx + key_idx
                    if not is_store_reachable_swa_chunk(
                        abs_chunk_idx,
                        num_chunks,
                        group_config.alignment_chunk_count,
                        group_config.sliding_window_size_in_chunks,
                        group_config.is_eagle_group,
                    ):
                        continue
                    new_offload_keys.append(offload_key)

            if not new_offload_keys:
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            store_output = self.manager.prepare_store(
                new_offload_keys, req_status.req_context
            )
            if store_output is None:
                self._connector_stats.increase_counter(
                    _ConnectorMetricName.ALLOCATION_FAILURE
                )
                logger.warning("Request %s: cannot store chunks", req_id)
                continue

            if not store_output.keys_to_store:
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            self._touch(req_status)

            keys_to_store = set(store_output.keys_to_store)

            group_sizes: list[int] = []
            block_indices: list[int] = []
            src_block_ids: list[int] = []
            fenced_block_ids: list[int] = []
            deferred_fence_block_ids: list[int] = []
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                if group_config.is_inert:
                    # PATCH: zero-sized entry keeps the worker aligned.
                    group_sizes.append(0)
                    block_indices.append(0)
                    continue
                is_sliding_window = (
                    group_config.sliding_window_size_in_chunks is not None
                )
                num_chunks = req_status.storable_chunks(
                    group_config, group_state, num_offloadable_tokens
                )
                start_chunk_idx = group_state.next_stored_chunk_idx
                block_ids = group_state.block_ids
                num_group_blocks = 0
                start_gpu_block_idx: int | None = None
                for idx, offload_key in enumerate(
                    group_state.offload_keys[start_chunk_idx:num_chunks]
                ):
                    if offload_key not in keys_to_store:
                        continue

                    chunk_idx = start_chunk_idx + idx

                    self._events_tracker.record_store(
                        req, group_config, chunk_idx, offload_key
                    )

                    gpu_block_idx = chunk_idx * blocks_per_chunk
                    for i in range(blocks_per_chunk):
                        block_id = block_ids[gpu_block_idx + i]
                        if block_id == 0:
                            continue
                        if start_gpu_block_idx is None:
                            start_gpu_block_idx = gpu_block_idx + i
                        src_block_ids.append(block_id)
                        num_group_blocks += 1
                        if is_sliding_window:
                            fenced_block_ids.append(block_id)
                        else:
                            deferred_fence_block_ids.append(block_id)

                group_sizes.append(num_group_blocks)
                block_indices.append(start_gpu_block_idx or 0)
                group_state.next_stored_chunk_idx = max(
                    group_state.next_stored_chunk_idx, num_chunks
                )

            src_spec = GPULoadStoreSpec(
                src_block_ids, group_sizes=group_sizes, block_indices=block_indices
            )
            dst_spec = store_output.store_spec

            job_id = self._generate_job_id()
            # a store can only be issued when no load is pending.
            if req_status.transfer_jobs:
                any_jid = next(iter(req_status.transfer_jobs))
                assert self._jobs[any_jid].is_store
            req_status.transfer_jobs.add(job_id)

            # Watch sliding window blocks as they may get evicted
            # before the request finishes
            for bid in fenced_block_ids:
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)

            # the non-sliding window blocks will be watched only
            # when the request finishes
            self._jobs[job_id] = TransferJobStatus(
                req_id=req_id,
                pending_count=self.config.num_workers,
                keys=set(keys_to_store),
                is_store=True,
                deferred_fence_block_ids=deferred_fence_block_ids,
                fenced_block_ids=fenced_block_ids or None,
            )

            store_jobs[job_id] = TransferJob(
                req_id=req_id, src_spec=src_spec, dst_spec=dst_spec
            )

            logger.debug(
                "Request %s offloading %s chunks upto %d tokens (job %d)",
                req_id,
                len(keys_to_store),
                num_offloadable_tokens,
                job_id,
            )

            if req.is_finished():
                # Register non-sliding-window blocks for flush detection.
                for bid in deferred_fence_block_ids:
                    self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)
                    if bid in self._current_batch_allocated_block_ids:
                        self._current_batch_jobs_to_flush.add(job_id)

        return store_jobs

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        self._update_req_states(scheduler_output)
        schedule_end_context = ScheduleEndContext(
            new_req_ids=[req.req_id for req in scheduler_output.scheduled_new_reqs],
            preempted_req_ids=scheduler_output.preempted_req_ids or (),
        )
        self.manager.on_schedule_end(schedule_end_context)

        # Flush jobs for preempted requests.
        for req_id in scheduler_output.preempted_req_ids or ():
            req_status = self._req_status.get(req_id)
            if req_status is None or not req_status.transfer_jobs:
                continue
            any_jid = next(iter(req_status.transfer_jobs))
            assert self._jobs[any_jid].is_store
            self._current_batch_jobs_to_flush.update(req_status.transfer_jobs)

        # Flush jobs that contain re-allocated blocks.
        if (
            self._block_id_to_pending_jobs
            and not self._block_id_to_pending_jobs.keys().isdisjoint(
                self._current_batch_allocated_block_ids
            )
        ):
            self._current_batch_jobs_to_flush.update(
                jid
                for bid in self._current_batch_allocated_block_ids
                if bid in self._block_id_to_pending_jobs
                for jid in self._block_id_to_pending_jobs[bid]
            )

        partial_store_jobs = self._build_partial_tail_store_jobs(scheduler_output)
        normal_store_jobs = self._build_store_jobs(scheduler_output)
        meta = OffloadingConnectorMetadata(
            load_jobs=self._current_batch_load_jobs,
            store_jobs=partial_store_jobs | normal_store_jobs,
            jobs_to_flush=self._current_batch_jobs_to_flush,
        )

        # All prepare_store calls for finished requests have been issued.
        # Signal on_request_finished and clean up state where possible.
        for req_id in scheduler_output.finished_req_ids or ():
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            req_status.finished_signaled = True
            self.manager.on_request_finished(req_status.req_context)
            if not req_status.transfer_jobs:
                del self._req_status[req_id]
        self._current_batch_load_jobs = {}
        self._current_batch_jobs_to_flush = set()
        self._current_batch_allocated_block_ids = set()
        return meta

    def has_pending_push_work(self) -> bool:
        """Whether the engine must keep stepping.

        While True, build_connector_meta() and update_connector_output()
        continue to be called even when no requests are scheduled.
        """
        return bool(self._jobs) or self.manager.has_pending_work()

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, OffloadingWorkerMetadata):
            assert meta is None
            meta = OffloadingWorkerMetadata()
        if not meta.transfer_stats.is_empty():
            transfer_stats = OffloadingConnectorStats()
            if not meta.transfer_stats.load.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_BYTES,
                    meta.transfer_stats.load.bytes,
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_TIME,
                    meta.transfer_stats.load.time,
                )
                for size in meta.transfer_stats.load.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.LOAD_SIZE, size
                    )
            if not meta.transfer_stats.store.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_BYTES,
                    meta.transfer_stats.store.bytes,
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_TIME,
                    meta.transfer_stats.store.time,
                )
                for size in meta.transfer_stats.store.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.STORE_SIZE, size
                    )
            self._connector_stats.aggregate(transfer_stats)

        for job_id, count in meta.completed_jobs.items():
            assert count > 0
            if job_id < self._stale_job_threshold:
                logger.debug(
                    "Skipping stale completed job %d (pre-reset counter: %d)",
                    job_id,
                    self._stale_job_threshold,
                )
                continue
            job_status = self._jobs[job_id]
            job_status.pending_count -= count
            if job_status.pending_count > 0:
                continue
            assert job_status.pending_count == 0

            req_status = self._req_status[job_status.req_id]
            if job_status.is_store:
                self.manager.complete_store(job_status.keys, req_status.req_context)
            else:
                self.manager.complete_load(job_status.keys, req_status.req_context)
                if self._chunks_being_loaded:
                    self._chunks_being_loaded.difference_update(job_status.keys)
            if self._block_id_to_pending_jobs:
                # Sliding window blocks are tracked from store creation
                # and must be cleaned up unconditionally.
                self._remove_pending_job(job_id, job_status.fenced_block_ids)
                # Non-sliding-window blocks are only tracked after
                # request_finished, so only clean up for finished requests.
                if req_status.req.is_finished():
                    self._remove_pending_job(
                        job_id, job_status.deferred_fence_block_ids
                    )

            del self._jobs[job_id]
            req_status.transfer_jobs.remove(job_id)
            if req_status.finished_signaled and not req_status.transfer_jobs:
                del self._req_status[job_status.req_id]

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats: OffloadingConnectorStats | None = None
        if not self._connector_stats.is_empty():
            stats = self._connector_stats
            self._connector_stats = OffloadingConnectorStats()

        manager_stats = self.manager.get_stats()
        if manager_stats is not None:
            if stats is None:
                stats = manager_stats
            else:
                stats.aggregate(manager_stats)

        return stats

    def request_finished(
        self,
        request: Request,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        req_status = self._req_status.get(request.request_id)

        if req_status is None:
            # Untracked request (offloading never started): no in-flight jobs,
            # nothing was deferred, so finalize immediately.
            req_context = _create_req_context(request)
            self.manager.on_new_request(req_context)
            self.manager.on_request_finished(req_context)
            return False, None

        self._maybe_observe_lookup_async_delay(req_status)

        # Update offload keys with final block hash so _build_store_jobs can
        # create store jobs for the last block(s) on the next schedule step.
        req_status.update_offload_keys()

        # Keep req_status alive: _build_store_jobs will process finished_req_ids
        # on the next step and handle cleanup after creating store jobs.
        # Register deferred fences so future block reuse triggers a flush via
        # _block_id_to_pending_jobs.
        for job_id in req_status.transfer_jobs:
            job_status = self._jobs[job_id]
            for bid in job_status.deferred_fence_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)

        return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Drain pending KV cache events.

        Complete metadata is available only when self-describing KV events
        are enabled, and only for full-attention groups. Other shapes retain
        the previous placeholder payload so consumers can ignore them.

        Yields:
            ``BlockStored`` or ``BlockRemoved`` events corresponding to
            the underlying :class:`OffloadingEvent` stream.
        """
        yield from self._events_tracker.take_events(self.manager.take_events())

    def reset_cache(self) -> None:
        """Reset the offloading manager cache, evicting all stored chunks."""

        # reset_cache cannot be called in the middle of a schedule step
        assert not self._current_batch_load_jobs
        assert not self._current_batch_jobs_to_flush
        assert not self._current_batch_allocated_block_ids

        # Flush all in-flight jobs
        self._current_batch_jobs_to_flush.update(self._jobs.keys())

        for req_id, status in list(self._req_status.items()):
            if status.req.is_finished():
                if not status.finished_signaled:
                    self.manager.on_request_finished(status.req_context)
                del self._req_status[req_id]

        # Reset offloading manager cache
        self.manager.reset_cache()

        # Reset store progress so active requests re-offload from chunk 0.
        for status in self._req_status.values():
            for group_state in status.group_states:
                group_state.next_stored_chunk_idx = 0
            status.transfer_jobs.clear()
            status.partial_tail_boundary = None

        # Discard jobs and save job_counter to be able to discard worker responses
        self._stale_job_threshold = self._job_counter
        self._jobs.clear()
        self._block_id_to_pending_jobs.clear()

        # The manager pool is empty; pending event payloads and announced
        # reference counts are stale.
        self._events_tracker.reset()

        # Note: _current_batch_jobs_to_flush is intentionally NOT cleared.
        # The load flush IDs collected above must be delivered to workers.
        if self._chunks_being_loaded is not None:
            self._chunks_being_loaded.clear()

    def shutdown(self) -> None:
        self.manager.shutdown()
