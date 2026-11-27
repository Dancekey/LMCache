# SPDX-License-Identifier: Apache-2.0
"""NUMA-sharded pinned-DRAM allocator for LMCache L1."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
import sys
import threading
from typing import Any, List, Optional, Union

import torch

from lmcache import device_ops, torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import get_size_bytes
from lmcache.v1.memory_allocators.lazy_memory_allocator import (
    AddressManager,
    LazyMemoryAllocator,
    TensorMemoryAllocator,
)
from lmcache.v1.memory_allocators.numa_shard_planner import (
    NumaNodeInfo,
    ShardPlan,
    discover_process_numa_nodes,
    plan_numa_shards,
    read_numa_memfree_bytes,
)
from lmcache.v1.memory_management import MemoryFormat, MemoryObj

logger = init_logger(__name__)


@dataclass
class _PinnedAllocation:
    shard_idx: int
    numa_node: int
    ptr: int
    owner_dev: int
    owner_ctx: object | None
    released: bool = False


@dataclass
class _AllocatedShard:
    allocation: _PinnedAllocation
    size_bytes: int
    buffer: torch.Tensor
    allocator: TensorMemoryAllocator


_UNRELEASED_INIT_LOCK = threading.Lock()
_UNRELEASED_INIT_SHARDS: list[tuple[_PinnedAllocation, Callable[[int], Any]]] = []


def retry_unreleased_init_shards() -> int:
    """Retry releasing pinned shards stranded by a constructor rollback failure.

    Returns:
        Number of shards that still remain unreleased after this retry pass.
    """
    with _UNRELEASED_INIT_LOCK:
        pending = list(_UNRELEASED_INIT_SHARDS)
        _UNRELEASED_INIT_SHARDS.clear()

    still_unreleased: list[tuple[_PinnedAllocation, Callable[[int], Any]]] = []
    for allocation, free_fn in pending:
        if allocation.released:
            continue
        try:
            _safe_free_pinned_shard(allocation, free_fn)
        except Exception:
            logger.error(
                "Retry free_pinned_ptr failed for stranded init shard %d "
                "(node %d, ptr=0x%x)",
                allocation.shard_idx,
                allocation.numa_node,
                allocation.ptr,
                exc_info=True,
            )
        if not allocation.released:
            still_unreleased.append((allocation, free_fn))

    with _UNRELEASED_INIT_LOCK:
        _UNRELEASED_INIT_SHARDS.extend(still_unreleased)
        return len(_UNRELEASED_INIT_SHARDS)


def _npu_is_available() -> bool:
    npu_mod = getattr(torch, "npu", None)
    if npu_mod is None:
        return False
    try:
        return bool(npu_mod.is_available())
    except Exception:
        return False


def _capture_owner_context() -> tuple[int, object | None]:
    """Capture owning accelerator device/context when required for teardown."""
    if not _npu_is_available():
        return 0, None
    import acl  # type: ignore[import-not-found]

    dev = int(torch.npu.current_device())
    ctx, rc = acl.rt.get_context()
    if rc != 0 or not ctx:
        torch.npu.set_device(dev)
        ctx, rc = acl.rt.get_context()
    if rc != 0 or not ctx:
        raise RuntimeError(f"Cannot capture pinned allocation ACL context: {rc}")
    return dev, ctx


def _resolve_pinned_alloc_free() -> tuple[Callable[[int, int], int], Callable[[int], Any]]:
    """Resolve backend pinned-memory alloc/free primitives."""
    if torch_device_type == "npu":
        import lmcache_ascend.c_ops as ascend_c_ops  # type: ignore[import-not-found]

        alloc_fn = ascend_c_ops.alloc_pinned_ptr
        free_fn = getattr(ascend_c_ops, "free_pinned_ptr_checked", None)
        if free_fn is None:
            raise RuntimeError(
                "LMCache-Ascend c_ops must be rebuilt with "
                "free_pinned_ptr_checked before using L1 NUMA sharding"
            )
        return alloc_fn, free_fn

    return device_ops.alloc_pinned_ptr, device_ops.free_pinned_ptr


def _safe_free_pinned_shard(
    shard: _PinnedAllocation,
    free_fn: Callable[[int], Any],
    *,
    npu_module: Any = None,
    acl_module: Any = None,
) -> None:
    """Release one pinned shard under its original owner context when tracked.

    When an ACL owner context is tracked, synchronizes the owner device
    *before* restoring ``owner_ctx`` because ``torch.npu.synchronize(dev)``
    uses ``c10::NPUGuard``, which resets the thread ACL context upon exit.
    """
    if shard.released:
        return
    if shard.owner_ctx is None:
        free_fn(shard.ptr)
        shard.released = True
        return

    npu_mod = npu_module if npu_module is not None else torch.npu
    if acl_module is None:
        import acl as acl_module  # type: ignore[import-not-found]

    prev_ctx, prev_rc = acl_module.rt.get_context()
    prev_dev = int(npu_mod.current_device())
    try:
        npu_mod.synchronize(shard.owner_dev)
        npu_mod.set_device(shard.owner_dev)
        rc = acl_module.rt.set_context(shard.owner_ctx)
        if rc != 0:
            raise RuntimeError(
                f"Cannot select ACL owner context for shard {shard.shard_idx}: {rc}"
            )
        free_fn(shard.ptr)
        shard.released = True
    finally:
        release_failed = sys.exc_info()[0] is not None
        try:
            npu_mod.set_device(prev_dev)
            if prev_rc == 0 and prev_ctx:
                rc = acl_module.rt.set_context(prev_ctx)
                if rc != 0:
                    raise RuntimeError(f"Cannot restore caller ACL context: {rc}")
        except Exception:
            if not release_failed:
                raise
            logger.exception(
                "Could not restore caller ACL context after a release failure"
            )


class NumaShardedPinnedAllocator(LazyMemoryAllocator):
    """Opt-in multi-shard pinned host allocator spread across NUMA nodes.

    Subclasses :class:`LazyMemoryAllocator` so ``isinstance(obj.parent(),
    LazyMemoryAllocator)`` remains ``True`` on async DMA transfer paths
    (``gpu_ops.py``), while ``exposes_single_buffer = False`` signals
    :meth:`L1MemoryManager.get_l1_memory_desc` to return ``None``.
    """

    exposes_single_buffer: bool = False

    def __init__(
        self,
        size_in_bytes: int,
        align_bytes: int = AddressManager.ALIGN_BYTES,
        numa_policy: str = "auto",
        numa_nodes: Sequence[int] | None = None,
        numa_headroom_bytes: int = 32 << 30,
        *,
        alloc_fn: Callable[[int, int], int] | None = None,
        free_fn: Callable[[int], Any] | None = None,
        topo_nodes: Sequence[NumaNodeInfo] | None = None,
        memfree_reader: Callable[[Sequence[int]], dict[int, int]] | None = None,
        affinity_getter: Callable[[], frozenset[int]] | None = None,
        affinity_setter: Callable[[frozenset[int]], None] | None = None,
        sysfs_root: Path | None = None,
    ) -> None:
        # Intentionally bypass LazyMemoryAllocator.__init__ (single large arena).
        if align_bytes <= 0 or (align_bytes & (align_bytes - 1)) != 0:
            raise ValueError("align_bytes must be a positive power of two")

        if alloc_fn is None or free_fn is None:
            resolved_alloc, resolved_free = _resolve_pinned_alloc_free()
            alloc_fn = alloc_fn or resolved_alloc
            free_fn = free_fn or resolved_free

        self._alloc_fn = alloc_fn
        self._free_fn = free_fn
        self._align_bytes = align_bytes
        self._numa_policy = numa_policy
        self._lock = threading.Lock()
        self._rr_cursor = 0
        self._closed = False
        self._shards: list[_AllocatedShard] = []

        if memfree_reader is None:
            if sysfs_root is not None:
                memfree_reader = lambda nids: read_numa_memfree_bytes(
                    nids, sysfs_root=sysfs_root
                )
            else:
                memfree_reader = read_numa_memfree_bytes

        if topo_nodes is None:
            discover_kwargs: dict[str, Any] = {}
            if sysfs_root is not None:
                discover_kwargs["sysfs_root"] = sysfs_root
            nodes = discover_process_numa_nodes(numa_nodes, **discover_kwargs)
        else:
            nodes = list(topo_nodes)

        all_node_ids = [n.node_id for n in nodes]
        memfree_before = memfree_reader(all_node_ids)

        plans: list[ShardPlan] = plan_numa_shards(
            total_bytes=size_in_bytes,
            align_bytes=align_bytes,
            policy=numa_policy,
            nodes=nodes,
            memfree_by_node=memfree_before,
            headroom_bytes=numa_headroom_bytes,
        )
        self._plans = list(plans)
        self._final_size = sum(p.size_bytes for p in plans)
        self._curr_size = self._final_size

        if affinity_getter is None and hasattr(os, "sched_getaffinity"):
            affinity_getter = lambda: frozenset(os.sched_getaffinity(0))
        if affinity_setter is None and hasattr(os, "sched_setaffinity"):
            affinity_setter = lambda cpus: os.sched_setaffinity(0, cpus)

        orig_affinity = affinity_getter() if affinity_getter is not None else None
        owner_dev, owner_ctx = _capture_owner_context()

        acquired: list[_PinnedAllocation] = []
        try:
            try:
                for shard_idx, plan in enumerate(plans):
                    mem_pre_node = memfree_reader([plan.node_id]).get(plan.node_id, -1)
                    if affinity_setter is not None and plan.cpus:
                        affinity_setter(plan.cpus)

                    ptr = int(self._alloc_fn(plan.size_bytes, 0))
                    if ptr == 0:
                        raise RuntimeError(
                            "Pinned memory allocation returned a null pointer"
                        )
                    allocation = _PinnedAllocation(
                        shard_idx=shard_idx,
                        numa_node=plan.node_id,
                        ptr=ptr,
                        owner_dev=owner_dev,
                        owner_ctx=owner_ctx,
                    )
                    acquired.append(allocation)
                    if ptr % align_bytes != 0:
                        raise RuntimeError(
                            f"Pinned shard base 0x{ptr:x} on NUMA node {plan.node_id} "
                            f"is not aligned to align_bytes={align_bytes} "
                            f"(remainder {ptr % align_bytes})"
                        )
                    arr_type = ctypes.c_uint8 * plan.size_bytes
                    buf = torch.frombuffer(arr_type.from_address(ptr), dtype=torch.uint8)
                    sub_alloc = TensorMemoryAllocator(
                        tensor=buf,
                        align_bytes=align_bytes,
                        init_address_space=plan.size_bytes,
                    )
                    self._shards.append(
                        _AllocatedShard(allocation, plan.size_bytes, buf, sub_alloc)
                    )
                    mem_post_node = memfree_reader([plan.node_id]).get(plan.node_id, -1)
                    logger.info(
                        "L1 NUMA shard[%d/%d] allocated: node=%d, size=%.2f GiB, "
                        "ptr=0x%x, node_MemFree=%.2f -> %.2f GiB (delta=%.2f GiB)",
                        shard_idx + 1,
                        len(plans),
                        plan.node_id,
                        plan.size_bytes / (1 << 30),
                        ptr,
                        mem_pre_node / (1 << 30) if mem_pre_node >= 0 else -1.0,
                        mem_post_node / (1 << 30) if mem_post_node >= 0 else -1.0,
                        (mem_pre_node - mem_post_node) / (1 << 30)
                        if (mem_pre_node >= 0 and mem_post_node >= 0)
                        else 0.0,
                    )
            finally:
                if orig_affinity is not None and affinity_setter is not None:
                    affinity_setter(orig_affinity)
            memfree_after = memfree_reader(all_node_ids)
        except BaseException as init_exc:
            logger.error(
                "L1 NUMA shard allocation failed after %d/%d shards; "
                "rolling back allocated shards in reverse order.",
                len(acquired),
                len(plans),
                exc_info=True,
            )
            for allocated in reversed(acquired):
                try:
                    _safe_free_pinned_shard(allocated, self._free_fn)
                except Exception:
                    logger.error(
                        "Rollback free_pinned_ptr failed for shard %d (node %d)",
                        allocated.shard_idx,
                        allocated.numa_node,
                        exc_info=True,
                    )
            unreleased = [a for a in acquired if not a.released]
            if unreleased:
                setattr(init_exc, "_unreleased_numa_shards", tuple(unreleased))
                with _UNRELEASED_INIT_LOCK:
                    for stranded in unreleased:
                        _UNRELEASED_INIT_SHARDS.append((stranded, self._free_fn))
                logger.error(
                    "Quarantined %d unreleased NUMA shard(s) after rollback failure; "
                    "call retry_unreleased_init_shards() to retry cleanup.",
                    len(unreleased),
                )
            self._shards.clear()
            raise
        summary = ", ".join(
            f"node{nid}:{memfree_before.get(nid, 0) / (1 << 30):.1f}->"
            f"{memfree_after.get(nid, 0) / (1 << 30):.1f}GiB"
            for nid in all_node_ids
        )
        logger.info(
            "Initialized NumaShardedPinnedAllocator: policy=%s, shards=%d, "
            "total=%.2f GiB, MemFree summary=[%s]",
            numa_policy,
            len(self._shards),
            self._final_size / (1 << 30),
            summary,
        )

    @property
    def shard_plans(self) -> tuple[ShardPlan, ...]:
        """Return immutable shard plans realized by this allocator."""
        return tuple(self._plans)

    @property
    def num_shards(self) -> int:
        """Return the number of active NUMA shards."""
        return len(self._shards)

    def _tag_obj(self, obj: MemoryObj, shard_idx: int) -> MemoryObj:
        obj.parent_allocator = self
        setattr(obj, "_numa_shard_idx", shard_idx)
        return obj

    def _locate_shard_idx(self, memory_obj: MemoryObj) -> int:
        idx = getattr(memory_obj, "_numa_shard_idx", None)
        if isinstance(idx, int) and 0 <= idx < len(self._shards):
            return idx
        ptr = memory_obj.data_ptr
        for shard in self._shards:
            base = shard.allocation.ptr
            if base <= ptr < base + shard.size_bytes:
                return shard.allocation.shard_idx
        raise ValueError(
            f"MemoryObj ptr=0x{ptr:x} does not belong to any shard of "
            "NumaShardedPinnedAllocator"
        )

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """Allocate one contiguous object strictly inside a single NUMA shard."""
        with self._lock:
            n = len(self._shards)
            if self._closed or n == 0:
                return None
            start = self._rr_cursor
            for offset in range(n):
                idx = (start + offset) % n
                obj = self._shards[idx].allocator.allocate(
                    shapes, dtypes, fmt, allocator_type
                )
                if obj is not None:
                    self._rr_cursor = (idx + 1) % n
                    return self._tag_obj(obj, idx)
            return None

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        """Allocate ``batch_size`` objects, allowing batches to span shards atomically.

        Each individual :class:`MemoryObj` remains contiguous within one shard.
        If total free capacity across all shards cannot satisfy ``batch_size``,
        all tentatively allocated objects are freed back before returning ``None``.
        """
        if batch_size <= 0:
            return []

        with self._lock:
            n = len(self._shards)
            if self._closed or n == 0:
                return None

            norm_shapes, norm_dtypes = self._shards[
                0
            ].allocator._adapt_shapes_and_dtypes(shapes, dtypes)
            unit_raw_size = get_size_bytes(norm_shapes, norm_dtypes)
            unit_aligned_size = self._shards[
                0
            ].allocator.address_manager.compute_aligned_size(unit_raw_size)
            start = self._rr_cursor

            # Fast path: fulfill the whole batch from a single shard.
            for offset in range(n):
                idx = (start + offset) % n
                shard = self._shards[idx]
                if (
                    shard.allocator.address_manager.get_free_size()
                    < unit_aligned_size * batch_size
                ):
                    continue
                objs = shard.allocator.batched_allocate(
                    shapes, dtypes, batch_size, fmt, allocator_type
                )
                if objs is not None:
                    self._rr_cursor = (idx + 1) % n
                    return [self._tag_obj(o, idx) for o in objs]

            # Cross-shard path: gather pieces across shards, rollback on OOM.
            collected: list[tuple[int, MemoryObj]] = []
            remaining = batch_size
            for offset in range(n):
                if remaining == 0:
                    break
                idx = (start + offset) % n
                shard = self._shards[idx]
                max_fit = min(
                    remaining,
                    shard.allocator.address_manager.get_free_size()
                    // unit_aligned_size,
                )
                if max_fit <= 0:
                    continue

                take = max_fit
                while remaining > 0 and take > 0:
                    objs = shard.allocator.batched_allocate(
                        shapes, dtypes, take, fmt, allocator_type
                    )
                    if objs is not None:
                        for obj in objs:
                            collected.append((idx, self._tag_obj(obj, idx)))
                        remaining -= len(objs)
                        max_next = (
                            shard.allocator.address_manager.get_free_size()
                            // unit_aligned_size
                        )
                        take = min(remaining, max_next, take)
                    else:
                        take //= 2

            if remaining > 0:
                by_shard: dict[int, list[MemoryObj]] = {}
                for idx, obj in collected:
                    by_shard.setdefault(idx, []).append(obj)
                for idx, group in by_shard.items():
                    self._shards[idx].allocator.batched_free(
                        group, allocator_type, update_stats=True
                    )
                return None

            self._rr_cursor = (start + 1) % n
            return [obj for _, obj in collected]

    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ) -> None:
        """Free a single memory object back to its owning NUMA shard."""
        if not memory_obj.is_valid():
            return
        with self._lock:
            if self._closed or not self._shards:
                return
            idx = self._locate_shard_idx(memory_obj)
            self._shards[idx].allocator.free(memory_obj, allocator_type)

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        """Group memory objects by owning NUMA shard and free each batch."""
        if not memory_objs:
            return
        with self._lock:
            if self._closed or not self._shards:
                return
            by_shard: dict[int, list[MemoryObj]] = {}
            for obj in memory_objs:
                if not obj.is_valid():
                    continue
                idx = self._locate_shard_idx(obj)
                by_shard.setdefault(idx, []).append(obj)
            for idx, group in by_shard.items():
                self._shards[idx].allocator.batched_free(
                    group, allocator_type, update_stats=update_stats
                )

    def get_memory_usage(self) -> tuple[int, int]:
        """Return ``(used_bytes, total_bytes)`` summed across all NUMA shards."""
        with self._lock:
            total = sum(
                s.allocator.address_manager.get_heap_size() for s in self._shards
            )
            free = sum(
                s.allocator.address_manager.get_free_size() for s in self._shards
            )
            return total - free, total

    def memcheck(self) -> bool:
        """Verify internal address-manager consistency across all NUMA shards."""
        with self._lock:
            return all(s.allocator.memcheck() for s in self._shards)

    def get_underlying_buffer(self) -> torch.Tensor:
        raise RuntimeError(
            "NumaShardedPinnedAllocator manages multiple NUMA shards and exposes "
            "no single contiguous underlying buffer."
        )

    def get_address_manager(self) -> AddressManager:
        raise RuntimeError(
            "NumaShardedPinnedAllocator manages multiple NUMA shards and exposes "
            "no single AddressManager."
        )

    def close(self) -> None:
        """Release shards in reverse order and retain failed frees for retry."""
        with self._lock:
            self._closed = True
            first_exc: BaseException | None = None
            for shard in reversed(self._shards):
                allocation = shard.allocation
                try:
                    _safe_free_pinned_shard(allocation, self._free_fn)
                except BaseException as exc:
                    logger.exception(
                        "Failed to close NUMA shard %d (node %d, ptr=0x%x)",
                        allocation.shard_idx,
                        allocation.numa_node,
                        allocation.ptr,
                    )
                    if first_exc is None:
                        first_exc = exc
            self._shards = [s for s in self._shards if not s.allocation.released]
            if first_exc is not None:
                raise first_exc
