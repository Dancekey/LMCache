# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for NUMA shard planning and NumaShardedPinnedAllocator."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
    get_arg_parser,
    l1_exposes_single_memory_region,
    parse_args_to_config,
    validate_l1_numa_sharding_compatibility,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.memory_manager.l1_memory_manager import L1MemoryManager
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_allocators.numa_shard_planner import (
    NumaNodeInfo,
    discover_process_numa_nodes,
    plan_numa_shards,
    read_numa_memfree_bytes,
)
from lmcache.v1.memory_allocators.numa_sharded_pinned_allocator import (
    NumaShardedPinnedAllocator,
    _PinnedAllocation,
    _safe_free_pinned_shard,
    retry_unreleased_init_shards,
)
from lmcache.v1.memory_management import MemoryFormat

GiB = 1 << 30
MiB = 1 << 20


class _FakePinnedPool:
    """In-process ctypes backing store mimicking alloc_pinned_ptr/free_pinned_ptr."""

    _BASE_ALIGN = 65536

    def __init__(
        self,
        fail_on_call: int | None = None,
        misalign_by: int = 0,
    ) -> None:
        self._fail_on_call = fail_on_call
        self._misalign_by = misalign_by
        self._call_count = 0
        self._buffers: dict[int, ctypes.Array] = {}
        self.allocated_ptrs: list[int] = []
        self.freed_ptrs: list[int] = []

    def alloc(self, size: int, flags: int = 0) -> int:
        self._call_count += 1
        if self._fail_on_call is not None and self._call_count == self._fail_on_call:
            raise RuntimeError("pinned host allocation failed")
        pad = self._BASE_ALIGN + abs(self._misalign_by)
        buf = (ctypes.c_uint8 * (size + pad))()
        raw_ptr = ctypes.addressof(buf)
        aligned_ptr = (raw_ptr + (self._BASE_ALIGN - 1)) & ~(self._BASE_ALIGN - 1)
        ptr = aligned_ptr + self._misalign_by
        self._buffers[ptr] = buf
        self.allocated_ptrs.append(ptr)
        return ptr

    def free(self, ptr: int) -> None:
        self.freed_ptrs.append(ptr)
        self._buffers.pop(ptr, None)


def _make_nodes(count: int = 8, cpus_per_node: int = 24) -> list[NumaNodeInfo]:
    return [
        NumaNodeInfo(
            node_id=i,
            cpus=frozenset(range(i * cpus_per_node, (i + 1) * cpus_per_node)),
            mem_total_bytes=256 * GiB,
        )
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# 1. Topology discovery & sysfs parsing
# ---------------------------------------------------------------------------


def test_discover_process_numa_nodes_respects_cpuset_and_mems(tmp_path: Path):
    (tmp_path / "online").write_text("0-3\n", encoding="utf-8")
    for nid, (cpulist, mem_kb, free_kb) in {
        0: ("0-7", 256 * 1024 * 1024, 180 * 1024 * 1024),
        1: ("8-15", 256 * 1024 * 1024, 170 * 1024 * 1024),
        2: ("16-23", 0, 0),  # memory-less node -> excluded
        3: ("24-31", 256 * 1024 * 1024, 160 * 1024 * 1024),
    }.items():
        ndir = tmp_path / f"node{nid}"
        ndir.mkdir()
        (ndir / "cpulist").write_text(f"{cpulist}\n", encoding="utf-8")
        (ndir / "meminfo").write_text(
            f"Node {nid} MemTotal: {mem_kb} kB\nNode {nid} MemFree: {free_kb} kB\n",
            encoding="utf-8",
        )

    nodes = discover_process_numa_nodes(
        sysfs_root=tmp_path, allowed_cpus=frozenset(range(0, 16))
    )
    assert [n.node_id for n in nodes] == [0, 1]
    assert nodes[0].cpus == frozenset(range(0, 8))

    status = tmp_path / "status"
    status.write_text("Mems_allowed_list:\t1,3\n", encoding="utf-8")
    nodes = discover_process_numa_nodes(
        sysfs_root=tmp_path,
        allowed_cpus=frozenset(range(16)),
        status_path=status,
    )
    assert [node.node_id for node in nodes] == [1]

    with pytest.raises(ValueError, match="No online NUMA node"):
        discover_process_numa_nodes(
            sysfs_root=tmp_path,
            allowed_cpus=frozenset(range(8)),
            status_path=status,
        )

    memfree = read_numa_memfree_bytes([0, 1, 3], sysfs_root=tmp_path)
    assert memfree[0] == 180 * GiB
    assert memfree[3] == 160 * GiB


# ---------------------------------------------------------------------------
# 2. Parameter-free MemFree water-filling planner
# ---------------------------------------------------------------------------


def test_plan_balanced_nodes_produces_exact_even_split():
    nodes = _make_nodes(8)
    memfree = {i: 180 * GiB for i in range(8)}
    plans = plan_numa_shards(
        total_bytes=480 * GiB,
        align_bytes=4096,
        nodes=nodes,
        memfree_by_node=memfree,
    )
    assert len(plans) == 8
    assert [p.size_bytes for p in plans] == [60 * GiB] * 8
    assert sum(p.size_bytes for p in plans) == 480 * GiB


def test_plan_skewed_nodes_equalizes_post_alloc_memfree_and_avoids_tight_nodes():
    nodes = _make_nodes(8)
    memfree = {
        0: 184 * GiB,
        1: 145 * GiB,
        2: 8 * GiB,
        3: 12 * GiB,
        4: 173 * GiB,
        5: 147 * GiB,
        6: 14 * GiB,
        7: 9 * GiB,
    }
    plans = plan_numa_shards(
        total_bytes=480 * GiB,
        align_bytes=4096,
        nodes=nodes,
        memfree_by_node=memfree,
    )
    by_node = {p.node_id: p.size_bytes for p in plans}
    assert sum(by_node.values()) == 480 * GiB
    for tight in (2, 3, 6, 7):
        assert tight not in by_node
    post_free = [memfree[n] - by_node[n] for n in (0, 1, 4, 5)]
    assert max(post_free) - min(post_free) <= 65536
    assert min(post_free) >= 42 * GiB
    for n in (0, 1, 4, 5):
        assert by_node[n] % 65536 == 0


def test_water_fill_keeps_aligned_remainder_within_node_cap():
    block = 65536
    nodes = _make_nodes(4)
    plan = plan_numa_shards(
        10 * block, block, nodes, dict.fromkeys(range(4), 3 * block)
    )
    assert [shard.size_bytes // block for shard in plan] == [3, 3, 2, 2]
    assert all(shard.size_bytes <= 3 * block for shard in plan)


def test_water_fill_drops_below_floor_nodes_and_equalizes_spacious_nodes():
    block = 65536
    nodes = _make_nodes(5)
    memfree = {0: 1 * block, **{i: 4 * block for i in range(1, 5)}}
    plan = plan_numa_shards(12 * block, block, nodes, memfree)
    assert [(s.node_id, s.size_bytes // block) for s in plan] == [
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 3),
    ]
    assert all(s.size_bytes <= memfree[s.node_id] for s in plan)


def test_plan_low_total_memfree_falls_back_to_even_split():
    nodes = _make_nodes(8)
    memfree_low = {i: 30 * GiB for i in range(8)}
    plans = plan_numa_shards(
        total_bytes=480 * GiB,
        align_bytes=4096,
        nodes=nodes,
        memfree_by_node=memfree_low,
    )
    assert sum(p.size_bytes for p in plans) == 480 * GiB
    assert [p.size_bytes for p in plans] == [60 * GiB] * 8


# ---------------------------------------------------------------------------
# 3. Allocator: single-obj contiguity, cross-shard batch, OOM atomic rollback
# ---------------------------------------------------------------------------


def test_sharded_allocator_cross_shard_batch_data_integrity_and_oom_rollback():
    pool = _FakePinnedPool()
    nodes = _make_nodes(4, cpus_per_node=4)
    memfree = {i: 128 * GiB for i in range(4)}
    affinity_log: list[frozenset[int]] = []

    alloc = NumaShardedPinnedAllocator(
        size_in_bytes=16 * MiB,
        align_bytes=4096,
        topo_nodes=nodes,
        alloc_fn=pool.alloc,
        free_fn=pool.free,
        memfree_reader=lambda nids: {n: memfree[n] for n in nids},
        affinity_getter=lambda: frozenset(range(16)),
        affinity_setter=lambda cpus: affinity_log.append(frozenset(cpus)),
    )
    try:
        assert isinstance(alloc, LazyMemoryAllocator)
        assert alloc.exposes_single_buffer is False
        assert alloc.num_shards == 4
        assert affinity_log[:4] == [n.cpus for n in nodes]
        assert affinity_log[-1] == frozenset(range(16))

        shape = torch.Size([1024, 1024])
        dtype = torch.uint8

        batch = alloc.batched_allocate(
            shape, dtype, batch_size=10, fmt=MemoryFormat.KV_2LTD
        )
        assert batch is not None
        assert len(batch) == 10

        shard_indices = set()
        for idx, obj in enumerate(batch):
            assert isinstance(obj.parent(), LazyMemoryAllocator)
            assert obj.parent() is alloc
            s_idx = getattr(obj, "_numa_shard_idx")
            shard_indices.add(s_idx)
            base = pool.allocated_ptrs[s_idx]
            size = alloc.shard_plans[s_idx].size_bytes
            assert base <= obj.data_ptr
            assert obj.data_ptr % 4096 == 0
            assert obj.data_ptr + obj.get_size() <= base + size
            obj.tensor.fill_(idx + 1)

        assert len(shard_indices) >= 3
        for idx, obj in enumerate(batch):
            assert int(obj.tensor[0, 0].item()) == idx + 1
            assert int(obj.tensor[-1, -1].item()) == idx + 1

        used_before, total = alloc.get_memory_usage()
        assert total == 16 * MiB
        assert used_before == 10 * MiB

        oom = alloc.batched_allocate(
            shape, dtype, batch_size=7, fmt=MemoryFormat.KV_2LTD
        )
        assert oom is None
        assert alloc.get_memory_usage() == (used_before, total)
        assert alloc.memcheck() is True

        rem = alloc.batched_allocate(
            shape, dtype, batch_size=6, fmt=MemoryFormat.KV_2LTD
        )
        assert rem is not None and len(rem) == 6
        assert alloc.get_memory_usage() == (16 * MiB, 16 * MiB)

        alloc.batched_free(batch)
        for obj in rem:
            alloc.free(obj)
        assert alloc.get_memory_usage() == (0, 16 * MiB)
        assert alloc.memcheck() is True

        with pytest.raises(RuntimeError, match="multiple NUMA shards"):
            alloc.get_underlying_buffer()
        with pytest.raises(RuntimeError, match="multiple NUMA shards"):
            alloc.get_address_manager()
    finally:
        alloc.close()

    assert pool.freed_ptrs == list(reversed(pool.allocated_ptrs))
    alloc.close()
    assert len(pool.freed_ptrs) == 4


# ---------------------------------------------------------------------------
# 4. Partial-init rollback, alignment check, double-fault quarantine, close retry
# ---------------------------------------------------------------------------


def test_sharded_allocator_init_failure_rolls_back_reverse_and_restores_affinity():
    pool = _FakePinnedPool(fail_on_call=3)
    nodes = _make_nodes(4, cpus_per_node=4)
    affinity_log: list[frozenset[int]] = []

    with pytest.raises(RuntimeError, match="pinned host allocation failed"):
        NumaShardedPinnedAllocator(
            size_in_bytes=16 * MiB,
            align_bytes=4096,
            topo_nodes=nodes,
            alloc_fn=pool.alloc,
            free_fn=pool.free,
            memfree_reader=lambda nids: {n: 128 * GiB for n in nids},
            affinity_getter=lambda: frozenset(range(16)),
            affinity_setter=lambda cpus: affinity_log.append(frozenset(cpus)),
        )

    assert len(pool.allocated_ptrs) == 2
    assert pool.freed_ptrs == list(reversed(pool.allocated_ptrs))
    assert affinity_log[-1] == frozenset(range(16))


def test_shard_wrap_failure_releases_the_current_native_pointer():
    pool = _FakePinnedPool()
    with (
        patch.object(torch, "frombuffer", side_effect=RuntimeError("wrap failed")),
        pytest.raises(RuntimeError, match="wrap failed"),
    ):
        NumaShardedPinnedAllocator(
            MiB,
            topo_nodes=_make_nodes(2),
            alloc_fn=pool.alloc,
            free_fn=pool.free,
            memfree_reader=lambda ids: dict.fromkeys(ids, GiB),
            affinity_getter=lambda: frozenset(),
            affinity_setter=lambda _cpus: None,
        )
    assert len(pool.allocated_ptrs) == 1
    assert pool.freed_ptrs == pool.allocated_ptrs


def test_close_retries_only_unreleased_shards():
    pool = _FakePinnedPool()
    attempts: list[int] = []

    def free(ptr: int) -> None:
        attempts.append(ptr)
        if len(attempts) == 1:
            raise RuntimeError("free failed once")
        pool.free(ptr)

    allocator = NumaShardedPinnedAllocator(
        MiB,
        topo_nodes=_make_nodes(2),
        alloc_fn=pool.alloc,
        free_fn=free,
        memfree_reader=lambda ids: dict.fromkeys(ids, GiB),
        affinity_getter=lambda: frozenset(),
        affinity_setter=lambda _cpus: None,
    )
    first, second = pool.allocated_ptrs
    with pytest.raises(RuntimeError, match="free failed once"):
        allocator.close()
    assert allocator.allocate(torch.Size([1]), torch.uint8) is None
    allocator.close()
    allocator.close()
    assert attempts == [second, first, second]


def test_misaligned_shard_base_pointer_fails_fast_and_rolls_back():
    pool = _FakePinnedPool(misalign_by=64)
    with pytest.raises(RuntimeError, match="not aligned to align_bytes=4096"):
        NumaShardedPinnedAllocator(
            size_in_bytes=4 * MiB,
            align_bytes=4096,
            topo_nodes=_make_nodes(2),
            alloc_fn=pool.alloc,
            free_fn=pool.free,
            memfree_reader=lambda ids: dict.fromkeys(ids, GiB),
            affinity_getter=lambda: frozenset(),
            affinity_setter=lambda _cpus: None,
        )
    assert len(pool.allocated_ptrs) == 1
    assert pool.freed_ptrs == pool.allocated_ptrs


def test_init_rollback_double_fault_quarantines_and_retries_unreleased_shards():
    retry_unreleased_init_shards()
    pool = _FakePinnedPool(fail_on_call=3)
    fail_free_once = {"armed": True}

    def flaky_free(ptr: int) -> None:
        if fail_free_once["armed"] and ptr == pool.allocated_ptrs[0]:
            fail_free_once["armed"] = False
            raise RuntimeError("rollback free failed on shard 0")
        pool.free(ptr)

    with pytest.raises(RuntimeError, match="pinned host allocation failed") as exc_info:
        NumaShardedPinnedAllocator(
            size_in_bytes=16 * MiB,
            align_bytes=4096,
            topo_nodes=_make_nodes(4),
            alloc_fn=pool.alloc,
            free_fn=flaky_free,
            memfree_reader=lambda ids: dict.fromkeys(ids, 128 * GiB),
            affinity_getter=lambda: frozenset(range(16)),
            affinity_setter=lambda _cpus: None,
        )

    stranded = getattr(exc_info.value, "_unreleased_numa_shards", ())
    assert len(stranded) == 1
    assert stranded[0].ptr == pool.allocated_ptrs[0]
    assert stranded[0].released is False
    assert pool.freed_ptrs == [pool.allocated_ptrs[1]]

    remaining = retry_unreleased_init_shards()
    assert remaining == 0
    assert stranded[0].released is True
    assert pool.freed_ptrs == [pool.allocated_ptrs[1], pool.allocated_ptrs[0]]


# ---------------------------------------------------------------------------
# 5. L1MemoryManager & StorageManager integration
# ---------------------------------------------------------------------------


def test_l1_memory_manager_and_storage_manager_pure_dram_sharded_startup():
    pool = _FakePinnedPool()
    nodes = _make_nodes(4, cpus_per_node=4)

    def _patched_sharded_factory(
        size_in_bytes: int,
        align_bytes: int = 4096,
    ) -> NumaShardedPinnedAllocator:
        return NumaShardedPinnedAllocator(
            size_in_bytes=size_in_bytes,
            align_bytes=align_bytes,
            topo_nodes=nodes,
            alloc_fn=pool.alloc,
            free_fn=pool.free,
            memfree_reader=lambda nids: {n: 128 * GiB for n in nids},
            affinity_getter=lambda: frozenset(range(16)),
            affinity_setter=lambda _cpus: None,
        )

    fake_spec = SimpleNamespace(device_type="cuda", is_pin_supported=True)
    with (
        patch("lmcache.v1.distributed.config.current_device_spec", fake_spec),
        patch(
            "lmcache.v1.memory_allocators.numa_sharded_pinned_allocator.NumaShardedPinnedAllocator",
            side_effect=_patched_sharded_factory,
        ),
    ):
        mem_cfg = L1MemoryManagerConfig(
            size_in_bytes=8 * MiB,
            use_lazy=True,
            align_bytes=4096,
            use_numa_sharding=True,
        )
        mgr = L1MemoryManager(mem_cfg)
        try:
            assert mgr.get_l1_memory_desc() is None
            layout = MemoryLayoutDesc(
                shapes=[torch.Size([256, 1024])], dtypes=[torch.uint8]
            )
            err, objs = mgr.allocate(layout, count=6)
            assert err == L1Error.SUCCESS
            assert len(objs) == 6
            used, total = mgr.get_memory_usage()
            assert used == 6 * 256 * 1024
            assert total == 8 * MiB
            assert mgr.free(objs) == L1Error.SUCCESS
            assert mgr.get_memory_usage() == (0, 8 * MiB)
        finally:
            mgr.close()

        sm_cfg = StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(memory_config=mem_cfg),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(adapters=[]),
        )
        assert l1_exposes_single_memory_region(sm_cfg) is False
        if sys.platform != "win32":
            sm = StorageManager(sm_cfg)
            try:
                with pytest.raises(ValueError, match="no registerable memory buffer"):
                    _ = sm.l1_memory_desc
            finally:
                sm.close()


def test_sharded_allocator_thread_safety_under_concurrent_alloc_and_free():
    pool = _FakePinnedPool()
    nodes = _make_nodes(4, cpus_per_node=4)
    alloc = NumaShardedPinnedAllocator(
        size_in_bytes=32 * MiB,
        align_bytes=4096,
        topo_nodes=nodes,
        alloc_fn=pool.alloc,
        free_fn=pool.free,
        memfree_reader=lambda nids: {n: 128 * GiB for n in nids},
        affinity_getter=lambda: frozenset(range(16)),
        affinity_setter=lambda _cpus: None,
    )
    errors: list[BaseException] = []
    shape = torch.Size([256, 1024])

    def _worker(seed: int) -> None:
        try:
            for step in range(40):
                objs = alloc.batched_allocate(shape, torch.uint8, batch_size=4)
                if objs is not None:
                    for obj in objs:
                        obj.tensor.fill_((seed + step) % 251)
                    alloc.batched_free(objs)
        except BaseException as exc:
            errors.append(exc)

    try:
        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert alloc.get_memory_usage() == (0, 32 * MiB)
        assert alloc.memcheck() is True
    finally:
        alloc.close()


# ---------------------------------------------------------------------------
# 6. Opt-in CLI/env configuration & compatibility validation
# ---------------------------------------------------------------------------


def test_opt_in_default_disabled_cli_env_and_compatibility_validation():
    fake_spec = SimpleNamespace(device_type="npu", is_pin_supported=True)
    with (
        patch("lmcache.v1.distributed.config.current_device_spec", fake_spec),
        patch.dict(os.environ, {}, clear=False),
    ):
        os.environ.pop("LMCACHE_L1_NUMA_SHARDING", None)
        parser = get_arg_parser()

        # Default MUST remain False (opt-in feature, zero default behavior change).
        args_default = parser.parse_args(
            ["--l1-size-gb", "480", "--eviction-policy", "LRU"]
        )
        cfg_default = parse_args_to_config(args_default)
        assert cfg_default.l1_manager_config.memory_config.use_numa_sharding is False
        assert l1_exposes_single_memory_region(cfg_default) is True

        # Explicit --l1-numa-sharding enables sharding; P2P / engine_driven raise ValueError.
        args_on = parser.parse_args(
            ["--l1-size-gb", "480", "--eviction-policy", "LRU", "--l1-numa-sharding"]
        )
        cfg_on = parse_args_to_config(args_on)
        assert cfg_on.l1_manager_config.memory_config.use_numa_sharding is True
        assert l1_exposes_single_memory_region(cfg_on) is False

        with pytest.raises(ValueError, match="incompatible with: P2P"):
            validate_l1_numa_sharding_compatibility(cfg_on, p2p_enabled=True)
        with pytest.raises(ValueError, match="supported_transfer_mode"):
            validate_l1_numa_sharding_compatibility(
                cfg_on, transfer_mode="engine_driven"
            )

        # --gds-l1-path + --l1-numa-sharding raises ValueError.
        args_gds = parser.parse_args(
            [
                "--l1-size-gb",
                "64",
                "--eviction-policy",
                "LRU",
                "--gds-l1-path",
                "/tmp/gds_slab",
                "--l1-numa-sharding",
            ]
        )
        with pytest.raises(ValueError, match="GDS L1"):
            parse_args_to_config(args_gds)

        # --no-l1-use-lazy + --l1-numa-sharding raises ValueError.
        args_nolazy = parser.parse_args(
            [
                "--l1-size-gb",
                "64",
                "--eviction-policy",
                "LRU",
                "--no-l1-use-lazy",
                "--l1-numa-sharding",
            ]
        )
        with pytest.raises(ValueError, match="requires --l1-use-lazy"):
            parse_args_to_config(args_nolazy)

        # --l1-manager JSON rejects mixing with --l1-numa-sharding CLI or env var.
        l1_json = (
            '{"type": "DRAM", "tag": "main", "size_gb": 16, '
            '"eviction": {"eviction_policy": "LRU"}}'
        )
        args_mgr_cli = parser.parse_args(
            ["--l1-manager", l1_json, "--l1-numa-sharding"]
        )
        with pytest.raises(ValueError, match="Use either --l1-manager or legacy L1 flags"):
            parse_args_to_config(args_mgr_cli)

        args_mgr_only = parser.parse_args(["--l1-manager", l1_json])
        with patch.dict(os.environ, {"LMCACHE_L1_NUMA_SHARDING": "1"}, clear=False):
            with pytest.raises(ValueError, match="not supported with --l1-manager JSON"):
                parse_args_to_config(args_mgr_only)

    # Environment variable opt-in works without CLI flag, and --no-l1-numa-sharding overrides it.
    with (
        patch("lmcache.v1.distributed.config.current_device_spec", fake_spec),
        patch.dict(os.environ, {"LMCACHE_L1_NUMA_SHARDING": "true"}, clear=False),
    ):
        args_env = parser.parse_args(
            ["--l1-size-gb", "480", "--eviction-policy", "LRU"]
        )
        assert (
            parse_args_to_config(args_env).l1_manager_config.memory_config.use_numa_sharding
            is True
        )
        args_override = parser.parse_args(
            ["--l1-size-gb", "480", "--eviction-policy", "LRU", "--no-l1-numa-sharding"]
        )
        assert (
            parse_args_to_config(args_override)
            .l1_manager_config.memory_config.use_numa_sharding
            is False
        )


# ---------------------------------------------------------------------------
# 7. Owner context safe teardown ordering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("owner_context_rc", [0, 801])
def test_safe_free_pinned_shard_owner_context_ordering(owner_context_rc: int):
    events: list[tuple[str, object]] = []
    state = {"context": "caller_ctx"}

    class _FakeNpu:
        def is_available(self) -> bool:
            return True

        def current_device(self) -> int:
            return 7

        def synchronize(self, dev: int) -> None:
            events.append(("synchronize", dev))

        def set_device(self, dev: int) -> None:
            events.append(("set_device", dev))
            state["context"] = f"default-{dev}"

    class _FakeAclRt:
        def get_context(self):
            events.append(("get_context", None))
            return state["context"], 0

        def set_context(self, ctx) -> int:
            events.append(("set_context", ctx))
            if ctx == "owner_ctx" and owner_context_rc:
                return owner_context_rc
            state["context"] = ctx
            return 0

    fake_acl = SimpleNamespace(rt=_FakeAclRt())
    shard = _PinnedAllocation(
        shard_idx=0,
        numa_node=0,
        ptr=0xDEAD0000,
        owner_dev=2,
        owner_ctx="owner_ctx",
    )

    def free() -> None:
        _safe_free_pinned_shard(
            shard,
            lambda ptr: events.append(("free_fn", state["context"])),
            npu_module=_FakeNpu(),
            acl_module=fake_acl,
        )

    if owner_context_rc:
        with pytest.raises(RuntimeError, match="Cannot select ACL owner context"):
            free()
    else:
        free()
    assert events == [
        ("get_context", None),
        ("synchronize", 2),
        ("set_device", 2),
        ("set_context", "owner_ctx"),
        *([] if owner_context_rc else [("free_fn", "owner_ctx")]),
        ("set_device", 7),
        ("set_context", "caller_ctx"),
    ]
    assert state["context"] == "caller_ctx"
    assert shard.released is (owner_context_rc == 0)
