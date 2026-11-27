# SPDX-License-Identifier: Apache-2.0
"""NUMA topology discovery and MemFree-equalizing shard planning for L1 memory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re

from lmcache.logging import init_logger

logger = init_logger(__name__)

_SYSFS_NODE_ROOT = Path("/sys/devices/system/node")
_THREAD_STATUS = Path("/proc/thread-self/status")
_MEMINFO_LINE_RE = re.compile(r"^Node\s+(\d+)\s+(\w+):\s+(\d+)\s+kB$")


@dataclass(frozen=True)
class NumaNodeInfo:
    """Process-usable NUMA node descriptor."""

    node_id: int
    cpus: frozenset[int]
    mem_total_bytes: int


@dataclass(frozen=True)
class ShardPlan:
    """Planned pinned-memory shard assigned to one NUMA node."""

    node_id: int
    size_bytes: int
    cpus: frozenset[int]


def parse_cpu_or_node_list(spec: str) -> list[int]:
    """Parse a Linux sysfs cpulist/nodelist string such as ``'0-3,8,10-11'``."""
    result: list[int] = []
    text = spec.strip()
    if not text:
        return result
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid range {token!r} in list {spec!r}")
            result.extend(range(start, end + 1))
        else:
            result.append(int(token))
    return sorted(set(result))


def align_down(size: int, align_bytes: int) -> int:
    """Round ``size`` down to a multiple of power-of-two ``align_bytes``."""
    if align_bytes <= 0 or (align_bytes & (align_bytes - 1)) != 0:
        raise ValueError(
            f"align_bytes must be a positive power of two, got {align_bytes}"
        )
    if size <= 0:
        return 0
    return size & (~(align_bytes - 1))


def align_up(size: int, align_bytes: int) -> int:
    """Round ``size`` up to a multiple of power-of-two ``align_bytes``."""
    if align_bytes <= 0 or (align_bytes & (align_bytes - 1)) != 0:
        raise ValueError(
            f"align_bytes must be a positive power of two, got {align_bytes}"
        )
    if size <= 0:
        return 0
    return (size + align_bytes - 1) & (~(align_bytes - 1))


def _parse_node_meminfo(meminfo_path: Path) -> dict[str, int]:
    """Read key -> bytes from ``/sys/devices/system/node/nodeX/meminfo``."""
    values: dict[str, int] = {}
    try:
        content = meminfo_path.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in content.splitlines():
        m = _MEMINFO_LINE_RE.match(line.strip())
        if m:
            values[m.group(2)] = int(m.group(3)) * 1024
    return values


def read_numa_memfree_bytes(
    node_ids: Sequence[int],
    *,
    sysfs_root: Path = _SYSFS_NODE_ROOT,
) -> dict[int, int]:
    """Read current ``MemFree`` in bytes for each requested NUMA node."""
    result: dict[int, int] = {}
    for nid in node_ids:
        info = _parse_node_meminfo(sysfs_root / f"node{nid}" / "meminfo")
        if "MemFree" in info:
            result[nid] = info["MemFree"]
    return result


def discover_process_numa_nodes(
    *,
    sysfs_root: Path = _SYSFS_NODE_ROOT,
    allowed_cpus: frozenset[int] | None = None,
    status_path: Path = _THREAD_STATUS,
) -> list[NumaNodeInfo]:
    """Discover NUMA nodes with memory and CPUs accessible to this process.

    Intersects the thread's CPU affinity (``sched_getaffinity``) and allowed
    memory nodes (``Mems_allowed_list``) with sysfs NUMA topology. Hosts
    without sysfs NUMA topology fall back to a single implicit node.
    """
    if allowed_cpus is None:
        if hasattr(os, "sched_getaffinity"):
            allowed_cpus = frozenset(os.sched_getaffinity(0))
        else:
            allowed_cpus = frozenset()

    allowed_mems = None
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("Mems_allowed_list:"):
                allowed_mems = set(parse_cpu_or_node_list(line.split(":", 1)[1]))
                break

    online_path = sysfs_root / "online"
    if not online_path.is_file():
        if allowed_mems is not None:
            raise ValueError("Cannot verify NUMA placement without sysfs topology")
        return [NumaNodeInfo(node_id=0, cpus=allowed_cpus, mem_total_bytes=0)]

    online_ids = parse_cpu_or_node_list(online_path.read_text(encoding="utf-8"))
    discovered: list[NumaNodeInfo] = []
    for nid in online_ids:
        if allowed_mems is not None and nid not in allowed_mems:
            continue
        node_dir = sysfs_root / f"node{nid}"
        cpulist_path = node_dir / "cpulist"
        meminfo = _parse_node_meminfo(node_dir / "meminfo")
        mem_total = meminfo.get("MemTotal", 0)
        if mem_total <= 0:
            continue

        node_cpus: frozenset[int] = frozenset()
        if cpulist_path.is_file():
            node_cpus = frozenset(
                parse_cpu_or_node_list(cpulist_path.read_text(encoding="utf-8"))
            )
        usable_cpus = (
            node_cpus & allowed_cpus if (allowed_cpus and node_cpus) else node_cpus
        )
        if allowed_cpus and not usable_cpus:
            continue
        discovered.append(
            NumaNodeInfo(node_id=nid, cpus=usable_cpus, mem_total_bytes=mem_total)
        )

    if not discovered:
        raise ValueError("No online NUMA node has both permitted CPUs and memory")
    return discovered


def _split_even(
    total_bytes: int,
    nodes: Sequence[NumaNodeInfo],
    align_bytes: int,
) -> list[ShardPlan]:
    """Split ``total_bytes`` evenly across ``nodes``, preserving exact aligned sum."""
    base_blocks, remainder = divmod(total_bytes // align_bytes, len(nodes))
    plans: list[ShardPlan] = []
    for idx, node in enumerate(nodes):
        size = (base_blocks + int(idx < remainder)) * align_bytes
        if size > 0:
            plans.append(
                ShardPlan(node_id=node.node_id, size_bytes=size, cpus=node.cpus)
            )
    return plans


def plan_numa_shards(
    total_bytes: int,
    align_bytes: int,
    nodes: Sequence[NumaNodeInfo],
    memfree_by_node: Mapping[int, int] | None = None,
) -> list[ShardPlan]:
    """Compute per-NUMA shard byte quotas summing exactly to aligned ``total_bytes``.

    Uses parameter-free ``MemFree`` water-filling to maximize the minimum
    post-allocation ``MemFree`` across accessible NUMA nodes:
      - On balanced hosts, this produces an exact even split across all nodes.
      - On skewed hosts (where some nodes have low raw ``MemFree``), quota
        automatically shifts to spacious nodes and assigns 0 to tight nodes.
      - When host-wide raw ``MemFree`` is below ``total_bytes`` (or unavailable),
        falls back to an even split across accessible nodes with a warning so
        kernel page-cache reclaim spreads across nodes.
    """
    if not nodes:
        raise ValueError("Cannot plan NUMA shards with an empty node list")

    effective_align = max(align_bytes, 65536)
    if effective_align & (effective_align - 1) != 0:
        raise ValueError(f"align_bytes must be a power of two, got {align_bytes}")

    aligned_total = align_up(total_bytes, effective_align)
    if aligned_total <= 0:
        raise ValueError(f"total_bytes must be positive, got {total_bytes}")

    if len(nodes) == 1:
        only = nodes[0]
        return [
            ShardPlan(node_id=only.node_id, size_bytes=aligned_total, cpus=only.cpus)
        ]

    if not memfree_by_node:
        logger.warning(
            "NUMA MemFree unavailable; falling back to even split across %d nodes",
            len(nodes),
        )
        return _split_even(aligned_total, nodes, effective_align)

    need_blocks = aligned_total // effective_align
    avail_blocks = {
        n.node_id: max(0, int(memfree_by_node.get(n.node_id, 0))) // effective_align
        for n in nodes
    }
    total_avail_blocks = sum(avail_blocks.values())

    if total_avail_blocks < need_blocks:
        logger.warning(
            "Total NUMA raw MemFree (%.2f GiB) is below requested L1 size "
            "(%.2f GiB); falling back to even split across %d accessible nodes "
            "so OS page-cache reclaim spreads evenly.",
            (total_avail_blocks * effective_align) / (1 << 30),
            aligned_total / (1 << 30),
            len(nodes),
        )
        return _split_even(aligned_total, nodes, effective_align)

    # Iterative water-filling: find the maximum post-allocation MemFree floor L
    # such that sum(max(0, avail_i - L)) == need_blocks.
    active = list(nodes)
    while True:
        surplus = sum(avail_blocks[n.node_id] for n in active) - need_blocks
        floor_level, rem_surplus = divmod(surplus, len(active))
        next_active = [n for n in active if avail_blocks[n.node_id] > floor_level]
        if len(next_active) == len(active):
            break
        active = next_active

    # Distribute the exact surplus blocks across active nodes so each keeps
    # floor_level (first len-rem_surplus nodes) or floor_level + 1 (last
    # rem_surplus nodes), giving every active node alloc_i >= 0 and
    # sum(alloc_i) == need_blocks without exceeding avail_blocks[i].
    split_idx = len(active) - rem_surplus
    plans: list[ShardPlan] = []
    for idx, node in enumerate(active):
        keep = floor_level + int(idx >= split_idx)
        alloc_blocks = avail_blocks[node.node_id] - keep
        if alloc_blocks > 0:
            plans.append(
                ShardPlan(
                    node_id=node.node_id,
                    size_bytes=alloc_blocks * effective_align,
                    cpus=node.cpus,
                )
            )
    return plans
