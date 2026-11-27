# SPDX-License-Identifier: Apache-2.0
"""NUMA topology discovery and headroom-aware shard planning for L1 memory."""

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
_MIN_EFFECTIVE_SHARD_BYTES = 64 << 20  # 64 MiB minimum shard floor when shifting
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
    explicit_nodes: Sequence[int] | None = None,
    *,
    sysfs_root: Path = _SYSFS_NODE_ROOT,
    allowed_cpus: frozenset[int] | None = None,
    status_path: Path = _THREAD_STATUS,
) -> list[NumaNodeInfo]:
    """Discover NUMA nodes with memory and CPUs accessible to this process.

    Intersects the thread's CPU affinity and allowed memory nodes with sysfs
    topology. Hosts without sysfs NUMA topology can only use the implicit
    single-node plan.
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
        if explicit_nodes or allowed_mems is not None:
            raise ValueError("Cannot verify NUMA placement without sysfs topology")
        return [NumaNodeInfo(node_id=0, cpus=allowed_cpus, mem_total_bytes=0)]

    online_ids = parse_cpu_or_node_list(online_path.read_text(encoding="utf-8"))
    if explicit_nodes is not None:
        requested = list(dict.fromkeys(int(n) for n in explicit_nodes))
        missing = [n for n in requested if n not in online_ids]
        if missing:
            raise ValueError(
                f"Requested NUMA nodes {missing} are not online (online={online_ids})"
            )
        candidate_ids = requested
    else:
        candidate_ids = online_ids

    discovered: list[NumaNodeInfo] = []
    for nid in candidate_ids:
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

    if explicit_nodes is not None and len(discovered) != len(candidate_ids):
        unusable = set(candidate_ids) - {node.node_id for node in discovered}
        raise ValueError(
            f"Requested NUMA nodes {sorted(unusable)} lack permitted CPUs or memory"
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


def _water_fill(
    total_bytes: int,
    nodes: Sequence[NumaNodeInfo],
    caps_by_node: Mapping[int, int],
    align_bytes: int,
) -> list[ShardPlan]:
    """Distribute ``total_bytes`` evenly subject to per-node upper bounds ``caps``."""
    quotas: dict[int, int] = {n.node_id: 0 for n in nodes}
    unfilled = [n.node_id for n in nodes if caps_by_node.get(n.node_id, 0) > 0]
    remaining = total_bytes

    while unfilled and remaining > 0:
        share = align_down(remaining // len(unfilled), align_bytes)
        if share == 0:
            slack_sorted = sorted(
                unfilled,
                key=lambda nid: caps_by_node[nid] - quotas[nid],
                reverse=True,
            )
            for nid in slack_sorted:
                slack = caps_by_node[nid] - quotas[nid]
                take = min(remaining, align_down(slack, align_bytes))
                if take > 0:
                    quotas[nid] += take
                    remaining -= take
                if remaining == 0:
                    break
            break

        capped_any = False
        next_unfilled: list[int] = []
        for nid in unfilled:
            cap = caps_by_node[nid]
            if cap <= share:
                quotas[nid] = cap
                remaining -= cap
                capped_any = True
            else:
                next_unfilled.append(nid)

        if not capped_any:
            base_blocks, rem_blocks = divmod(
                remaining // align_bytes, len(unfilled)
            )
            for idx, nid in enumerate(unfilled):
                take = (base_blocks + int(idx < rem_blocks)) * align_bytes
                cap_slack = align_down(caps_by_node[nid] - quotas[nid], align_bytes)
                if take > cap_slack:
                    take = cap_slack
                quotas[nid] += take
                remaining -= take
            if remaining > 0:
                for nid in sorted(
                    unfilled,
                    key=lambda x: caps_by_node[x] - quotas[x],
                    reverse=True,
                ):
                    slack = align_down(caps_by_node[nid] - quotas[nid], align_bytes)
                    take = min(remaining, slack)
                    if take > 0:
                        quotas[nid] += take
                        remaining -= take
                    if remaining == 0:
                        break
            break
        unfilled = next_unfilled

    if remaining != 0:
        raise RuntimeError(
            f"Water-filling left unallocated remainder={remaining} bytes "
            f"(total={total_bytes}, caps={dict(caps_by_node)})"
        )

    # Fold tiny shards (< min_shard) into nodes with largest remaining headroom.
    min_shard = min(
        _MIN_EFFECTIVE_SHARD_BYTES,
        max(align_bytes, total_bytes // max(len(nodes) * 4, 1)),
    )
    min_shard = align_up(max(align_bytes, min_shard), align_bytes)
    tiny_nodes = [nid for nid, q in quotas.items() if 0 < q < min_shard]
    for tiny_nid in tiny_nodes:
        tiny_q = quotas[tiny_nid]
        donors = [
            nid
            for nid in quotas
            if nid != tiny_nid
            and quotas[nid] > 0
            and caps_by_node[nid] - quotas[nid] >= tiny_q
        ]
        if donors:
            target = max(donors, key=lambda nid: caps_by_node[nid] - quotas[nid])
            quotas[target] += tiny_q
            quotas[tiny_nid] = 0

    node_map = {n.node_id: n for n in nodes}
    return [
        ShardPlan(
            node_id=nid,
            size_bytes=quotas[nid],
            cpus=node_map[nid].cpus,
        )
        for nid in [n.node_id for n in nodes]
        if quotas[nid] > 0
    ]


def plan_numa_shards(
    total_bytes: int,
    align_bytes: int,
    policy: str,
    nodes: Sequence[NumaNodeInfo],
    memfree_by_node: Mapping[int, int] | None = None,
    headroom_bytes: int = 32 << 30,
) -> list[ShardPlan]:
    """Compute per-NUMA shard byte quotas summing exactly to aligned ``total_bytes``.

    Strategies:
      - ``even``: Equal split across all candidate nodes.
      - ``auto`` / ``nodes``:
          * Case A (balanced nodes where every ``MemFree - headroom`` satisfies
            an equal share): exact equal split across all candidate nodes.
          * Case B (skewed ``MemFree`` where ``sum(max(0, MemFree - headroom))
            >= total_bytes``): capped water-filling shifts quota away from tight
            nodes to spacious nodes while preserving ``headroom_bytes`` on every
            spacious node.
          * Case C (host-wide raw ``MemFree`` cannot preserve full headroom):
            - If ``sum(MemFree) >= total_bytes``, relax headroom evenly to
              equalize post-allocation ``MemFree`` and log a warning.
            - If ``sum(MemFree) < total_bytes``, fall back to an even split
              across candidate nodes so kernel page-cache reclaim spreads evenly.
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

    if policy == "even" or not memfree_by_node:
        return _split_even(aligned_total, nodes, effective_align)

    if policy not in ("auto", "nodes"):
        raise ValueError(f"Unsupported NUMA sharding policy: {policy!r}")

    node_ids = [n.node_id for n in nodes]
    if any(nid not in memfree_by_node for nid in node_ids):
        logger.warning(
            "Incomplete per-NUMA MemFree readings (%s for nodes %s); "
            "falling back to even split.",
            sorted(memfree_by_node),
            node_ids,
        )
        return _split_even(aligned_total, nodes, effective_align)

    headroom = max(0, int(headroom_bytes))
    caps_with_headroom = {
        nid: align_down(max(0, memfree_by_node[nid] - headroom), effective_align)
        for nid in node_ids
    }
    equal_plans = _split_even(aligned_total, nodes, effective_align)

    # Case A: Clean / balanced nodes -> exact equal split.
    if all(p.size_bytes <= caps_with_headroom[p.node_id] for p in equal_plans):
        return equal_plans

    # Case B: Skewed MemFree, total safe capacity covers total_bytes.
    total_safe_cap = sum(caps_with_headroom.values())
    if total_safe_cap >= aligned_total:
        plans = _water_fill(aligned_total, nodes, caps_with_headroom, effective_align)
        logger.info(
            "L1 NUMA water-filling shifted quotas away from tight nodes "
            "(headroom=%.2f GiB): %s",
            headroom / (1 << 30),
            [(p.node_id, round(p.size_bytes / (1 << 30), 2)) for p in plans],
        )
        return plans

    # Case C1: Raw MemFree sum covers total_bytes, but not full headroom.
    raw_caps = {
        nid: align_down(max(0, memfree_by_node[nid]), effective_align)
        for nid in node_ids
    }
    total_raw_cap = sum(raw_caps.values())
    if total_raw_cap >= aligned_total:
        reduced_headroom = align_down(
            max(0, (total_raw_cap - aligned_total) // len(nodes)), effective_align
        )
        relaxed_caps = {
            nid: align_down(
                max(0, memfree_by_node[nid] - reduced_headroom), effective_align
            )
            for nid in node_ids
        }
        if sum(relaxed_caps.values()) < aligned_total:
            relaxed_caps = raw_caps
        logger.warning(
            "Host NUMA MemFree (sum=%.2f GiB) cannot preserve full headroom "
            "(%.2f GiB/node) for L1 %.2f GiB; relaxing headroom to %.2f GiB/node. "
            "Consider clearing stale OS page cache before starting services.",
            total_raw_cap / (1 << 30),
            headroom / (1 << 30),
            aligned_total / (1 << 30),
            reduced_headroom / (1 << 30),
        )
        return _water_fill(aligned_total, nodes, relaxed_caps, effective_align)

    # Case C2: Host raw MemFree sum < total_bytes (page cache fills RAM).
    logger.warning(
        "Host NUMA raw MemFree sum (%.2f GiB) is below requested L1 size "
        "(%.2f GiB); falling back to even split across nodes %s so kernel "
        "page-cache reclaim spreads evenly.",
        total_raw_cap / (1 << 30),
        aligned_total / (1 << 30),
        node_ids,
    )
    return _split_even(aligned_total, nodes, effective_align)
