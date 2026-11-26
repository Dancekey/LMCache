# SPDX-License-Identifier: Apache-2.0
"""Opt-in lock-free monotonic JSONL trace emitter for PP bubble attribution.

Disabled by default (zero overhead beyond checking a module boolean). When
``AGY_MP_TRACE_DIR`` is set to a directory path, appends per-PID JSONL records
stamped with ``time.monotonic_ns()`` so standalone ``MPCacheServer`` DMA spans
and ``vLLM`` worker/scheduler spans share the same host OS monotonic clock.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterator

_TRACE_DIR = os.environ.get("AGY_MP_TRACE_DIR", "").strip()
_NODE_LABEL = os.environ.get("AGY_MP_TRACE_NODE", "").strip()
_ENABLED = bool(_TRACE_DIR)

_fd: int | None = None
_fd_pid: int | None = None
_init_lock = threading.Lock()


def is_enabled() -> bool:
    """Return whether monotonic PP bubble tracing is active in this process."""
    return _ENABLED


def _ensure_fd() -> int | None:
    global _fd, _fd_pid
    if not _ENABLED:
        return None
    pid = os.getpid()
    if _fd is not None and _fd_pid == pid:
        return _fd
    with _init_lock:
        if _fd is not None and _fd_pid == pid:
            return _fd
        try:
            folder = Path(_TRACE_DIR)
            folder.mkdir(parents=True, exist_ok=True)
            prefix = f"{_NODE_LABEL}-" if _NODE_LABEL else ""
            path = folder / f"mp-trace-{prefix}{pid}.jsonl"
            _fd = os.open(
                str(path),
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o644,
            )
            _fd_pid = pid
            return _fd
        except OSError:
            return None


def emit_event(event: str, **fields: Any) -> None:
    """Write one JSONL trace record; silently drops I/O errors."""
    if not _ENABLED:
        return
    fd = _ensure_fd()
    if fd is None:
        return
    try:
        payload: dict[str, Any] = {
            "event": event,
            "node": _NODE_LABEL,
            "pid": os.getpid(),
            "tid": threading.get_native_id(),
            "mono_ns": time.monotonic_ns(),
        }
        payload.update(fields)
        os.write(fd, (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))
    except Exception:
        pass


def emit_span(event: str, start_ns: int, end_ns: int | None = None, **fields: Any) -> None:
    """Write one [start_ns, end_ns] interval record."""
    if not _ENABLED:
        return
    done_ns = time.monotonic_ns() if end_ns is None else end_ns
    emit_event(
        event,
        start_ns=start_ns,
        end_ns=done_ns,
        duration_ns=max(0, done_ns - start_ns),
        **fields,
    )


@contextmanager
def trace_span(event: str, **fields: Any) -> Iterator[None]:
    """Context manager recording host monotonic start/end timestamps."""
    if not _ENABLED:
        yield
        return
    t0 = time.monotonic_ns()
    try:
        yield
    finally:
        emit_span(event, t0, time.monotonic_ns(), **fields)
