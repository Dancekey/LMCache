# SPDX-License-Identifier: Apache-2.0
"""Exercise inline lookup_with_result aggregation without device imports."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]


def methods(relative: str, owner_name: str, names: set[str], namespace: dict):
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner_name
    )
    nodes = [
        n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    if {n.name for n in nodes} != set(names):
        raise AssertionError(f"Missing real methods in {relative}")
    for n in nodes:
        n.decorator_list = []
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    code = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(code, str(path), "exec"), namespace)
    return type(
        owner_name + "ActualMethods", (), {n.name: namespace[n.name] for n in nodes}
    )


class Reply:
    def __init__(self, value=None, ready: bool = True):
        self.value, self.ready = value, ready
        self.timeouts: list = []

    def query(self) -> bool:
        return self.ready

    def result(self, timeout=None):
        self.timeouts.append(timeout)
        if not self.ready:
            raise TimeoutError("reply not ready")
        return self.value


def inline_stages(left, right, *, enabled: bool = True):
    clock = SimpleNamespace(now=0.0)
    cls = methods(
        "lmcache/integration/vllm/vllm_multi_process_adapter.py",
        "LMCacheMPSchedulerAdapter",
        {"check_lookup_result"},
        {
            "time": SimpleNamespace(monotonic=lambda: clock.now),
            "logger": logging.getLogger("inline-lookup-contract"),
        },
    )
    obj = cls()
    obj._pending_lookups = {"r"}
    obj._finished_lookup_results = {}
    obj._unacked_lookups = {}
    obj._lookup_status = {}
    obj._per_server_hits = {}
    obj._server_urls = ["early", "late"]
    obj.is_healthy = True
    obj._mq_timeout = 5.0
    obj._nonblocking_lookup_status = True
    obj._pp_query_fastpath = enabled
    obj.lmcache_tokens_per_chunk = 4
    obj.timed_out = []
    obj._mark_lookup_timed_out = lambda url: obj.timed_out.append(url)
    obj._free_inconsistent_lookup_locks = lambda *args, **kwargs: None
    obj._health_events = {url: threading.Event() for url in obj._server_urls}
    obj.sent = []
    obj.replies = {"early": Reply(0), "late": Reply(2, ready=False)}

    def query(url, request_id):
        obj.sent.append((url, request_id))
        return obj.replies[url]

    obj.req_clients = {
        url: SimpleNamespace(query_prefetch_status=lambda rid, url=url: query(url, rid))
        for url in obj._server_urls
    }
    obj._unacked_lookups["r"] = SimpleNamespace(
        futures={"early": left, "late": right}, submitted_at=0.0
    )
    return obj


class InlineLookupTest(unittest.TestCase):
    def test_inline_counts_use_common_prefix_without_status(self):
        adapter = inline_stages(Reply(15), Reply(14))
        self.assertEqual(adapter.check_lookup_result("r"), 14 * 4)
        self.assertEqual(adapter.check_lookup_result("r"), 14 * 4)
        self.assertEqual(adapter.sent, [])

    def test_pending_or_legacy_peer_alone_uses_status(self):
        adapter = inline_stages(Reply(14), Reply(None))
        self.assertIsNone(adapter.check_lookup_result("r"))
        self.assertEqual(adapter.sent, [("late", "r")])
        adapter.replies["late"].ready = True
        self.assertEqual(adapter.check_lookup_result("r"), 2 * 4)
        self.assertEqual(adapter.sent, [("late", "r")])

    def test_inline_probe_uses_original_submit_and_consume_paths_once(self):
        cls = methods(
            "lmcache/v1/multiprocess/modules/lookup.py",
            "LookupModule",
            {"lookup_with_result"},
            {},
        )
        for value in (0, 14, None):
            with self.subTest(value=value):
                server, calls = cls(), []
                key = SimpleNamespace(request_id="r")
                server.lookup = lambda k, tp: calls.append((k, tp))
                server.query_prefetch_status = lambda r: calls.append(r) or value
                self.assertEqual(server.lookup_with_result(key, 16), value)
                self.assertEqual(calls, [(key, 16), "r"])


if __name__ == "__main__":
    unittest.main()
