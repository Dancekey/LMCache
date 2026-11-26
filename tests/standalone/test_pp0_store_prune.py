"""Exercise the STORE adapter and context on CPU without importing device runtimes."""

import ast
import logging
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[2] / "lmcache"


def load_class(path, name, methods, **bindings):
    """Load real methods while stubbing transport and tracing dependencies."""
    source = ast.parse(path.read_bytes(), filename=str(path))
    cls = next(
        node for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    cls.bases = []
    cls.body = [
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    for method in cls.body:
        method.decorator_list = [
            node for node in method.decorator_list
            if isinstance(node, ast.Name) and node.id == "staticmethod"
        ]
    preamble = ast.parse("from __future__ import annotations").body
    module = ast.Module(body=[*preamble, cls], type_ignores=[])
    bindings.update(
        _ks_store_probe=SimpleNamespace(mark=lambda *_: None),
        logger=logging.getLogger(__name__),
    )
    exec(compile(module, str(path), "exec"), bindings)
    return bindings[name]


TransferContext = load_class(
    ROOT / "v1/multiprocess/transfer_context/worker_transfer.py",
    "LMCacheDrivenTransferContext",
    {
        "_store_settled", "submit_store", "prune_settled_stores",
        "_prune_settled_stores_locked", "flush_inflight_stores",
    },
)
WorkerAdapter = load_class(
    ROOT / "integration/vllm/vllm_multi_process_adapter.py",
    "LMCacheMPWorkerAdapter",
    {"submit_store_request", "batched_submit_store_requests"},
    LMCacheDrivenTransferContext=TransferContext,
)


class StorePruneTest(unittest.TestCase):
    def setUp(self):
        context = self.context = TransferContext()
        context._inflight_lock = threading.Lock()
        context._mq_timeout = 5.0
        context._instance_id = 0
        context._device = SimpleNamespace(type="npu")
        context._event_backend = Mock()
        self.completed = self.make_future(True)
        self.pending = self.make_future(False)
        context._inflight_stores = [self.completed, self.pending]
        self.new_futures = [self.make_future(i == 0) for i in range(3)]
        self.seen = []
        context._req_client = SimpleNamespace(store=self.send_store)
        adapter = self.adapter = WorkerAdapter()
        adapter.transfer_ctx = context
        adapter.parallel_strategy = SimpleNamespace(vllm_worker_id=0, pp_stage_local=True)
        adapter._ensure_heartbeat_started = lambda: None
        adapter._create_key = lambda *_, request_id, **options: request_id
        adapter._block_ids_per_group = lambda op: [[op.start]]
        adapter.is_healthy = adapter.is_kv_writer = True
        adapter.lazy_offload = adapter._kv_events_enabled = False
        adapter.blocks_in_chunk = 1
        adapter.kv_caches, adapter.store_futures, adapter.store_events = {}, {}, {}
        self.op = SimpleNamespace(token_ids=[1], start=0, end=1)
        self.event = object()

    @staticmethod
    def make_future(ready):
        return SimpleNamespace(query=Mock(return_value=ready), wait=Mock(return_value=True))

    def send_store(self, key, instance_id, block_ids, event_handle):
        self.assertEqual(key, str(len(self.seen)))
        self.assertEqual(block_ids, [[0]])
        self.assertIs(event_handle, self.context._event_backend.export_event.return_value)
        future = self.new_futures[len(self.seen)]
        self.seen.append(list(self.context._inflight_stores))
        return SimpleNamespace(to_device_future=lambda **_: future)

    def test_pp0_prunes_once_and_tracks_each_store_before_next_send(self):
        self.adapter.batched_submit_store_requests(
            ["0", "1", "2"], [self.op] * 3, self.event
        )

        self.assertEqual(self.seen, [
            [self.pending],
            [self.pending, self.new_futures[0]],
            [self.pending, *self.new_futures[:2]],
        ])
        self.completed.query.assert_called_once()
        self.pending.query.assert_called_once()
        for index, future in enumerate(self.new_futures):
            future.query.assert_not_called()
            self.assertIs(self.adapter.store_futures[str(index)], future)
            self.assertIs(self.adapter.store_events[str(index)], self.event)

        self.context.flush_inflight_stores()
        self.assertEqual(self.context._inflight_stores, [])
        self.completed.wait.assert_not_called()
        self.new_futures[0].wait.assert_not_called()
        for future in [self.pending, *self.new_futures[1:]]:
            future.wait.assert_called_once_with(timeout=5.0)

    def test_pp1_keeps_per_request_pruning(self):
        self.adapter.parallel_strategy.vllm_worker_id = 16
        self.adapter.batched_submit_store_requests(
            ["0", "1", "2"], [self.op] * 3, self.event
        )
        self.assertEqual(self.pending.query.call_count, 3)
        self.new_futures[0].query.assert_called_once()
        self.assertNotIn(self.new_futures[0], self.context._inflight_stores)

    def test_single_store_keeps_per_request_pruning(self):
        self.adapter.submit_store_request("0", self.op, self.event)
        self.completed.query.assert_called_once()
        self.assertEqual(
            self.context._inflight_stores, [self.pending, self.new_futures[0]]
        )


if __name__ == "__main__":
    unittest.main()
