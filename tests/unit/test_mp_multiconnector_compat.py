# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LMCacheMPConnector compatibility under vLLM MultiConnector.

Executes the actual LMCacheMPConnector methods from source without requiring
vLLM or a device runtime to be installed.
"""

# Standard
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import ast
import unittest

ROOT = Path(__file__).resolve().parents[2]
CONNECTOR_PATH = ROOT / "lmcache/integration/vllm/lmcache_mp_connector.py"


def _load_connector_method(name: str, namespace: dict[str, Any]) -> Any:
    tree = ast.parse(CONNECTOR_PATH.read_text(encoding="utf-8"))
    cls_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LMCacheMPConnector"
    )
    fn_node = next(
        node
        for node in cls_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[fn_node], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), str(CONNECTOR_PATH), "exec"),
        namespace,
    )
    return namespace[name]


class _FakeTracker:
    def __init__(self, allocated_block_ids: dict[int, list[int]] | None = None):
        self.allocated_block_ids: dict[int, list[int]] = (
            dict(allocated_block_ids) if allocated_block_ids else {}
        )
        self.appended: list[tuple[tuple[list[int], ...], int]] = []
        self.scheduled_tokens = 0

    def append_block_ids(
        self,
        new_block_ids: tuple[list[int], ...],
        relocation_window: int = 0,
    ) -> None:
        self.appended.append((new_block_ids, relocation_window))
        for group_idx, block_ids in enumerate(new_block_ids):
            if block_ids:
                self.allocated_block_ids.setdefault(group_idx, []).extend(block_ids)

    def increase_num_scheduled_tokens(self, num_tokens: int) -> None:
        self.scheduled_tokens += num_tokens


class _FakeMetadata:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    def add_request_metadata(self, request_meta: Any) -> None:
        self.requests.append(request_meta)


class TestMPMultiConnectorCompat(unittest.TestCase):
    def setUp(self) -> None:
        self._handshake_fn = _load_connector_method(
            "set_xfer_handshake_metadata_pp_aware",
            {"Any": Any, "Mapping": Mapping},
        )
        self._store_meta_calls: list[tuple[_FakeTracker, int, dict[int, int], bool]] = []

        def _fake_get_store_metadata(
            tracker: _FakeTracker,
            lmcache_tokens_per_chunk: int,
            group_tokens_per_block: dict[int, int],
            save_decode_cache: bool = False,
        ) -> Any:
            self._store_meta_calls.append(
                (
                    tracker,
                    lmcache_tokens_per_chunk,
                    group_tokens_per_block,
                    save_decode_cache,
                )
            )
            if not tracker.allocated_block_ids:
                return None
            return SimpleNamespace(
                direction="STORE",
                block_ids=dict(tracker.allocated_block_ids),
            )

        self._process_new_requests_fn = _load_connector_method(
            "_process_new_requests",
            {
                "Any": Any,
                "SchedulerOutput": Any,
                "LMCacheMPConnectorMetadata": Any,
                "LMCacheMPRequestMetadata": SimpleNamespace(
                    GetStoreMetadata=_fake_get_store_metadata
                ),
            },
        )

    def _run_process_new_requests(
        self,
        *,
        existing_blocks: dict[int, list[int]] | None,
        new_block_ids: tuple[list[int], ...],
        can_store: bool = True,
        mamba_relocation_window: int = 3,
    ) -> tuple[_FakeTracker, _FakeMetadata]:
        tracker = _FakeTracker(existing_blocks)
        metadata = _FakeMetadata()
        connector = SimpleNamespace(
            scheduler_adapter=SimpleNamespace(lmcache_tokens_per_chunk=256),
            _get_request_tracker=lambda _req_id: tracker,
            _mamba_relocation_window=mamba_relocation_window,
            _can_store=can_store,
            _group_tokens_per_block={0: 128},
            _save_decode_cache=False,
            lazy_offload=False,
        )
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req_id="req-1", block_ids=new_block_ids)
            ],
            num_scheduled_tokens={"req-1": 256},
        )
        self._process_new_requests_fn(connector, scheduler_output, metadata)
        return tracker, metadata

    def test_pp_handshake_accepts_tuple_keyed_metadata_without_mutation(self) -> None:
        pp0_meta = object()
        pp1_meta = object()
        metadata = {(0, 0): pp0_meta, (1, 0): pp1_meta}
        snapshot = dict(metadata)

        result = self._handshake_fn(SimpleNamespace(), metadata)

        self.assertIsNone(result)
        self.assertEqual(metadata, snapshot)

    def test_pp_handshake_accepts_nested_metadata_without_mutation(self) -> None:
        pp0_meta = object()
        pp1_meta = object()
        metadata = {0: {0: pp0_meta}, 1: {0: pp1_meta}}
        snapshot = {stage: dict(ranks) for stage, ranks in metadata.items()}

        result = self._handshake_fn(SimpleNamespace(), metadata)

        self.assertIsNone(result)
        self.assertEqual(metadata, snapshot)

    def test_non_selected_writer_recovers_initial_blocks_and_stores(self) -> None:
        tracker, metadata = self._run_process_new_requests(
            existing_blocks=None,
            new_block_ids=([4, 5],),
            can_store=True,
            mamba_relocation_window=3,
        )

        self.assertEqual(tracker.appended, [(([4, 5],), 3)])
        self.assertEqual(tracker.allocated_block_ids, {0: [4, 5]})
        self.assertEqual(tracker.scheduled_tokens, 256)
        self.assertEqual(len(metadata.requests), 1)
        self.assertEqual(metadata.requests[0].block_ids, {0: [4, 5]})

    def test_selected_loader_blocks_are_preserved_without_duplication(self) -> None:
        tracker, metadata = self._run_process_new_requests(
            existing_blocks={0: [1, 2]},
            new_block_ids=([1, 2],),
            can_store=True,
        )

        self.assertEqual(tracker.appended, [])
        self.assertEqual(tracker.allocated_block_ids, {0: [1, 2]})
        self.assertEqual(len(metadata.requests), 1)
        self.assertEqual(metadata.requests[0].block_ids, {0: [1, 2]})

    def test_empty_new_request_blocks_are_not_fabricated(self) -> None:
        tracker, metadata = self._run_process_new_requests(
            existing_blocks=None,
            new_block_ids=(),
            can_store=True,
        )

        self.assertEqual(tracker.appended, [])
        self.assertEqual(tracker.allocated_block_ids, {})
        self.assertEqual(metadata.requests, [])

    def test_consumer_only_connector_does_not_schedule_store(self) -> None:
        tracker, metadata = self._run_process_new_requests(
            existing_blocks=None,
            new_block_ids=([7, 8],),
            can_store=False,
        )

        self.assertEqual(tracker.allocated_block_ids, {0: [7, 8]})
        self.assertEqual(metadata.requests, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
