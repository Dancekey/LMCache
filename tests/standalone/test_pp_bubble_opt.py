# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU contract tests for PP bubble profiling & optimizations."""

from __future__ import annotations

import ast
import contextlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = ROOT / "lmcache/integration/vllm/vllm_multi_process_adapter.py"
CONNECTOR_PATH = ROOT / "lmcache/integration/vllm/lmcache_mp_connector.py"


def _extract_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    fn = ast.parse(ast.unparse(fn)).body[0]
    assert isinstance(fn, ast.FunctionDef)
    fn.decorator_list = []
    return fn


def _compile_method(node: ast.FunctionDef, ns: dict) -> object:
    mod = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), "<source>", "exec"), ns)
    return ns[node.name]


class FakeFuture:
    def __init__(self, ok: bool = True, settled: bool = False, raise_err: bool = False):
        self._ok = ok
        self._settled = settled
        self._raise_err = raise_err
        self.waited_streams: list[object] = []

    def wait_on_stream(self, stream: object, timeout: float | None = None) -> bool:
        self.waited_streams.append((stream, timeout))
        if self._raise_err:
            raise TimeoutError("simulated timeout")
        return self._ok

    def query(self) -> bool:
        return self._settled


class PPBubbleOptTests(unittest.TestCase):
    def test_pp_bubble_trace_emitter_disabled_and_enabled(self) -> None:
        import importlib.util
        import tempfile

        trace_file = ROOT / "lmcache/v1/multiprocess/pp_bubble_trace.py"

        def _load_fresh():
            spec = importlib.util.spec_from_file_location("_pp_bubble_trace_test", trace_file)
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        with tempfile.TemporaryDirectory() as tmp:
            old_dir = os.environ.pop("AGY_MP_TRACE_DIR", None)
            old_node = os.environ.pop("AGY_MP_TRACE_NODE", None)
            try:
                trace_mod = _load_fresh()
                self.assertFalse(trace_mod.is_enabled())
                with trace_mod.trace_span("noop.span"):
                    pass
                self.assertEqual(list(Path(tmp).glob("*.jsonl")), [])

                os.environ["AGY_MP_TRACE_DIR"] = tmp
                os.environ["AGY_MP_TRACE_NODE"] = "n137"
                trace_mod = _load_fresh()
                self.assertTrue(trace_mod.is_enabled())
                with trace_mod.trace_span("worker.wait_for_save", pp_rank=0):
                    pass
                if trace_mod._fd is not None:
                    os.close(trace_mod._fd)
                    trace_mod._fd = None
                files = list(Path(tmp).glob("mp-trace-n137-*.jsonl"))
                self.assertEqual(len(files), 1)
                rows = [
                    json.loads(line)
                    for line in files[0].read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["event"], "worker.wait_for_save")
                self.assertEqual(rows[0]["pp_rank"], 0)
                self.assertGreaterEqual(rows[0]["duration_ns"], 0)
            finally:
                if old_dir is None:
                    os.environ.pop("AGY_MP_TRACE_DIR", None)
                else:
                    os.environ["AGY_MP_TRACE_DIR"] = old_dir
                if old_node is None:
                    os.environ.pop("AGY_MP_TRACE_NODE", None)
                else:
                    os.environ["AGY_MP_TRACE_NODE"] = old_node

    def test_pp0_and_pp1_writers_both_enable_async_store_export(self) -> None:
        node = _extract_method(
            ADAPTER_PATH, "LMCacheMPWorkerAdapter", "_send_register_kv_caches_request"
        )

        class DummyDrivenCtx:
            def register_pp(self, *_args, **_kwargs):
                return SimpleNamespace(
                    instance_id=7,
                    tp_rank=0,
                    profile=SimpleNamespace(
                        locator="loc", layer_names=("l0",), mla_only=True
                    ),
                    read_policy="legacy",
                    cache_namespace="ns",
                )

        created_ctx: list[DummyDrivenCtx] = []

        def fake_create(*_args, **_kwargs):
            ctx = DummyDrivenCtx()
            created_ctx.append(ctx)
            return ctx

        ns = {
            "os": os,
            "create_transfer_context": fake_create,
            "LMCacheDrivenTransferContext": DummyDrivenCtx,
            "ExportQueue": lambda dev: ("export_queue", dev),
            "torch_dev": "npu_dev",
            "PPRegistrationAck": SimpleNamespace,
            "EngineType": SimpleNamespace(VLLM="vllm"),
        }
        fn = _compile_method(node, ns)

        old_env = os.environ.get("KS_STORE_ASYNC_EXPORT")
        os.environ["KS_STORE_ASYNC_EXPORT"] = "1"
        try:
            for worker_id, is_writer, expect_export in (
                (0, True, True),    # PP0 Rank 0 writer -> MUST get ExportQueue!
                (1, False, False),  # PP0 non-writer -> skipped
                (16, True, True),   # PP1 Rank 16 writer -> gets ExportQueue
            ):
                created_ctx.clear()
                owner = SimpleNamespace(
                    instance_id=7,
                    req_client=object(),
                    _mp_transfer_mode="lmcache_driven",
                    parallel_strategy=SimpleNamespace(
                        pp_stage_local=True, vllm_worker_id=worker_id
                    ),
                    is_kv_writer=is_writer,
                    _layout_hints={},
                    _pp_registration_info=SimpleNamespace(
                        tp_rank=0,
                        locator="loc",
                        layer_names=("l0",),
                        mla_only=True,
                        read_policy="legacy",
                    ),
                    _pp_base_model_name="glm5",
                    world_size=1,
                    _mq_timeout=10.0,
                    engine_group_infos=[],
                    _pp_profile=None,
                )
                fn(owner, {"l0": object()})
                has_export = hasattr(created_ctx[0], "_ks_store_exports")
                self.assertEqual(
                    has_export,
                    expect_export,
                    f"worker_id={worker_id} writer={is_writer} export={has_export}",
                )
        finally:
            if old_env is None:
                os.environ.pop("KS_STORE_ASYNC_EXPORT", None)
            else:
                os.environ["KS_STORE_ASYNC_EXPORT"] = old_env

    def test_in_step_sync_retrieve_orders_stream_and_handles_failures(self) -> None:
        batch_node = _extract_method(
            ADAPTER_PATH, "LMCacheMPWorkerAdapter", "batched_submit_retrieve_requests"
        )
        ns = {
            "torch_dev": SimpleNamespace(current_stream=lambda: "npu_stream_0"),
            "logger": SimpleNamespace(exception=lambda *_a, **_k: None),
        }
        batch_fn = _compile_method(batch_node, ns)

        fut_ok = FakeFuture(ok=True, settled=False)
        fut_fail = FakeFuture(ok=False, settled=True)
        futures_iter = iter([fut_ok, fut_fail])

        owner = SimpleNamespace(
            _ensure_heartbeat_started=lambda: None,
            is_healthy=True,
            transfer_ctx=SimpleNamespace(
                submit_retrieve=lambda *_a, **_k: next(futures_iter)
            ),
            _create_key=lambda *_a, **_k: "key",
            kv_caches={"l0": object()},
            _block_ids_per_group=lambda op: op.block_ids,
            blocks_in_chunk=4,
            _mq_timeout=30.0,
            error_block_ids=set(),
            _failed_request_ids=set(),
            _sync_retrieve_inflight=[],
            retrieve_futures={},
        )
        ops = [
            SimpleNamespace(
                token_ids=[1, 2, 3, 4],
                block_ids=[[10, 11]],
                flat_block_ids=[10, 11],
                start=0,
                end=4,
                skip_first_n_tokens=0,
            ),
            SimpleNamespace(
                token_ids=[5, 6, 7, 8],
                block_ids=[[20, 21]],
                flat_block_ids=[20, 21],
                start=0,
                end=4,
                skip_first_n_tokens=0,
            ),
        ]
        ev = object()
        batch_fn(owner, ["req-ok", "req-fail"], ops, ev, sync_on_stream=True)

        # Must NOT populate async retrieve_futures (avoids scheduler finished_recving leak)
        self.assertEqual(owner.retrieve_futures, {})
        # Both futures waited on current device stream
        self.assertEqual(fut_ok.waited_streams, [("npu_stream_0", 30.0)])
        self.assertEqual(fut_fail.waited_streams, [("npu_stream_0", 30.0)])
        # Successful sync retrieve retained until device event settles
        self.assertEqual(owner._sync_retrieve_inflight, [(fut_ok, ev)])
        # Failed sync retrieve recorded in error_block_ids for vLLM sync recompute
        self.assertEqual(owner.error_block_ids, {20, 21})
        self.assertEqual(owner._failed_request_ids, {"req-fail"})

    def test_connector_matched_tokens_returns_sync_flag_when_enabled(self) -> None:
        node = _extract_method(
            CONNECTOR_PATH, "LMCacheMPConnector", "get_num_new_matched_tokens"
        )
        ns = {
            "LMCacheMPRequestState": SimpleNamespace(
                BYPASS_LMCACHE="BYPASS", READY="READY", PREFETCHING="PREFETCH"
            ),
            "RequestStatus": SimpleNamespace(PREEMPTED="PREEMPTED"),
            "time": SimpleNamespace(monotonic=lambda: 100.0),
            "logger": SimpleNamespace(
                warning=lambda *_a, **_k: None,
                info=lambda *_a, **_k: None,
                debug=lambda *_a, **_k: None,
            ),
        }
        fn = _compile_method(node, ns)

        def make_connector(in_step_sync: bool):
            tracker = SimpleNamespace(
                state="PREFETCH",
                num_lmcache_hit_tokens=0,
                lookup_started_at=None,
                get_token_ids=lambda: list(range(512)),
                cache_salt="",
                request_configs=None,
                num_vllm_hit_tokens=0,
                num_stored_tokens=0,
            )
            return SimpleNamespace(
                _get_or_create_request_tracker=lambda _r: tracker,
                _reserve_last_token_for_lookup=False,
                _read_policy="legacy",
                _hit_alignment_tokens=128,
                _in_step_sync_retrieve=in_step_sync,
                _connector_stats=SimpleNamespace(record_lookup=lambda _d: None),
                scheduler_adapter=SimpleNamespace(
                    maybe_submit_lookup_request=lambda *_a, **_k: True,
                    check_lookup_result=lambda _rid: 256,
                    lmcache_tokens_per_chunk=256,
                ),
            )

        req = SimpleNamespace(
            request_id="r1",
            num_computed_tokens=0,
            status="WAITING",
            all_token_ids=list(range(512)),
        )
        self.assertEqual(fn(make_connector(False), req, 0), (256, True))
        self.assertEqual(fn(make_connector(True), req, 0), (256, False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
