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

    def test_mla_tp_broadcast_retrieve_packs_and_copies_strided_and_int8_planes(
        self,
    ) -> None:
        import sys
        import types
        import torch

        bcast_node = _extract_method(
            CONNECTOR_PATH, "LMCacheMPConnector", "_broadcast_retrieved_blocks_tp"
        )
        load_node = _extract_method(
            CONNECTOR_PATH, "LMCacheMPConnector", "start_load_kv"
        )

        # Simulate TP=2 broadcast from src=0 (writer) -> dst=1 (non-writer)
        src_full = torch.randn((16, 4, 1, 12), dtype=torch.float32)
        src_kvc = src_full[:, :, :, :8]
        src_kpe = src_full[:, :, :, 8:]
        src_li8 = torch.randint(-64, 64, (16, 4, 1, 6), dtype=torch.int8)

        dst_full = torch.zeros_like(src_full)
        dst_kvc = dst_full[:, :, :, :8]
        dst_kpe = dst_full[:, :, :, 8:]
        dst_li8 = torch.zeros_like(src_li8)

        wire_buffers: list[torch.Tensor] = []

        class FakeTPGroupWriter:
            def broadcast(self, buf: torch.Tensor, src: int = 0) -> torch.Tensor:
                self.last_src = src
                wire_buffers.append(buf.clone())
                return buf

        class FakeTPGroupNonWriter:
            def broadcast(self, buf: torch.Tensor, src: int = 0) -> torch.Tensor:
                buf.copy_(wire_buffers.pop(0))
                return buf

        fake_vllm_dist = types.ModuleType("vllm.distributed")
        saved_mods = {
            k: sys.modules.get(k) for k in ("vllm", "vllm.distributed")
        }
        sys.modules["vllm"] = types.ModuleType("vllm")
        sys.modules["vllm.distributed"] = fake_vllm_dist
        try:
            ns = {
                "torch": torch,
                "Any": object,
                "LMCacheMPConnectorMetadata": SimpleNamespace,
                "nullcontext": contextlib.nullcontext,
            }
            bcast_fn = _compile_method(bcast_node, ns)
            load_fn = _compile_method(load_node, ns)

            # skip_first_n_tokens=4 (bs=4 -> skip first 1 APC block id=1; broadcast blocks 3 and 7)
            ops = [
                SimpleNamespace(
                    skip_first_n_tokens=4,
                    flat_block_ids=[1, 3, 7],
                ),
                SimpleNamespace(
                    skip_first_n_tokens=0,
                    flat_block_ids=[7, 9],  # duplicate block 7 deduplicated!
                ),
            ]
            meta = SimpleNamespace(
                delay_free_req_ids=set(),
                requests=[
                    SimpleNamespace(
                        direction="RETRIEVE",
                        request_id="r1",
                        op=ops[0],
                        cache_salt="",
                        request_configs=None,
                    ),
                    SimpleNamespace(
                        direction="RETRIEVE",
                        request_id="r2",
                        op=ops[1],
                        cache_salt="",
                        request_configs=None,
                    ),
                ],
            )

            writer_submits: list[tuple] = []
            fake_vllm_dist.get_tp_group = lambda: FakeTPGroupWriter()
            writer_conn = SimpleNamespace(
                _get_connector_metadata=lambda: meta,
                _delay_free_req_ids=set(),
                _in_step_sync_retrieve=True,
                _mla_tp_broadcast_retrieve=True,
                _pp_rank=0,
                _trace_span=None,
                worker_adapter=SimpleNamespace(
                    is_kv_writer=True,
                    kv_caches={"l0": (src_kvc, src_kpe), "l1_idx": [src_li8]},
                    parallel_strategy=SimpleNamespace(
                        mla_only=True, tp_size=16, dcp_size=1
                    ),
                    create_recorded_event=lambda: "ev0",
                    batched_submit_retrieve_requests=lambda *a, **kw: writer_submits.append(
                        (a, kw)
                    ),
                    _ensure_heartbeat_started=lambda: None,
                ),
            )
            writer_conn._broadcast_retrieved_blocks_tp = (
                lambda o: bcast_fn(writer_conn, o)
            )
            load_fn(writer_conn, None)
            self.assertEqual(len(writer_submits), 1)
            self.assertEqual(writer_submits[0][1], {
                "cache_salts": ["", ""],
                "request_configs_list": [None, None],
                "sync_on_stream": True,
            })
            self.assertEqual(len(wire_buffers), 1)

            nonwriter_submits: list[tuple] = []
            hb_called: list[bool] = []
            fake_vllm_dist.get_tp_group = lambda: FakeTPGroupNonWriter()
            nonwriter_conn = SimpleNamespace(
                _get_connector_metadata=lambda: meta,
                _delay_free_req_ids=set(),
                _in_step_sync_retrieve=True,
                _mla_tp_broadcast_retrieve=True,
                _pp_rank=0,
                _trace_span=None,
                worker_adapter=SimpleNamespace(
                    is_kv_writer=False,
                    kv_caches={"l0": (dst_kvc, dst_kpe), "l1_idx": [dst_li8]},
                    parallel_strategy=SimpleNamespace(
                        mla_only=True, tp_size=16, dcp_size=1
                    ),
                    create_recorded_event=lambda: "ev1",
                    batched_submit_retrieve_requests=lambda *a, **kw: nonwriter_submits.append(
                        (a, kw)
                    ),
                    _ensure_heartbeat_started=lambda: hb_called.append(True),
                ),
            )
            nonwriter_conn._broadcast_retrieved_blocks_tp = (
                lambda o: bcast_fn(nonwriter_conn, o)
            )
            load_fn(nonwriter_conn, None)

            # Non-writer sends ZERO server retrieve RPCs, starts heartbeat, and receives exact blocks [3, 7, 9]!
            self.assertEqual(nonwriter_submits, [])
            self.assertEqual(hb_called, [True])
            self.assertTrue(torch.equal(dst_full[[3, 7, 9]], src_full[[3, 7, 9]]))
            self.assertTrue(torch.equal(dst_li8[[3, 7, 9]], src_li8[[3, 7, 9]]))
            # Skipped APC block 1 untouched on dst!
            self.assertEqual(float(dst_full[1].abs().sum()), 0.0)
        finally:
            for k, v in saved_mods.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_single_mla_reader_sets_num_kv_readers_to_one(self) -> None:
        tree = ast.parse(ADAPTER_PATH.read_text(encoding="utf-8"))
        ps_cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ParallelStrategy"
        )
        from dataclasses import dataclass
        import math

        mod = ast.Module(body=[ps_cls], type_ignores=[])
        ns = {"dataclass": dataclass, "math": math}
        exec(compile(ast.fix_missing_locations(mod), "<ps>", "exec"), ns)
        ParallelStrategy = ns["ParallelStrategy"]

        ps_default = ParallelStrategy(
            mla_only=True, vllm_world_size=32, vllm_worker_id=0,
            tp_size=16, pp_size=2, n_servers=2, pp_stage_local=True,
        )
        self.assertEqual(ps_default.num_kv_readers, 16)

        ps_bcast = ParallelStrategy(
            mla_only=True, vllm_world_size=32, vllm_worker_id=0,
            tp_size=16, pp_size=2, n_servers=2, pp_stage_local=True,
            single_mla_reader=True,
        )
        self.assertEqual(ps_bcast.num_kv_readers, 1)

        ps_dcp2 = ParallelStrategy(
            mla_only=True, vllm_world_size=32, vllm_worker_id=0,
            tp_size=16, pp_size=2, n_servers=2, dcp_size=2, pp_stage_local=True,
            single_mla_reader=True,
        )
        self.assertEqual(ps_dcp2.num_kv_readers, 8)

    def test_apc_covered_lookup_skip_and_store_dedup(self) -> None:
        import time

        fn_node = _extract_method(
            CONNECTOR_PATH, "LMCacheMPConnector", "get_num_new_matched_tokens"
        )
        on_new_node = _extract_method(
            CONNECTOR_PATH, "LMCacheMPConnector", "on_new_request"
        )
        ns = {
            "time": time,
            "LMCacheMPRequestState": SimpleNamespace(
                PREFETCHING="PREFETCHING",
                READY="READY",
                BYPASS_LMCACHE="BYPASS_LMCACHE",
            ),
            "RequestStatus": SimpleNamespace(PREEMPTED="PREEMPTED"),
            "KVConnectorRole": SimpleNamespace(SCHEDULER="SCHEDULER", WORKER="WORKER"),
            "logger": SimpleNamespace(
                warning=lambda *_a, **_k: None,
                info=lambda *_a, **_k: None,
                debug=lambda *_a, **_k: None,
            ),
        }
        get_matched = _compile_method(fn_node, ns)
        on_new = _compile_method(on_new_node, ns)

        lookups_sent: list[str] = []
        cleaned: list[str] = []

        def make_tracker(prompt_len: int = 8192):
            return SimpleNamespace(
                state="PREFETCHING",
                num_lmcache_hit_tokens=0,
                num_vllm_hit_tokens=0,
                num_stored_tokens=0,
                borrowed_restore_submitted=False,
                allocated_block_ids={},
                lookup_started_at=None,
                cache_salt="",
                request_configs=None,
                get_token_ids=lambda: [1] * prompt_len,
            )

        tracker_full = make_tracker(8192)
        conn = SimpleNamespace(
            role="SCHEDULER",
            _eager_prefetch=True,
            _skip_lookup_when_apc_covered=True,
            _reserve_last_token_for_lookup=True,
            _hit_alignment_tokens=128,
            _read_policy="legacy",
            _in_step_sync_retrieve=True,
            _get_or_create_request_tracker=lambda _r: tracker_full,
            _connector_stats=SimpleNamespace(record_lookup=lambda _d: None),
            scheduler_adapter=SimpleNamespace(
                lmcache_tokens_per_chunk=512,
                maybe_submit_lookup_request=lambda req_id, **_k: (lookups_sent.append(req_id) or True),
                check_lookup_result=lambda _id: 7168,
                cleanup_lookup_result=lambda req_id: cleaned.append(req_id),
            ),
        )
        req = SimpleNamespace(
            request_id="req-apc-full",
            num_computed_tokens=7808,
            all_token_ids=[1] * 8192,
            status="WAITING",
            resumable=False,
        )
        # 1. on_new_request must NOT fire eager lookup when _skip_lookup_when_apc_covered=True
        on_new(conn, req)
        self.assertEqual(lookups_sent, [])

        # 2. num_computed_tokens=7808 >= aligned_lookup_end(8191//512*512=7680): skip lookup completely!
        matched, is_async = get_matched(conn, req, 7808)
        self.assertEqual((matched, is_async), (0, False))
        self.assertEqual(lookups_sent, [])
        self.assertEqual(tracker_full.num_stored_tokens, 7680)
        self.assertEqual(tracker_full.num_vllm_hit_tokens, 7808)

        # 3. single_prefix partial APC (7296 < 7680) with L1 hit=7168 under transfer_borrow:
        #    tracker.num_stored_tokens MUST advance to 7168 so [0..7168) is NOT re-stored!
        tracker_sp = make_tracker(8192)
        conn._get_or_create_request_tracker = lambda _r: tracker_sp
        conn._read_policy = "transfer_borrow"
        req_sp = SimpleNamespace(
            request_id="req-single-prefix",
            num_computed_tokens=7296,
            all_token_ids=[1] * 8192,
            status="WAITING",
            resumable=False,
        )
        matched_sp, is_async_sp = get_matched(conn, req_sp, 7296)
        self.assertEqual((matched_sp, is_async_sp), (0, False))
        self.assertEqual(lookups_sent, ["req-single-prefix"])
        self.assertEqual(tracker_sp.num_stored_tokens, 7168)


if __name__ == "__main__":
    unittest.main(verbosity=2)
