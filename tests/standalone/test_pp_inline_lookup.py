# SPDX-License-Identifier: Apache-2.0
"""Exercise inline lookup_with_result aggregation without device imports."""

from types import SimpleNamespace
import unittest

from test_lookup_deadline_contract import Reply, methods, scheduler


def inline_stages(left, right, *, enabled=True):
    obj, _clock, _ = scheduler()
    obj._pp_query_fastpath = enabled
    obj._server_urls = ["early", "late"]
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
