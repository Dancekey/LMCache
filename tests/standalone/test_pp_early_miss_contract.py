# SPDX-License-Identifier: Apache-2.0
"""Verify PP zero-hit short-circuit on both inline and fallback STATUS paths."""

from types import SimpleNamespace
import unittest

from test_lookup_deadline_contract import Reply
from test_pp_inline_lookup import inline_stages


class EarlyMissTest(unittest.TestCase):
    def test_inline_zero_short_circuits_without_waiting_for_peer(self):
        pending = Reply(ready=False)
        adapter = inline_stages(Reply(0), pending)
        self.assertEqual(adapter.check_lookup_result("r"), 0)
        self.assertEqual(adapter._finished_lookup_results["r"], 0)
        self.assertIs(adapter._unacked_lookups["r"].futures["late"], pending)
        self.assertEqual(adapter.sent, [])

    def test_status_zero_short_circuits_without_waiting_for_peer(self):
        adapter = inline_stages(Reply(None), Reply(None))
        self.assertEqual(adapter.check_lookup_result("r"), 0)
        self.assertEqual(adapter._finished_lookup_results["r"], 0)
        self.assertFalse(adapter.replies["late"].ready)

    def test_disabled_waits_for_all_stages(self):
        adapter = inline_stages(Reply(None), Reply(None), enabled=False)
        self.assertIsNone(adapter.check_lookup_result("r"))
        adapter.replies["late"].ready = True
        self.assertEqual(adapter.check_lookup_result("r"), 0)


if __name__ == "__main__":
    unittest.main()
