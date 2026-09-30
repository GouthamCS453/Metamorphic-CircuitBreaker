"""Tests for the additive fallback controller.

Run: python -m unittest fallback.test_controller
"""

import unittest

from fallback.controller import FallbackAction, FallbackController


class FallbackControllerTests(unittest.TestCase):
    def test_closed_is_normal(self):
        c = FallbackController()
        d = c.evaluate("CLOSED", now=0)
        self.assertEqual(d.action, FallbackAction.NORMAL)

    def test_half_open_requests_takeover(self):
        c = FallbackController()
        d = c.evaluate("HALF_OPEN", now=0)
        self.assertEqual(d.action, FallbackAction.TAKEOVER_REQUESTED)
        self.assertLessEqual(d.countdown_s, 10.0)

    def test_unoccupied_half_open_pulls_over_immediately(self):
        c = FallbackController()
        d = c.evaluate("HALF_OPEN", driver_present=False, now=0)
        self.assertEqual(d.action, FallbackAction.SAFE_PULL_OVER)

    def test_unoccupied_open_pulls_over_immediately(self):
        c = FallbackController()
        d = c.evaluate("OPEN", driver_present=False, now=0)
        self.assertEqual(d.action, FallbackAction.SAFE_PULL_OVER)

    def test_unacknowledged_open_escalates_after_timeout(self):
        c = FallbackController()
        first = c.evaluate("OPEN", driver_present=True, now=10)
        self.assertEqual(first.action, FallbackAction.TAKEOVER_REQUESTED)
        second = c.evaluate("OPEN", driver_present=True, now=20.01)
        self.assertEqual(second.action, FallbackAction.SAFE_PULL_OVER)

    def test_acknowledgement_prevents_pull_over(self):
        c = FallbackController()
        c.evaluate("OPEN", driver_present=True, now=0)
        c.acknowledge_takeover()
        d = c.evaluate("OPEN", driver_present=True, now=1)
        self.assertEqual(d.action, FallbackAction.TAKEOVER_REQUESTED)
        self.assertTrue(d.acknowledged)


if __name__ == "__main__":
    unittest.main()
