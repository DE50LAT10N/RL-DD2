from __future__ import annotations

import argparse
import unittest

from scripts.check_qrdqn_ready import _threshold_failures


class CheckQRDQNReadyTest(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            min_win_rate=0.35,
            min_composite_score=0.0,
            min_survival_rate=0.60,
            max_bad_pass_rate=0.08,
            max_wasted_heal_rate=0.08,
            max_critical_heal_miss_rate=0.35,
            min_kill_confirm_rate=0.25,
        )

    def test_threshold_failures_pass_for_good_metrics(self) -> None:
        failures = _threshold_failures(
            self._args(),
            {"win_rate": 0.55, "composite_score": 12.0, "survival_rate": 0.9},
            {
                "bad_pass_rate": 0.01,
                "wasted_heal_rate": 0.02,
                "critical_heal_miss_rate": 0.1,
                "kill_confirm_opportunities": 4,
                "kill_confirm_rate": 0.5,
            },
        )

        self.assertEqual(failures, [])

    def test_threshold_failures_report_each_metric(self) -> None:
        failures = _threshold_failures(
            self._args(),
            {"win_rate": 0.1, "composite_score": -3.0, "survival_rate": 0.2},
            {
                "bad_pass_rate": 0.5,
                "wasted_heal_rate": 0.2,
                "critical_heal_miss_rate": 0.8,
                "kill_confirm_opportunities": 2,
                "kill_confirm_rate": 0.0,
            },
        )

        self.assertIn("win_rate<0.35", failures)
        self.assertIn("composite_score<0.0", failures)
        self.assertIn("survival_rate<0.6", failures)
        self.assertIn("bad_pass_rate>0.08", failures)
        self.assertIn("wasted_heal_rate>0.08", failures)
        self.assertIn("critical_heal_miss_rate>0.35", failures)
        self.assertIn("kill_confirm_rate<0.25", failures)


if __name__ == "__main__":
    unittest.main()
