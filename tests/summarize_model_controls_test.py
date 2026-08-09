"""Deterministic tests for matched model-control summaries."""

import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import summarize_model_controls as controls


class ModelControlSummaryTest(unittest.TestCase):
    def test_bootstrap_interval_excludes_one_for_uniform_speedup(self):
        interval = controls.bootstrap_interval([4.0, 4.1, 4.2, 4.3], 1000, random.Random(7))
        self.assertGreater(interval[0], 1.0)

    def test_energy_agreement_uses_five_combined_errors(self):
        point = lambda mean: {"derived": {"energy": {"mean": mean, "standard_error": 0.1}}}
        self.assertTrue(controls.energy_agrees(point(0.0), point(0.6)))
        self.assertFalse(controls.energy_agrees(point(0.0), point(1.0)))


if __name__ == "__main__":
    unittest.main()
