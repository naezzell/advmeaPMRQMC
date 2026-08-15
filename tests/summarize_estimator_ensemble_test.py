"""Deterministic bootstrap helper tests."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import summarize_estimator_ensemble as summary


class EstimatorEnsembleTest(unittest.TestCase):
    def test_quantile_interpolation(self):
        self.assertEqual(summary.quantile([1, 2, 3], 0.5), 2)
        self.assertEqual(summary.quantile([1, 3], 0.25), 1.5)


if __name__ == "__main__":
    unittest.main()
