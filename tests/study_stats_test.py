"""Deterministic statistical tests for the study analysis layer."""

import math
import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import study_stats as stats


class StudyStatsTest(unittest.TestCase):
    def test_ratio_mean_and_iat(self):
        rows = [{"sign": 1.0, "signed_obs_2": value} for value in (1.0, 2.0, 3.0, 4.0)]
        self.assertEqual(stats.ratio_mean(rows, "obs_2"), 2.5)
        self.assertEqual(stats.ratio_trace(rows, "obs_2"), [1.0, 1.5, 2.0, 2.5])
        self.assertTrue(stats.integrated_autocorrelation([1, -1] * 100)["valid"])

    def test_correlated_trace_has_larger_iat(self):
        rng = random.Random(123)
        independent = [rng.gauss(0, 1) for _ in range(3000)]
        correlated = []
        value = 0.0
        for _ in range(3000):
            value = 0.95 * value + rng.gauss(0, 1)
            correlated.append(value)
        self.assertGreater(stats.integrated_autocorrelation(correlated)["iat"],
                           stats.integrated_autocorrelation(independent)["iat"] * 5)

    def test_rank_rhat(self):
        rng = random.Random(7)
        good = [[rng.gauss(0, 1) for _ in range(1000)] for _ in range(4)]
        bad = [list(chain) for chain in good]
        bad[-1] = [value + 3 for value in bad[-1]]
        self.assertLess(stats.split_rank_normalized_rhat(good), 1.01)
        self.assertGreater(stats.split_rank_normalized_rhat(bad), 1.05)

    def test_joint_jackknife_derived_observables(self):
        rows = []
        for index in range(1000):
            energy = -2.0 + (0.1 if index % 2 else -0.1)
            row = {"sign": 1.0}
            values = {"obs_2": energy, "obs_3": energy * energy,
                      "obs_6": -1.0, "obs_13": 3.0, "obs_14": 1.0}
            for name, value in values.items():
                row[f"signed_{name}"] = value
            rows.append(row)
        result = stats.joint_block_jackknife(rows, beta=2.0, spins=4, block_length=10)
        self.assertAlmostEqual(result["energy"]["mean"], -2.0)
        self.assertAlmostEqual(result["specific_heat"]["mean"], 0.04, places=10)
        self.assertAlmostEqual(result["energy_susceptibility"]["mean"], 1.0)
        self.assertAlmostEqual(result["fidelity_susceptibility"]["mean"], 0.5)
        self.assertAlmostEqual(result["effective_gap"]["mean"], 1.0)
        self.assertEqual(result["energy"]["blocks"], 100)

    def test_beta_convergence_requires_two_stable_increments(self):
        rows = []
        for beta, value in ((2, 1.2), (4, 1.0015), (8, 1.001), (16, 1.0005)):
            rows.append({"method": "fixed", "protocol": "cheap", "L": 4, "lambda": 3.0,
                         "seed": 1, "beta": beta, "beta_over_L": beta / 4,
                         "energy_density": value, "energy_density_se": 0.01})
        result = stats.beta_convergence(rows, "energy_density")[0]
        self.assertTrue(result["converged"])
        self.assertIsNone(result["next_beta_over_L"])


if __name__ == "__main__":
    unittest.main()
