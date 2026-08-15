"""Small analytic checks for exact thermal references."""

import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import exact_diagonalization as exact


class ExactDiagonalizationTest(unittest.TestCase):
    def test_single_spin_z_thermal_moments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "H.txt"
            path.write_text("-1 1 Z\n")
            result = exact.exact_thermal_observables(path, 1, 0.7)
        self.assertAlmostEqual(result["energy"], -math.tanh(0.7), places=12)
        self.assertAlmostEqual(result["specific_heat"],
                               0.7 ** 2 / math.cosh(0.7) ** 2, places=12)

    def test_split_gamma_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            fixed = Path(directory) / "fixed.txt"
            varying = Path(directory) / "gamma.txt"
            fixed.write_text("")
            varying.write_text("-1 1 Z\n")
            result = exact.exact_split_thermal_observables(fixed, varying, 2.0, 1, 0.7)
        self.assertAlmostEqual(result["energy"], -2 * math.tanh(1.4), places=12)


if __name__ == "__main__":
    unittest.main()
