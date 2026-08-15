"""Tests for archived-paper comparison helpers."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import compare_historical as historical


class HistoricalComparisonTest(unittest.TestCase):
    def test_inverse_variance_combination(self):
        points = [
            {"derived": {"energy": {"mean": 1.0, "standard_error": 0.5}}},
            {"derived": {"energy": {"mean": 3.0, "standard_error": 1.0}}},
        ]
        result = historical.inverse_variance(points, "energy")
        self.assertAlmostEqual(result["mean"], 1.4)
        self.assertAlmostEqual(result["standard_error"], (1 / 5) ** 0.5)
        self.assertEqual(result["runs"], 2)

    def test_archive_row_requires_unique_coordinate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.csv"
            path.write_text("L,lam,h,h_std\n3,0.5,-1,0.1\n")
            row = historical.select_archive_row(path, 3, 0.5, 1.0)
            self.assertEqual(row["h"], "-1")
            with self.assertRaises(ValueError):
                historical.select_archive_row(path, 4, 0.5, 1.0)

    def test_four_by_four_profile_combines_energy_components(self):
        row = {"Hdiag_value": "-1", "Hoffdiag_value": "-3",
               "Hdiag_std": "0.3", "Hoffdiag_std": "0.4",
               "offdiag_ES_value": "2", "offdiag_ES_std": "0.2",
               "offdiag_FS_value": "1", "offdiag_FS_std": "0.1"}
        values = historical.archived_observables(row, "paper1_four_by_four")
        self.assertEqual(values["energy"]["mean"], -4.0)
        self.assertAlmostEqual(values["energy"]["standard_error"], 0.5)


if __name__ == "__main__":
    unittest.main()
