"""Tests for matched estimator trace comparisons."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import compare_estimator_runs as compare


class CompareEstimatorRunsTest(unittest.TestCase):
    def write_run(self, root, run_id, value, measurement, wall):
        root.mkdir()
        with (root / "trace_stream.rank0.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "updates", "sign", "obs_0", "signed_obs_0", "measurement_seconds"))
            writer.writeheader()
            writer.writerow({"updates": 1, "sign": 1, "obs_0": value,
                             "signed_obs_0": value, "measurement_seconds": measurement})
        (root / "summary.json").write_text(json.dumps(
            {"run_id": run_id, "timings": {"simulation": wall}}))

    def test_correctness_and_speed_ratios(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_run(root / "fast", "f", 2.0, 1.0, 3.0)
            self.write_run(root / "slow", "s", 2.0, 4.0, 9.0)
            result = compare.compare(root / "fast", root / "slow")
        self.assertTrue(result["correctness_pass"])
        self.assertEqual(result["measurement_speedup"], 4.0)
        self.assertEqual(result["wall_speedup"], 3.0)


if __name__ == "__main__":
    unittest.main()
