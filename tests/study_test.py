"""Standard-library tests for the campaign planning layer."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import study


class StudyTest(unittest.TestCase):
    def test_square_edges(self):
        self.assertEqual(len(study.square_edges(3, False)), 12)
        self.assertEqual(len(study.square_edges(3, True)), 18)
        self.assertEqual(study.square_edges(2, False), [(1, 2), (1, 3), (2, 4), (3, 4)])

    def test_standard_and_rotated_specs(self):
        model = study.SquareTFIM()
        standard = model.build(3, 3.044, True, "standard")
        self.assertEqual(standard.spins, 9)
        self.assertEqual(len(standard.fixed_terms), 18)
        self.assertEqual(len(standard.lambda_terms), 9)
        self.assertIn("Z", standard.fixed_terms[0])
        self.assertIn("X", standard.lambda_terms[0])
        self.assertIn("global_z2", standard.supported_moves)
        rotated = model.build(3, 1.0, True, "rotated_parity")
        self.assertEqual(rotated.parity, 1)
        self.assertIn("X", rotated.fixed_terms[0])
        self.assertIn("Z", rotated.lambda_terms[0])

    def test_qcpt_schedules_end_at_target(self):
        for name in ("pure_beta", "diagonal_p0_5", "diagonal_p1", "diagonal_p2", "classical_dogleg"):
            schedule = study.qcpt_schedule(name, 8.0, 3.044)
            self.assertEqual(len(schedule), 4)
            self.assertAlmostEqual(schedule[-1]["beta"], 8.0)
            self.assertAlmostEqual(schedule[-1]["gamma"], 3.044)
            self.assertTrue(all(0 <= point["tau"] <= point["beta"] for point in schedule))

    def test_run_identity_is_order_independent(self):
        self.assertEqual(study.run_identity({"a": 1, "b": 2}),
                         study.run_identity({"b": 2, "a": 1}))
        self.assertNotEqual(study.run_identity({"a": 1}), study.run_identity({"a": 2}))

    def test_plan_cli_is_deterministic(self):
        config = ROOT / "benchmarking_tests" / "configs" / "desktop_smoke.json"
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            command = [sys.executable, str(ROOT / "experiments" / "study.py"),
                       "plan", "--config", str(config)]
            subprocess.run(command + ["--output", first], check=True)
            subprocess.run(command + ["--output", second], check=True)
            with (Path(first) / "plan.csv").open(newline="") as stream:
                first_rows = list(csv.DictReader(stream))
            with (Path(second) / "plan.csv").open(newline="") as stream:
                second_rows = list(csv.DictReader(stream))
            self.assertEqual(first_rows, second_rows)
            self.assertEqual(len(first_rows), 1)
            self.assertEqual(first_rows[0]["status"], "planned")

    def test_parameter_protocols(self):
        base = {field: "" for field in study.PLAN_FIELDS}
        base.update({"Tsteps": "10", "steps": "100", "steps_per_measurement": "1",
                     "beta": "2", "lambda": "3", "parity": "0", "qmax": "64",
                     "nbins": "10", "seed": "1000"})
        base["protocol"] = "cheap"
        cheap = study.parameter_text(base)
        self.assertIn("MEASURE_H2", cheap)
        self.assertNotIn("MEASURE_HOFFDIAG_FINT", cheap)
        self.assertIn("RESUME_CALCULATION", cheap)
        base["protocol"] = "advanced"
        advanced = study.parameter_text(base)
        self.assertIn("MEASURE_HOFFDIAG_FINT", advanced)
        base["move"] = "global_z2"
        self.assertIn("TFIM_GLOBAL_Z2_MOVE", study.parameter_text(base))

    def test_committed_campaign_matrices_have_unique_ids(self):
        configs = ROOT / "benchmarking_tests" / "configs"
        expected = {"desktop_smoke.json": 1, "desktop_pilot.json": 264,
                    "historical_anchors.json": 44, "ceiling_probes.json": 20,
                    "model_move_controls.json": 96}
        for name, count in expected.items():
            config = json.loads((configs / name).read_text())
            rows = study.expand_matrix(config, "test-commit")
            self.assertEqual(len(rows), count, name)
            self.assertEqual(len({row["run_id"] for row in rows}), count, name)


if __name__ == "__main__":
    unittest.main()
