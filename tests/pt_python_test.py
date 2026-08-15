import importlib.util
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name, relative_path):
    specification = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


gap = load("pt_convergence_gap", "experiments/pt_convergence_gap.py")
plot = load("plot_pt_convergence_gap", "experiments/plot_pt_convergence_gap.py")
loop = load("pt_benchmark_loop", "experiments/pt_benchmark_loop.py")
minimal = load("validate_pt_minimal", "experiments/validate_pt_minimal.py")
qcpt = load("validate_qcpt", "experiments/validate_qcpt.py")
edge = load("validate_qcpt_edge_cases", "experiments/validate_qcpt_edge_cases.py")
anneal_driver = load("beta_anneal_driver", "experiments/beta_anneal_driver.py")


class SignedTraceTest(unittest.TestCase):
    def test_cumulative_ratio_uses_signed_numerator(self):
        trace = [(10, 2.0, 1.0), (20, -6.0, -1.0), (30, 4.0, 1.0)]
        ratios = gap.cumulative_ratio_trace(trace)
        self.assertEqual(ratios[0], (10, 2.0))
        self.assertTrue(math.isnan(ratios[1][1]))
        self.assertEqual(ratios[2], (30, 0.0))

    def test_plot_running_mean_matches_ratio_estimator(self):
        rows = [
            {"updates": "10", "sign": "1", "obs_0": "2", "signed_obs_0": "2", "elapsed_seconds": "1"},
            {"updates": "20", "sign": "-1", "obs_0": "6", "signed_obs_0": "-6", "elapsed_seconds": "2"},
            {"updates": "30", "sign": "1", "obs_0": "4", "signed_obs_0": "4", "elapsed_seconds": "3"},
        ]
        _, means, _ = plot.running_mean(rows, "obs_0")
        self.assertEqual(means[0], 2.0)
        self.assertTrue(math.isnan(means[1]))
        self.assertEqual(means[2], 0.0)

    def test_old_unsigned_trace_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "signed_obs_0"):
            plot.running_mean([{"updates": "1", "sign": "1", "obs_0": "2"}], "obs_0")

    def test_convergence_requires_no_later_escape(self):
        trace = [(index, value, 1.0) for index, value in
                 enumerate((1.0, 1.0, 10.0, -5.0, -2.0, 1.0, 1.0), start=1)]
        # The cumulative ratio first enters the band immediately, escapes at
        # sample three, and only becomes durably accurate at sample five.
        self.assertEqual(gap.convergence_update(trace, 1.0, 0.25, 1, 2), 5)

    def test_iat_for_constant_linearized_estimator(self):
        trace = [(index, 2.0, 1.0) for index in range(10)]
        self.assertEqual(gap.integrated_autocorrelation_time(trace), 1.0)


class LoopParsingTest(unittest.TestCase):
    def test_schedule_parser(self):
        name, betas = loop.parse_schedule("pilot ladder=0.1,0.5,2")
        self.assertEqual(name, "pilot_ladder")
        self.assertEqual(betas, (0.1, 0.5, 2.0))

    def test_absolute_anneal_schedule_mapping(self):
        mapping = anneal_driver.parse_schedule_mapping(["1.5:/tmp/a", "3:/tmp/b"])
        self.assertEqual(set(mapping), {1.5, 3.0})

    def test_target_directory_names_are_distinct(self):
        self.assertNotEqual(anneal_driver.target_directory_name(0, 1.0),
                            anneal_driver.target_directory_name(1, 1.0))

    def test_annealing_manifest_hash_includes_schedule_contents(self):
        with tempfile.TemporaryDirectory() as temporary:
            schedule = Path(temporary) / "anneal.txt"
            schedule.write_text("0 0.1\n10 1.0\n")
            first, schedule_hash = anneal_driver.annealing_plan_hash(
                1.0, 0.5, 10, 1, 0.001, schedule)
            schedule.write_text("0 0.2\n10 1.0\n")
            second, _ = anneal_driver.annealing_plan_hash(
                1.0, 0.5, 10, 1, 0.001, schedule)
            self.assertEqual(len(schedule_hash), 64)
            self.assertNotEqual(first, second)


class MinimalValidationTest(unittest.TestCase):
    def test_direct_diagonalization_matches_two_spin_closed_form(self):
        scale = math.sqrt(minimal.J ** 2 + 4.0 * minimal.GAMMA ** 2)
        for beta in minimal.BETAS:
            expected = (2.0 * minimal.GAMMA / scale * math.sinh(beta * scale) /
                        (math.cosh(beta * minimal.J) + math.cosh(beta * scale)))
            self.assertAlmostEqual(minimal.exact_expectation(beta), expected, places=13)


class QCPTValidationTest(unittest.TestCase):
    def test_target_weight_negative_controls_are_sensitive(self):
        controls = qcpt.negative_controls()
        self.assertGreater(controls["q0_diagonal_freeze_disagreement"], 1e-8)
        self.assertGreater(controls["q2_offdiagonal_reuse_disagreement"], 1e-8)
        self.assertTrue(controls["passed"])

    def test_split_exact_fixture_has_nontrivial_gamma_dependence(self):
        low = qcpt.exact_point(0.7, 0.25)
        high = qcpt.exact_point(0.7, 0.75)
        self.assertGreater(abs(low["E"] - high["E"]), 1e-3)
        self.assertGreater(abs(low["Xavg"] - high["Xavg"]), 1e-3)

    def test_split_observable_prefers_fixed_supported_relation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("prepare.cpp", "tests/qcpt_fixed.txt", "tests/qcpt_gamma.txt",
                         "tests/qcpt_z.txt", "tests/qcpt_xsum.txt"):
                source = ROOT / name
                shutil.copy2(source, directory / source.name)
            subprocess.run(["g++", "-O1", "-std=c++11", "-o", "prepare.bin", "prepare.cpp"],
                           cwd=directory, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["./prepare.bin", "--hamiltonian-fixed", "qcpt_fixed.txt",
                            "--hamiltonian-gamma", "qcpt_gamma.txt", "qcpt_z.txt", "qcpt_z.txt",
                            "qcpt_z.txt", "qcpt_z.txt", "qcpt_xsum.txt"],
                           cwd=directory, check=True, stdout=subprocess.DEVNULL)
            generated = (directory / "hamiltonian.hpp").read_text()
            self.assertIn('std::bitset<Nop>("101")', generated)

    def test_uniform_gamma_expansion_order_oracle(self):
        zero = edge.exact_values(edge.FIXED_TF, edge.GAMMA_TF, 0.9, 0.0)
        finite = edge.exact_values(edge.FIXED_TF, edge.GAMMA_TF, 0.9, 1.0)
        self.assertAlmostEqual(zero["q"], 0.0, places=13)
        self.assertGreater(finite["q"], 0.0)

    def test_edge_case_paths_have_distinct_coordinates(self):
        self.assertEqual(len(edge.PURE_BETA), 4)
        self.assertEqual(len(edge.PURE_GAMMA), 4)
        self.assertEqual(len({beta for beta, _ in edge.PURE_GAMMA}), 1)
        self.assertEqual(len({gamma for _, gamma in edge.PURE_BETA}), 1)


if __name__ == "__main__":
    unittest.main()
