"""Deterministic statistical tests for the study analysis layer."""

import math
import random
import sys
import tempfile
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

    def test_rank_rhat_handles_identical_discrete_chains(self):
        chains = [[-1.0, -1.0, 1.0, 1.0] * 100 for _ in range(4)]
        self.assertLess(stats.split_rank_normalized_rhat(chains), 1.01)
        self.assertGreater(stats.split_rank_normalized_rhat(chains), 0.99)

    def test_discrete_ratio_centering_is_not_used_for_rhat(self):
        chains = []
        for seed in range(4):
            rng = random.Random(seed)
            chains.append([rng.choice([-12.0, -10.0, -8.0, -6.0]) for _ in range(200)])
        centered = [[value - sum(chain) / len(chain) for value in chain]
                    for chain in chains]
        self.assertLess(stats.split_rank_normalized_rhat(chains), 1.01)
        self.assertGreater(stats.split_rank_normalized_rhat(centered), 1.03)

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

    def test_jackknife_blocks_do_not_cross_streams(self):
        rows = []
        for stream in (0, 1):
            for _ in range(15):
                rows.append({"stream": stream, "sign": 1.0,
                             "signed_obs_2": -1.0, "signed_obs_3": 1.0})
        result = stats.joint_block_jackknife(
            rows, beta=1.0, spins=1, block_length=10,
            observable_columns=["obs_2", "obs_3"])
        self.assertEqual(result["energy"]["blocks"], 2)
        self.assertTrue(math.isnan(result["fidelity_susceptibility"]["mean"]))

    def test_beta_convergence_requires_two_stable_increments(self):
        rows = []
        for beta, value in ((2, 1.2), (4, 1.0015), (8, 1.001), (16, 1.0005)):
            rows.append({"method": "fixed", "protocol": "cheap", "L": 4, "lambda": 3.0,
                         "seed": 1, "beta": beta, "beta_over_L": beta / 4,
                         "energy_density": value, "energy_density_se": 0.01})
        result = stats.beta_convergence(rows, "energy_density")[0]
        self.assertTrue(result["converged"])
        self.assertIsNone(result["next_beta_over_L"])

    def test_qcpt_preserves_every_slot_as_production_point(self):
        planned = {"method": "qcpt", "beta": "8", "lambda": "3.0"}
        rows = [
            {"slot": slot, "beta": beta, "gamma": gamma, "sign": 1.0,
             "signed_obs_2": -1.0}
            for slot, beta, gamma in ((0, 1.0, 0.0), (1, 2.0, 1.5), (2, 4.0, 3.0))
        ]
        points = stats.split_parameter_points(rows, planned)
        self.assertEqual([point[0]["slot"] for point in points], [0, 1, 2])
        self.assertEqual([point[0]["lambda"] for point in points], [0.0, 1.5, 3.0])
        self.assertEqual([point[0]["beta"] for point in points], [1.0, 2.0, 4.0])

    def test_schedule_selection_enforces_transport_gates(self):
        records = []
        for schedule, acceptance, score in (("good", 0.2, 10.0), ("stuck", 0.1, 100.0)):
            for seed in (1, 2, 3, 4):
                records.append({
                    "run_id": f"{schedule}-{seed}", "L": 4, "target_lambda": 3.044,
                    "target_beta": 8.0, "protocol": "cheap", "representation": "standard",
                    "periodic": True, "move": "none", "schedule_name": schedule, "seed": seed,
                    "Tsteps": 10, "steps": 100, "steps_per_measurement": 1,
                    "target_ess_per_core_hour": score, "sweep_ess_per_core_hour": score * 2,
                    "worst_edge_acceptance": acceptance, "round_trips": 1,
                    "qmax_achieved": False, "sign_gate": True, "convergence_gate": True,
                })
        ranked = stats.rank_schedule_records(records)
        by_name = {row["schedule_name"]: row for row in ranked}
        self.assertTrue(by_name["good"]["eligible"])
        self.assertTrue(by_name["good"]["selected"])
        self.assertFalse(by_name["stuck"]["eligible"])
        self.assertFalse(by_name["stuck"]["selected"])

    def test_schedule_selection_can_optimize_declared_sweep_objective(self):
        records = []
        for schedule, endpoint, objective, coverage in (
                ("endpoint", 100.0, 10.0, 0.1), ("figure_grid", 20.0, 80.0, 0.6)):
            for seed in (1, 2, 3, 4):
                records.append({
                    "run_id": f"{schedule}-{seed}", "L": 4, "target_lambda": 3.044,
                    "target_beta": 8.0, "protocol": "cheap", "representation": "standard",
                    "periodic": True, "move": "none", "schedule_name": schedule, "seed": seed,
                    "Tsteps": 10, "steps": 100, "steps_per_measurement": 1,
                    "target_ess_per_core_hour": endpoint, "sweep_ess_per_core_hour": endpoint + objective,
                    "objective_ess_per_core_hour": objective,
                    "objective_coverage_fraction": coverage,
                    "worst_edge_acceptance": 0.3, "round_trips": 2,
                    "qmax_achieved": False, "sign_gate": True, "convergence_gate": True,
                })
        ranked = stats.rank_schedule_records(records, selection_metric="objective")
        selected = next(row for row in ranked if row["selected"])
        self.assertEqual(selected["schedule_name"], "figure_grid")
        self.assertEqual(selected["selection_metric"], "objective")

    def test_schedule_objective_counts_only_requested_coordinates(self):
        analysis = {
            "run_id": "qcpt", "method": "qcpt", "tuning": True, "L": 4,
            "target_lambda": 3.044, "target_beta": 4.0, "protocol": "cheap",
            "representation": "standard", "periodic": True, "move": "none",
            "schedule_name": "pure_beta", "seed": 1,
            "Tsteps": 10, "steps": 100, "steps_per_measurement": 1,
            "points": [
                {"point": 0, "lambda": 3.044, "beta_over_L": 0.5,
                 "effective_samples": 40, "core_hours": 1.0, "ess_per_core_hour": 40,
                 "mean_sign": 1, "convergence_pass": True},
                {"point": 1, "lambda": 3.044, "beta_over_L": 1.0,
                 "effective_samples": 60, "core_hours": 1.0, "ess_per_core_hour": 60,
                 "mean_sign": 1, "convergence_pass": True},
            ],
            "tempering": {"worst_edge_acceptance": 0.3, "round_trips": 1},
        }
        objective = {"lambda": [2.8, 3.044], "beta_over_L": [0.5, 1.0]}
        record = stats.schedule_candidate_records([analysis], objective)[0]
        self.assertEqual(record["objective_points_covered"], 2)
        self.assertEqual(record["objective_points_total"], 4)
        self.assertAlmostEqual(record["objective_coverage_fraction"], 0.5)
        self.assertAlmostEqual(record["objective_ess_per_core_hour"], 100.0)

    def test_selected_schedule_uses_disjoint_production_seeds(self):
        simulation = {"Tsteps": 10, "steps": 100, "steps_per_measurement": 1,
                      "qmax": 64, "nbins": 10, "max_wall_seconds": 100}
        template = stats.study.make_plan_row(
            "abc", "tuning", "qcpt", "cheap", 4, 3.044, 8.0, True,
            "standard", 0, 1100, True, simulation, schedule_name="pure_beta")
        winner = {"selected": True, "schedule_name": "pure_beta", "L": 4,
                  "target_lambda": 3.044, "target_beta": 8.0, "protocol": "cheap",
                  "representation": "standard", "move": "none",
                  "Tsteps": 10, "steps": 100, "steps_per_measurement": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats.study.write_csv(root / "plan.csv", [template])
            stats.study.write_json(root / "campaign_manifest.json", {
                "config": {"production_seeds": [5100, 6100, 7100, 8100]}
            })
            production = stats.selected_production_rows(root, [winner])
        self.assertEqual(sorted(int(row["seed"]) for row in production), [5100, 6100, 7100, 8100])
        self.assertTrue(all(row["tuning"] == "False" for row in production))
        self.assertTrue(all(row["source_commit"] == stats.study.source_commit() for row in production))
        self.assertTrue({row["run_id"] for row in production}.isdisjoint({template["run_id"]}))

    def test_schedule_ranking_separates_adaptive_resource_levels(self):
        records = []
        for steps in (100, 200):
            for seed in (1, 2, 3, 4):
                records.append({
                    "run_id": f"{steps}-{seed}", "L": 2, "target_lambda": 1.0,
                    "target_beta": 2.0, "protocol": "cheap", "representation": "standard",
                    "periodic": True, "move": "none", "schedule_name": "pure_beta",
                    "seed": seed, "Tsteps": 10, "steps": steps, "steps_per_measurement": 1,
                    "target_ess_per_core_hour": steps, "sweep_ess_per_core_hour": steps,
                    "worst_edge_acceptance": 0.3, "round_trips": 1,
                    "qmax_achieved": False, "sign_gate": True, "convergence_gate": True,
                })
        ranked = stats.rank_schedule_records(records)
        self.assertEqual(len(ranked), 2)
        self.assertEqual({row["steps"] for row in ranked}, {100, 200})

    def test_adaptive_extension_doubles_failed_resources(self):
        simulation = {"Tsteps": 10, "steps": 100, "steps_per_measurement": 1,
                      "qmax": 64, "nbins": 10, "max_wall_seconds": 100}
        template = stats.study.make_plan_row(
            "abc", "adaptive", "current_fixed", "cheap", 4, 3.044, 8.0, True,
            "standard", 0, 1100, False, simulation)
        analysis = {"run_id": template["run_id"], "analysis_pass": False, "points": [{
            "thermalization_pass": False, "correlation_pass": False,
            "effective_samples": 100, "precision": {"energy_density": False},
            "wall_seconds": 1.0,
        }]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats.study.write_csv(root / "plan.csv", [template])
            extensions, decisions = stats.adaptive_extension_rows(root, [analysis])
        self.assertEqual(len(extensions), 1)
        self.assertEqual(int(extensions[0]["Tsteps"]), 20)
        self.assertEqual(int(extensions[0]["steps"]), 200)
        self.assertEqual(decisions[0]["status"], "planned")

    def test_campaign_plan_registry_recovers_adaptive_manifest(self):
        simulation = {"Tsteps": 20, "steps": 200, "steps_per_measurement": 1,
                      "qmax": 64, "nbins": 10, "max_wall_seconds": 100}
        adaptive = stats.study.make_plan_row(
            "abc", "adaptive", "qcpt", "cheap", 2, 1.0, 2.0, True,
            "standard", 0, 1100, True, simulation, schedule_name="pure_beta")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats.study.write_csv(root / "plan.csv", [])
            run = root / "runs" / adaptive["run_id"]
            run.mkdir(parents=True)
            stats.study.write_json(run / "manifest.json", {"planned": adaptive})
            recovered = stats.campaign_plan_rows(root)
        self.assertEqual([row["run_id"] for row in recovered], [adaptive["run_id"]])

    def test_analysis_fingerprint_tracks_trace_and_analyzer_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text("manifest")
            (root / "summary.json").write_text("summary")
            trace = root / "trace.csv"
            trace.write_text("first")
            first = stats.analysis_fingerprint(root, [trace])
            trace.write_text("second")
            second = stats.analysis_fingerprint(root, [trace])
        self.assertNotEqual(first, second)

    def test_analyze_run_reuses_matching_cached_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text('{"planned":{"run_id":"cached"}}')
            (root / "summary.json").write_text('{}')
            trace = root / "trace.csv"
            trace.write_text("not,a,parseable,trace\n")
            fingerprint = stats.analysis_fingerprint(root, [trace])
            cached = {"run_id": "cached", "analysis_fingerprint": fingerprint,
                      "status": "cached-sentinel"}
            (root / "analysis.json").write_text(__import__("json").dumps(cached))
            self.assertEqual(stats.analyze_run(root), cached)


if __name__ == "__main__":
    unittest.main()
