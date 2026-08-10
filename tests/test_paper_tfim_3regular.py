import json
import math
import tempfile
import unittest
from pathlib import Path

from experiments.paper_tfim_3regular import (
    hamiltonian_terms,
    instance_metadata,
    random_3_regular_graph,
    validate_edges,
    write_instance,
)
from experiments.run_paper_tfim_3regular import (
    BETA_2019,
    expand_plan,
    gamma_2017,
    load_catalog,
    schedule_for_preset,
    write_schedule,
)


class PaperTfim3RegularTests(unittest.TestCase):
    def test_catalog_has_requested_sizes_and_counts(self):
        catalog_path = Path(__file__).parents[1] / "instances" / "paper_tfim_3regular_catalog.json"
        catalog = json.loads(catalog_path.read_text())
        counts = {}
        for entry in catalog["instances"]:
            counts[entry["n"]] = counts.get(entry["n"], 0) + 1
        self.assertEqual(counts, {4: 1, 6: 1, 8: 1, 10: 1, 12: 1, 36: 1, 60: 1,
                                  96: 50, 128: 50})

    def test_2019_plan_has_200_runs(self):
        catalog = load_catalog(Path(__file__).parents[1] / "instances" / "paper_tfim_3regular_catalog.json")
        plan = expand_plan(catalog, "2019")
        self.assertEqual(len(plan), 200)
        self.assertTrue(all(len(row["schedule"]) == 11 for row in plan))
        self.assertEqual(tuple(row["schedule"][i][0] for i in range(11) for row in plan[:1]), BETA_2019)

    def test_2017_gamma_curve(self):
        self.assertTrue(math.isclose(gamma_2017(1.0), 10.0 ** -0.5))
        schedule = schedule_for_preset("2017")
        self.assertTrue(all(math.isclose(gamma, gamma_2017(beta))
                            for beta, gamma, _ in schedule))

    def test_schedule_file_shape_matches_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.txt"
            write_schedule(path, "2019", 0.1)
            rows = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
            self.assertTrue(all(len(row.split()) == 2 for row in rows))
            write_schedule(path, "2017", None)
            rows = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
            self.assertTrue(all(len(row.split()) == 3 for row in rows))

    def test_graph_sizes_are_simple_and_3_regular(self):
        for n in (4, 6, 8, 10, 12, 36, 60, 96, 128):
            edges = random_3_regular_graph(n, 1234 + n)
            validate_edges(n, edges)

    def test_generation_is_reproducible(self):
        self.assertEqual(random_3_regular_graph(12, 17), random_3_regular_graph(12, 17))
        self.assertNotEqual(random_3_regular_graph(12, 17), random_3_regular_graph(12, 18))

    def test_split_terms_have_paper_signs(self):
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        fixed, gamma = hamiltonian_terms(4, edges)
        self.assertEqual(fixed[0], "1 1 Z 2 Z")
        self.assertEqual(gamma, ["-1 1 X", "-1 2 X", "-1 3 X", "-1 4 X"])

    def test_metadata_contains_reference_only_gamma_values(self):
        metadata = instance_metadata(4, 3, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        reference = metadata["reference_parameters"]
        self.assertEqual(reference["gamma_values"], [0.1, 0.4])
        self.assertIn("arbitrary Gamma", reference["gamma_values_note"])

    def test_written_instance_has_only_split_inputs_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_instance(Path(directory) / "instance", 4, 5)
            self.assertEqual({item.name for item in path.iterdir()},
                             {"H_fixed.txt", "H_gamma.txt", "instance.json"})
            metadata = json.loads((path / "instance.json").read_text())
            self.assertEqual(metadata["n"], 4)


if __name__ == "__main__":
    unittest.main()
