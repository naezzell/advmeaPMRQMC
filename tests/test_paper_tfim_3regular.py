import json
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


class PaperTfim3RegularTests(unittest.TestCase):
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
