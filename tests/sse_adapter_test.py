"""Unit tests for matched PMR/ALPS TFIM conventions."""

import sys
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import sse_adapter


class SSEAdapterTest(unittest.TestCase):
    def test_pauli_normalization(self):
        mapped = sse_adapter.pauli_to_alps(1.0, 3.044)
        self.assertEqual(mapped["Jxy"], 0.0)
        self.assertEqual(mapped["Jz"], -4.0)
        self.assertEqual(mapped["Gamma"], 6.088)

    def test_square_boundary_conditions_and_temperature(self):
        pbc = sse_adapter.tfim_parameters(8, 3.044, 16.0, True, 10, 100, 7)
        obc = sse_adapter.tfim_parameters(8, 3.044, 16.0, False, 10, 100, 7)
        self.assertEqual(pbc["LATTICE"], "square lattice")
        self.assertEqual(obc["LATTICE"], "open square lattice")
        self.assertEqual(pbc["T"], 1 / 16)
        self.assertEqual(pbc["L"], pbc["W"])

    def test_container_and_adapter_pin_same_source(self):
        lock = json.loads((ROOT / "containers" / "alps-sse" / "lock.json").read_text())
        dockerfile = (ROOT / "containers" / "alps-sse" / "Dockerfile").read_text()
        self.assertEqual(lock["source_commit"], sse_adapter.ALPS_COMMIT)
        self.assertIn(sse_adapter.ALPS_COMMIT, dockerfile)
        self.assertIn("ubuntu:24.04@sha256:", dockerfile)


if __name__ == "__main__":
    unittest.main()
