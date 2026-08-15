"""Tests for QCPT ensemble summarization helpers."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import summarize_qcpt_schedules as qcpt


class QCPTSummaryTest(unittest.TestCase):
    def test_archive_checksum_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "abc"
            run.mkdir(parents=True)
            (run / "manifest.json").write_text("manifest")
            (run / "trace_stream.rank0.csv").write_text("trace")
            first = qcpt.aggregate_archive(root, ["abc"])
            second = qcpt.aggregate_archive(root, ["abc"])
        self.assertEqual(first, second)
        self.assertEqual(first["stream_trace_bytes"], 5)


if __name__ == "__main__":
    unittest.main()
