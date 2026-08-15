"""Compare matched fast/reference estimator traces and timing components."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(fast_directory: Path, reference_directory: Path, tolerance: float = 1e-10):
    fast_paths = sorted(fast_directory.glob("trace_stream.rank*.csv"))
    reference_paths = sorted(reference_directory.glob("trace_stream.rank*.csv"))
    if len(fast_paths) != len(reference_paths) or not fast_paths:
        raise ValueError("fast/reference stream layouts differ")
    compared_rows = 0
    max_difference = 0.0
    mismatches = 0
    for fast_path, reference_path in zip(fast_paths, reference_paths):
        fast_rows, reference_rows = read_rows(fast_path), read_rows(reference_path)
        if len(fast_rows) != len(reference_rows):
            raise ValueError(f"trace lengths differ: {fast_path} and {reference_path}")
        columns = [column for column in fast_rows[0]
                   if column in ("updates", "sign") or
                   column.startswith("obs_") or column.startswith("signed_obs_")]
        for fast_row, reference_row in zip(fast_rows, reference_rows):
            compared_rows += 1
            for column in columns:
                difference = abs(float(fast_row[column]) - float(reference_row[column]))
                max_difference = max(max_difference, difference)
                mismatches += difference > tolerance
    fast_summary = json.loads((fast_directory / "summary.json").read_text())
    reference_summary = json.loads((reference_directory / "summary.json").read_text())
    fast_measurement = sum(float(read_rows(path)[-1]["measurement_seconds"]) for path in fast_paths)
    reference_measurement = sum(float(read_rows(path)[-1]["measurement_seconds"])
                                for path in reference_paths)
    fast_wall = float(fast_summary["timings"]["simulation"])
    reference_wall = float(reference_summary["timings"]["simulation"])
    result = {
        "fast_run_id": fast_summary["run_id"], "reference_run_id": reference_summary["run_id"],
        "compared_rows": compared_rows, "compared_columns_per_row": len(columns),
        "tolerance": tolerance, "max_abs_difference": max_difference,
        "mismatches": mismatches, "correctness_pass": mismatches == 0,
        "fast_measurement_core_seconds": fast_measurement,
        "reference_measurement_core_seconds": reference_measurement,
        "measurement_speedup": reference_measurement / fast_measurement,
        "fast_wall_seconds": fast_wall, "reference_wall_seconds": reference_wall,
        "wall_speedup": reference_wall / fast_wall,
    }
    manifest_path = fast_directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        planned = manifest["planned"]
        result["protocol"] = {"L": int(planned["L"]), "lambda": float(planned["lambda"]),
                              "beta": float(planned["beta"]), "seed": int(planned["seed"]),
                              "steps": int(planned["steps"]),
                              "steps_per_measurement": int(planned["steps_per_measurement"]),
                              "source_commit": manifest["source_commit"],
                              "hardware": manifest["environment"].get("cpu_model", "unknown")}
        result["fast_manifest_sha256"] = file_sha256(manifest_path)
        reference_manifest = reference_directory / "manifest.json"
        result["reference_manifest_sha256"] = file_sha256(reference_manifest)
        log = (fast_directory / "simulation.log").read_text()
        mean_q = re.search(r"Total mean\(q\) = ([^\n]+)", log)
        max_q = re.search(r"Total max\(q\) = ([^\n]+)", log)
        result["mean_q"] = float(mean_q.group(1)) if mean_q else float("nan")
        result["max_q"] = float(max_q.group(1)) if max_q else float("nan")
        result["archive_paths"] = [str(fast_directory), str(reference_directory)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fast")
    parser.add_argument("reference")
    parser.add_argument("--output")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    result = compare(Path(args.fast), Path(args.reference), args.tolerance)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")
    raise SystemExit(0 if result["correctness_pass"] else 1)


if __name__ == "__main__":
    main()
