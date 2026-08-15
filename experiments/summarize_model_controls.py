#!/usr/bin/env python3
"""Summarize matched TFIM representation and global-inversion controls."""

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


def quantile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    fraction = position - left
    return ordered[left] * (1 - fraction) + ordered[right] * fraction


def bootstrap_interval(values, samples, rng):
    bootstrapped = []
    for _ in range(samples):
        selected = [values[rng.randrange(len(values))] for _ in values]
        bootstrapped.append(statistics.median(selected))
    return [quantile(bootstrapped, 0.025), quantile(bootstrapped, 0.975)]


def point_map(analyses):
    return {(item["representation"], item["move"], int(item["seed"])): item["points"][0]
            for item in analyses if item["method"] == "current_fixed" and item.get("points")}


def energy_agrees(left, right):
    a, b = left["derived"]["energy"], right["derived"]["energy"]
    tolerance = 5.0 * math.hypot(a["standard_error"], b["standard_error"])
    return abs(a["mean"] - b["mean"]) <= tolerance


def archive_record(root: Path, run_id: str):
    directory = root / "runs" / run_id
    manifest_path = directory / "manifest.json"
    traces = sorted(directory.glob("trace_stream.rank*.csv"))
    checksum_lines = "".join(
        f"{hashlib.sha256(trace.read_bytes()).hexdigest()}  {trace.name}\n" for trace in traces)
    return {
        "path": str(directory),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "stream_trace_bytes": sum(trace.stat().st_size for trace in traces),
        "ordered_trace_checksum_manifest_sha256": hashlib.sha256(checksum_lines.encode()).hexdigest(),
    }


def summarize(path: Path, bootstrap_samples=10000, bootstrap_seed=20260809):
    analyses = json.loads(path.read_text())
    points = point_map(analyses)
    seeds = sorted(set(seed for representation, move, seed in points
                       if representation == "standard" and move == "none") &
                   set(seed for representation, move, seed in points
                       if representation == "standard" and move == "global_z2") &
                   set(seed for representation, move, seed in points
                       if representation == "rotated_parity" and move == "none"))
    if not seeds:
        raise ValueError("no complete matched control seeds")
    move_metrics = {"z_magnetization_iat_speedup": [],
                    "z_magnetization_ess_per_core_hour_speedup": [],
                    "energy_ess_per_core_hour_ratio": []}
    representation_metrics = {"energy_iat_speedup": [],
                              "energy_ess_per_core_hour_speedup": [],
                              "wall_speedup": []}
    paired = []
    move_correctness, move_convergence = [], []
    representation_correctness, representation_convergence = [], []
    for seed in seeds:
        standard = points[("standard", "none", seed)]
        moved = points[("standard", "global_z2", seed)]
        rotated = points[("rotated_parity", "none", seed)]
        z_standard_rate = standard["z_magnetization_effective_samples"] / standard["core_hours"]
        z_moved_rate = moved["z_magnetization_effective_samples"] / moved["core_hours"]
        move_values = {
            "z_magnetization_iat_speedup": standard["z_magnetization_median_iat"] /
            moved["z_magnetization_median_iat"],
            "z_magnetization_ess_per_core_hour_speedup": z_moved_rate / z_standard_rate,
            "energy_ess_per_core_hour_ratio": moved["ess_per_core_hour"] /
            standard["ess_per_core_hour"],
        }
        representation_values = {
            "energy_iat_speedup": rotated["energy_median_iat"] / standard["energy_median_iat"],
            "energy_ess_per_core_hour_speedup": standard["ess_per_core_hour"] /
            rotated["ess_per_core_hour"],
            "wall_speedup": rotated["wall_seconds"] / standard["wall_seconds"],
        }
        for name, value in move_values.items():
            move_metrics[name].append(value)
        for name, value in representation_values.items():
            representation_metrics[name].append(value)
        move_ok = energy_agrees(standard, moved)
        representation_ok = energy_agrees(standard, rotated)
        move_gate = all(point["thermalization_pass"] and point["correlation_pass"]
                        for point in (standard, moved))
        representation_gate = all(point["thermalization_pass"] and point["correlation_pass"]
                                  for point in (standard, rotated))
        move_correctness.append(move_ok)
        representation_correctness.append(representation_ok)
        move_convergence.append(move_gate)
        representation_convergence.append(representation_gate)
        paired.append({"seed": seed, "standard_run_id": standard["run_id"],
                       "global_z2_run_id": moved["run_id"],
                       "rotated_run_id": rotated["run_id"],
                       "move": move_values, "representation": representation_values,
                       "move_energy_agreement": move_ok,
                       "move_convergence_gate": move_gate,
                       "representation_energy_agreement": representation_ok,
                       "representation_convergence_gate": representation_gate})
    rng = random.Random(bootstrap_seed)

    def metric_summary(metrics):
        return {name: {"median": statistics.median(values),
                       "bootstrap_95_interval": bootstrap_interval(values, bootstrap_samples, rng),
                       "values": values}
                for name, values in metrics.items()}

    move_summary = metric_summary(move_metrics)
    representation_summary = metric_summary(representation_metrics)
    move_claim = (all(move_correctness) and all(move_convergence) and
                  move_summary["z_magnetization_ess_per_core_hour_speedup"]
                  ["bootstrap_95_interval"][0] > 1.0)
    representation_claim = (all(representation_correctness) and all(representation_convergence) and
                            representation_summary["energy_ess_per_core_hour_speedup"]
                            ["bootstrap_95_interval"][0] > 1.0)
    run_ids = sorted({run_id for item in paired for run_id in
                      (item["standard_run_id"], item["global_z2_run_id"], item["rotated_run_id"])})
    manifest = json.loads((path.parent / "runs" / run_ids[0] / "manifest.json").read_text())
    return {
        "schema_version": 1, "analysis_commit": "commit containing this artifact",
        "campaign_summary": str(path), "seeds": seeds,
        "simulation_base_commit": manifest["source_commit"],
        "simulation_dirty_worktree": manifest["dirty_worktree"],
        "environment": manifest["environment"],
        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": bootstrap_seed,
        "paired_runs": paired,
        "archive": {run_id: archive_record(path.parent, run_id) for run_id in run_ids},
        "global_z2": {"metrics": move_summary, "correctness_gate": all(move_correctness),
                      "convergence_gate": all(move_convergence), "speedup_claim_pass": move_claim},
        "representation": {"metrics": representation_summary,
                           "correctness_gate": all(representation_correctness),
                           "convergence_gate": all(representation_convergence),
                           "speedup_claim_pass": representation_claim},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    args = parser.parse_args()
    result = summarize(args.summary, args.bootstrap_samples, args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
