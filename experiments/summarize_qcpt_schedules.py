#!/usr/bin/env python3
"""Summarize matched QCPT schedule tuning and disjoint-seed production."""

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_stats
from summarize_model_controls import bootstrap_interval


def load_analyses(root: Path):
    analyses = []
    for path in sorted((root / "runs").glob("*/analysis.json")):
        analyses.append(json.loads(path.read_text()))
    return analyses


def aggregate_archive(root: Path, run_ids):
    trace_lines, manifest_lines, total = [], [], 0
    for run_id in sorted(run_ids):
        directory = root / "runs" / run_id
        manifest = directory / "manifest.json"
        manifest_lines.append(f"{hashlib.sha256(manifest.read_bytes()).hexdigest()}  {run_id}/manifest.json\n")
        for trace in sorted(directory.glob("trace_stream.rank*.csv")):
            total += trace.stat().st_size
            trace_lines.append(
                f"{hashlib.sha256(trace.read_bytes()).hexdigest()}  {run_id}/{trace.name}\n")
    return {
        "path": str(root / "runs"), "run_count": len(run_ids),
        "stream_trace_bytes": total,
        "ordered_trace_checksum_manifest_sha256": hashlib.sha256("".join(trace_lines).encode()).hexdigest(),
        "ordered_run_manifest_checksum_sha256": hashlib.sha256("".join(manifest_lines).encode()).hexdigest(),
    }


def summarize(root: Path, tuning_steps=100000, bootstrap_samples=10000,
              bootstrap_seed=20260809):
    analyses = load_analyses(root)
    config = json.loads((root / "campaign_manifest.json").read_text())["config"]
    tuning = [item for item in analyses
              if item.get("tuning") and int(item.get("steps", -1)) == tuning_steps]
    production = [item for item in analyses if not item.get("tuning")]
    records = study_stats.schedule_candidate_records(tuning, config["sweep_objective"])
    by_schedule = {}
    for record in records:
        by_schedule.setdefault(record["schedule_name"], {})[int(record["seed"])] = record
    winner = "pure_beta"
    winner_records = by_schedule[winner]
    rng = random.Random(bootstrap_seed)
    comparisons = {}
    for schedule, candidates in sorted(by_schedule.items()):
        if schedule == winner:
            continue
        seeds = sorted(set(winner_records) & set(candidates))
        objective = [winner_records[seed]["objective_ess_per_core_hour"] /
                     candidates[seed]["objective_ess_per_core_hour"] for seed in seeds]
        target = [winner_records[seed]["target_ess_per_core_hour"] /
                  candidates[seed]["target_ess_per_core_hour"] for seed in seeds]
        comparisons[schedule] = {
            "seeds": seeds,
            "winner_to_candidate_objective_ratio": {
                "median": statistics.median(objective),
                "bootstrap_95_interval": bootstrap_interval(objective, bootstrap_samples, rng),
                "values": objective,
            },
            "winner_to_candidate_target_ratio": {
                "median": statistics.median(target),
                "bootstrap_95_interval": bootstrap_interval(target, bootstrap_samples, rng),
                "values": target,
            },
        }
    schedule_summary = {}
    for schedule, candidates in sorted(by_schedule.items()):
        rows = list(candidates.values())
        schedule_summary[schedule] = {
            "run_ids": sorted(row["run_id"] for row in rows),
            "worst_edge_acceptance": min(row["worst_edge_acceptance"] for row in rows),
            "round_trips": sum(row["round_trips"] for row in rows),
            "all_convergence_gates": all(row["convergence_gate"] for row in rows),
            "all_sign_gates": all(row["sign_gate"] for row in rows),
            "any_qmax": any(row["qmax_achieved"] for row in rows),
            "median_target_ess_per_core_hour": statistics.median(
                row["target_ess_per_core_hour"] for row in rows),
            "median_objective_ess_per_core_hour": statistics.median(
                row["objective_ess_per_core_hour"] for row in rows),
            "objective_coverage_fraction": statistics.median(
                row["objective_coverage_fraction"] for row in rows),
        }
    exact_tuning = all(point.get("exact_pass", False) for item in tuning for point in item["points"])
    exact_production = all(point.get("exact_pass", False) for item in production for point in item["points"])
    production_convergence = all(point["convergence_pass"] for item in production for point in item["points"])
    relevant_ids = [item["run_id"] for item in tuning + production]
    first_manifest = json.loads((root / "runs" / relevant_ids[0] / "manifest.json").read_text())
    return {
        "schema_version": 1, "analysis_commit": "commit containing this artifact",
        "campaign": root.name, "environment": first_manifest["environment"],
        "tuning_steps": tuning_steps, "tuning_seeds": sorted(winner_records),
        "production_seeds": sorted(int(item["seed"]) for item in production),
        "production_run_ids": sorted(item["run_id"] for item in production),
        "selected_schedule": winner, "selection_metric": config["schedule_selection_metric"],
        "schedules": schedule_summary, "matched_comparisons": comparisons,
        "tuning_exact_gate": exact_tuning, "production_exact_gate": exact_production,
        "production_convergence_gate": production_convergence,
        "production_precision_gate": all(item["analysis_pass"] for item in production),
        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": bootstrap_seed,
        "archive": aggregate_archive(root, relevant_ids),
        "limitations": [
            "This L=2 smoke validates schedule selection and sweep reuse, not critical scaling.",
            "Each run contains one four-slot ladder; cross-seed R-hat is not yet computed jointly.",
            "The production precision gate fails even though exact and convergence gates pass.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.output_dir)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {args.result}")


if __name__ == "__main__":
    main()
