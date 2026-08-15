"""Run a reproducible grid of fixed-beta versus beta-PT benchmarks.

The loop delegates each grid point to ``pt_convergence_gap.py`` so every run
keeps its raw traces, exact reference, swap diagnostics, and configuration.
Use small systems with exact diagonalization as a correctness gate before
using the same loop on larger mixing benchmarks.

Example:
    python3 experiments/pt_benchmark_loop.py /tmp/pt_loop \
      --model max2sat --n 10 --seeds 11,12,13 \
      --rng-seeds 1000,2000 \
      --schedule coarse=0.1,0.3,0.7,1.5,3,5 \
      --schedule geometric=0.1,0.219,0.478,1.046,2.287,5 \
      --exchange-cadences 5,10,50 --independent-ladders 1,2 \
      --exact-reference --oversubscribe
"""

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path


def comma_ints(text):
    values = [int(value) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return values


def parse_schedule(specification):
    if "=" not in specification:
        raise argparse.ArgumentTypeError("schedule must have NAME=beta,beta,... format")
    name, text = specification.split("=", 1)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    try:
        betas = tuple(float(value) for value in text.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not name or len(betas) < 2 or any(a >= b for a, b in zip(betas, betas[1:])):
        raise argparse.ArgumentTypeError("schedule needs a name and strictly increasing betas")
    return name, betas


def read_csv(path):
    with Path(path).open(newline="") as input_file:
        return list(csv.DictReader(input_file))


def finite_median(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.median(values) if values else None


def convergence_wall(summary):
    update = summary.get("convergence_updates")
    last = summary.get("last_update")
    wall = summary.get("wall_seconds")
    if update is None or not last or wall is None:
        return None
    return float(wall) * float(update) / float(last)


def collect_run(run_directory, schedule_name, cadence, ladders):
    run_directory = Path(run_directory)
    report = json.loads((run_directory / "comparison_report.json").read_text())
    fixed = report["fixed_beta"]
    pt = report["beta_tempered_cold_slot"]
    reference = float(report["reference"]["value"])
    fixed_tolerance = float(fixed.get("accuracy_tolerance", report["reference"]["tolerance"]))
    pt_tolerance = float(pt.get("accuracy_tolerance", report["reference"]["tolerance"]))
    swaps = read_csv(run_directory / "beta_tempered" / "pt_swaps.csv")
    acceptance = [float(row["acceptance"]) for row in swaps]
    flow = read_csv(run_directory / "beta_tempered" / "pt_flow.csv")
    round_trips = int(float(flow[0]["round_trips"])) if flow else 0
    fixed_wall_to_accuracy = convergence_wall(fixed)
    pt_wall_to_accuracy = convergence_wall(pt)
    fixed_error = abs(float(fixed["final_mean"]) - reference)
    pt_error = abs(float(pt["final_mean"]) - reference)
    return {
        "schedule": schedule_name,
        "betas": report["betas"],
        "seed": report["seed"],
        "rng_seed_offset": report.get("rng_seed_offset", 1000),
        "cadence": cadence,
        "independent_ladders": ladders,
        "mpi_ranks": report["mpi_ranks_each_run"],
        "reference_source": report["reference"]["source"],
        "reference": reference,
        "fixed_tolerance": fixed_tolerance,
        "pt_tolerance": pt_tolerance,
        "fixed_final": fixed["final_mean"],
        "pt_final": pt["final_mean"],
        "fixed_error": fixed_error,
        "pt_error": pt_error,
        "fixed_pass": fixed_error <= fixed_tolerance,
        "pt_pass": pt_error <= pt_tolerance,
        "fixed_block_standard_error": fixed.get("block_standard_error"),
        "pt_block_standard_error": pt.get("block_standard_error"),
        "fixed_mean_sign": fixed.get("mean_sign"),
        "pt_mean_sign": pt.get("mean_sign"),
        "fixed_iat": fixed.get("integrated_autocorrelation_measurements"),
        "pt_iat": pt.get("integrated_autocorrelation_measurements"),
        "fixed_effective_samples_per_second": fixed.get("effective_samples_per_second"),
        "pt_effective_samples_per_second": pt.get("effective_samples_per_second"),
        "ess_rate_speedup": (pt["effective_samples_per_second"] / fixed["effective_samples_per_second"]
                             if fixed.get("effective_samples_per_second") not in (None, 0)
                             and pt.get("effective_samples_per_second") is not None else None),
        "fixed_convergence_updates": fixed.get("convergence_updates"),
        "pt_convergence_updates": pt.get("convergence_updates"),
        "fixed_wall_to_accuracy": fixed_wall_to_accuracy,
        "pt_wall_to_accuracy": pt_wall_to_accuracy,
        "update_speedup": (fixed["convergence_updates"] / pt["convergence_updates"]
                           if fixed.get("convergence_updates") is not None
                           and pt.get("convergence_updates") not in (None, 0) else None),
        "wall_speedup": (fixed_wall_to_accuracy / pt_wall_to_accuracy
                         if fixed_wall_to_accuracy is not None
                         and pt_wall_to_accuracy not in (None, 0) else None),
        "acceptance_min": min(acceptance) if acceptance else None,
        "acceptance_median": statistics.median(acceptance) if acceptance else None,
        "acceptance_max": max(acceptance) if acceptance else None,
        "round_trips": round_trips,
        "max_q": float(flow[0]["max_q"]) if flow else None,
        "qmax_achieved": bool(int(float(flow[0].get("qmax_achieved", 0)))) if flow else None,
        "run_directory": str(run_directory),
    }


def aggregate(rows):
    groups = {}
    for row in rows:
        key = (row["schedule"], row["cadence"], row["independent_ladders"])
        groups.setdefault(key, []).append(row)
    output = []
    for (schedule, cadence, ladders), samples in sorted(groups.items()):
        output.append({
            "schedule": schedule,
            "cadence": cadence,
            "independent_ladders": ladders,
            "seeds": len(samples),
            "correctness_pass_rate": sum(row["fixed_pass"] and row["pt_pass"] for row in samples) / len(samples),
            "median_fixed_error": finite_median(row["fixed_error"] for row in samples),
            "median_pt_error": finite_median(row["pt_error"] for row in samples),
            "median_update_speedup": finite_median(row["update_speedup"] for row in samples),
            "median_wall_speedup": finite_median(row["wall_speedup"] for row in samples),
            "median_ess_rate_speedup": finite_median(row["ess_rate_speedup"] for row in samples),
            "worst_edge_acceptance": min(row["acceptance_min"] for row in samples),
            "median_edge_acceptance": finite_median(row["acceptance_median"] for row in samples),
            "median_round_trips": finite_median(row["round_trips"] for row in samples),
            "median_fixed_sign": finite_median(row["fixed_mean_sign"] for row in samples),
            "median_pt_sign": finite_median(row["pt_mean_sign"] for row in samples),
            "any_qmax_achieved": any(row["qmax_achieved"] for row in samples),
        })
    return output


def write_flat_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, tuple)) else value
                             for key, value in row.items()})


def run_loop(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    driver = Path(__file__).resolve().with_name("pt_convergence_gap.py")
    rows = []
    for schedule_name, betas in args.schedule:
        beta_text = ",".join(f"{value:.17g}" for value in betas)
        for cadence in args.exchange_cadences:
            for ladders in args.independent_ladders:
                for seed in args.seeds:
                  for rng_seed in args.rng_seeds:
                    run_name = f"{schedule_name}_x{cadence}_l{ladders}_s{seed}_r{rng_seed}"
                    run_directory = output / "runs" / run_name
                    command = [
                        args.python, str(driver), str(run_directory),
                        "--model", args.model, "--n", str(args.n), "--gamma", str(args.gamma),
                        "--seed", str(seed), "--betas", beta_text, "--ladders", str(ladders),
                        "--rng-seed-offset", str(rng_seed),
                        "--Tsteps", str(args.Tsteps), "--steps", str(args.steps),
                        "--steps-per-measurement", str(args.steps_per_measurement),
                        "--updates-per-exchange", str(cadence), "--qmax", str(args.qmax),
                        "--nbins", str(args.nbins), "--block-measurements", str(args.block_measurements),
                        "--stability-blocks", str(args.stability_blocks),
                        "--tolerance-sigma", str(args.tolerance_sigma),
                        "--absolute-tolerance", str(args.absolute_tolerance),
                    ]
                    if args.exact_reference:
                        command.append("--exact-reference")
                    if args.oversubscribe:
                        command.append("--oversubscribe")
                    if args.dry_run:
                        print(" ".join(command))
                        continue
                    report_path = run_directory / "comparison_report.json"
                    if not (args.resume_existing and report_path.exists()):
                        subprocess.run(command, check=True)
                    rows.append(collect_run(run_directory, schedule_name, cadence, ladders))
                    write_flat_csv(output / "runs.csv", rows)
    if args.dry_run:
        return []
    summary = aggregate(rows)
    write_flat_csv(output / "summary.csv", summary)
    (output / "summary.json").write_text(json.dumps({"runs": rows, "summary": summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.require_correctness and any(row["correctness_pass_rate"] < 1.0 for row in summary):
        raise SystemExit("at least one configuration failed the exact-reference correctness gate")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--model", choices=("tfim", "max2sat"), default="max2sat")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--seeds", type=comma_ints, default=comma_ints("2019,2020,2021"))
    parser.add_argument("--rng-seeds", type=comma_ints, default=comma_ints("1000"),
                        help="independent deterministic QMC seed offsets")
    parser.add_argument("--schedule", action="append", type=parse_schedule,
                        default=None, help="repeat NAME=beta,beta,... (default: coarse six-level ladder)")
    parser.add_argument("--exchange-cadences", type=comma_ints, default=comma_ints("5,10,50"))
    parser.add_argument("--independent-ladders", type=comma_ints, default=comma_ints("1"))
    parser.add_argument("--Tsteps", type=int, default=100000)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--steps-per-measurement", type=int, default=100)
    parser.add_argument("--qmax", type=int, default=1000)
    parser.add_argument("--nbins", type=int, default=100)
    parser.add_argument("--block-measurements", type=int, default=20)
    parser.add_argument("--stability-blocks", type=int, default=10)
    parser.add_argument("--tolerance-sigma", type=float, default=2.0)
    parser.add_argument("--absolute-tolerance", type=float, default=0.01)
    parser.add_argument("--exact-reference", action="store_true")
    parser.add_argument("--require-correctness", action="store_true")
    parser.add_argument("--oversubscribe", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter used for child benchmarks (must have numpy for exact references)")
    args = parser.parse_args()
    if args.schedule is None:
        args.schedule = [parse_schedule("coarse=0.1,0.3,0.7,1.5,3,5")]
    if args.exact_reference and args.n > 12:
        parser.error("--exact-reference requires n <= 12")
    if args.require_correctness and not args.exact_reference:
        parser.error("--require-correctness also requires --exact-reference")
    run_loop(args)


if __name__ == "__main__":
    main()
