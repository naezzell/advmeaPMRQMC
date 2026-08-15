"""Run and summarize one custom-beta PT trial for the Figure 4 instance."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

try:
    from experiments.paper_tfim_3regular import write_instance
    from experiments.run_paper_tfim_3regular import (
        DEFAULT_CATALOG,
        SOURCE_FILES,
        load_catalog,
        prepare_run_directory,
    )
except ModuleNotFoundError:
    from paper_tfim_3regular import write_instance
    from run_paper_tfim_3regular import (
        DEFAULT_CATALOG,
        SOURCE_FILES,
        load_catalog,
        prepare_run_directory,
    )


def comma_floats(value: str) -> list[float]:
    values = [float(token) for token in value.split(",")]
    if len(values) < 2 or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("betas must contain at least two positive values")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise argparse.ArgumentTypeError("betas must be strictly increasing")
    return values


def run(command: list[str], cwd: Path, log_name: str) -> None:
    with (cwd / log_name).open("w") as stream:
        subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)


def summarize(run_dir: Path, n: int, gamma: float, betas: list[float]) -> dict:
    with (run_dir / "pmr_swaps.csv").open() as stream:
        swap_rows = list(csv.DictReader(stream))
    edges = []
    for row in swap_rows:
        edge = int(row["edge"])
        edges.append({
            "beta_left": betas[edge],
            "beta_right": betas[edge + 1],
            "attempts": int(row["attempts"]),
            "accepted": int(row["accepted"]),
            "acceptance": float(row["acceptance"]),
        })

    with (run_dir / "pmr_flow.csv").open() as stream:
        flow = next(csv.DictReader(stream))
    with (run_dir / "pmr_rank_timing.csv").open() as stream:
        timings = list(csv.DictReader(stream))
    elapsed = [float(row["elapsed_seconds"]) for row in timings]

    with (run_dir / "pmr_observables.csv").open() as stream:
        observable_rows = list(csv.DictReader(stream))
    plotted = {}
    for beta in (2.0, 10.0, 50.0):
        matches = [row for row in observable_rows
                   if float(row["beta"]) == beta and row["name"] == "H_{offdiag}"]
        if matches:
            row = matches[0]
            mean = float(row["mean"])
            stdev = float(row["stdev"])
            plotted[str(int(beta))] = {
                "H_offdiag": mean,
                "H_offdiag_stdev": stdev,
                "minus_H_offdiag_per_spin": -mean / n,
                "x_magnetization_per_spin": -mean / (gamma * n),
                "x_magnetization_stdev": stdev / (gamma * n),
            }

    acceptance = [edge["acceptance"] for edge in edges]
    return {
        "edges": edges,
        "acceptance_min": min(acceptance),
        "acceptance_max": max(acceptance),
        "edges_in_20_to_40_percent": sum(0.2 <= value <= 0.4 for value in acceptance),
        "edge_count": len(edges),
        "round_trips": int(flow["round_trips"]),
        "mean_q": float(flow["mean_q"]),
        "max_q": float(flow["max_q"]),
        "qmax_achieved": int(flow["qmax_achieved"]),
        "wall_seconds": max(elapsed),
        "rank_core_hours": sum(elapsed) / 3600.0,
        "figure4_observables": plotted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--betas", type=comma_floats, required=True)
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--instance-id", default="instance_001")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--tsteps", type=int, default=20000)
    parser.add_argument("--steps", type=int, default=80000)
    parser.add_argument("--steps-per-measurement", type=int, default=160)
    parser.add_argument("--updates-per-exchange", type=int, default=10)
    parser.add_argument("--nbins", type=int, default=10)
    parser.add_argument("--qmax", type=int, default=1000)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--instances-root", type=Path,
                        default=Path("generated_instances/paper_tfim_3regular"))
    parser.add_argument("--oversubscribe", action="store_true")
    args = parser.parse_args()

    source_root = Path(__file__).resolve().parents[1]
    run_dir = args.output.resolve()
    catalog = load_catalog(args.catalog)
    entry = next((item for item in catalog["instances"]
                  if item["id"] == args.instance_id and item["n"] == args.n), None)
    if entry is None:
        parser.error("instance ID and N are not present in the catalog")
    instance_dir = args.instances_root / f"N{args.n:03d}" / args.instance_id
    if not instance_dir.exists():
        write_instance(instance_dir, args.n, entry["seed"])

    prepare_run_directory(
        run_dir, instance_dir, source_root, args.n, entry["seed"], args.gamma,
        "2019", args.tsteps, args.steps, args.steps_per_measurement,
        args.nbins, args.qmax,
    )
    for source in SOURCE_FILES:
        shutil.copy2(source_root / source, run_dir / source)
    with (run_dir / "schedule.txt").open("w") as stream:
        stream.write("# beta tau\n")
        for beta in args.betas:
            stream.write(f"{beta:.17g} {beta / 2.0:.17g}\n")

    manifest = {
        "instance_id": args.instance_id,
        "n": args.n,
        "seed": entry["seed"],
        "gamma": args.gamma,
        "betas": args.betas,
        "tsteps": args.tsteps,
        "steps": args.steps,
        "steps_per_measurement": args.steps_per_measurement,
        "updates_per_exchange": args.updates_per_exchange,
        "nbins": args.nbins,
        "qmax": args.qmax,
    }
    (run_dir / "trial_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    run(["mpicxx", "-O2", "-std=c++11", "-o", "PMRQMC_pt_mpi.bin", "PMRQMC_pt_mpi.cpp"],
        run_dir, "compile.log")
    command = ["mpirun"]
    if args.oversubscribe:
        command.append("--oversubscribe")
    command += ["-n", str(len(args.betas)), "./PMRQMC_pt_mpi.bin",
                "--schedule", "schedule.txt",
                "--updates-per-exchange", str(args.updates_per_exchange),
                "--output-prefix", "pmr", "--timeseries-prefix", "timeseries.csv"]
    run(command, run_dir, "run.log")
    summary = summarize(run_dir, args.n, args.gamma, args.betas)
    (run_dir / "trial_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
