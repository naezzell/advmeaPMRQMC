"""Exact n=10 3-regular MAX2SAT QCPT campaign.

The campaign uses the deterministic instance generator from the beta-only
benchmark, separates its diagonal clause cost from the transverse X field,
and compares every QCPT slot against a dense Accelerate reference.  It also
records swap-edge attempts, round trips, signs, qmax, and the generated graph
degree checks.
"""

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

import pt_convergence_gap as benchmark
import validate_qcpt as fixture


PATH = ((0.25, 1.50), (0.70, 0.75), (1.60, 0.25), (3.50, 0.00))


def run(command, directory, log_name):
    with (Path(directory) / log_name).open("w") as log:
        subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)


def write_parameters(path, beta, gamma, args, absolute, seed):
    path.write_text(f"""
#define Tsteps {args.Tsteps}
#define steps {args.steps}
#define stepsPerMeasurement {args.interval}
#define beta {beta:.17g}
#define tau {beta / 2:.17g}
#define gamma {gamma:.17g}
#define parity_cond 0
#define qmax {args.qmax}
#define Nbins {args.nbins}
#define EXHAUSTIVE_CYCLE_SEARCH
#define GAPS_GEOMETRIC_PARAMETER 0.8
#define COMPOSITE_UPDATE_BREAK_PROBABILITY 0.9
#define EXACTLY_REPRODUCIBLE
#define RNG_SEED_OFFSET {seed}
#define MEASURE_H
""" + ("#define ABS_WEIGHTS\n" if absolute else ""))


def make_instance(stage, n, seed):
    terms, observable, metadata = benchmark.write_max2sat_instance(stage, n, seed)
    fixed, gamma = [], []
    for term in terms:
        (gamma if " X" in (" " + term) else fixed).append(term)
    (stage / "H_fixed.txt").write_text("\n".join(fixed) + "\n")
    (stage / "H_gamma.txt").write_text("\n".join(gamma) + "\n")
    (stage / "transverse_magnetization.txt").write_text("\n".join(observable) + "\n")
    (stage / "instance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def prepare(stage):
    run(["g++", "-O2", "-std=c++11", "-o", "prepare.bin", "prepare.cpp"], stage, "prepare_build.log")
    run(["./prepare.bin", "--hamiltonian-fixed", "H_fixed.txt", "--hamiltonian-gamma",
         "H_gamma.txt", "transverse_magnetization.txt"], stage, "prepare.log")


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_summary(path):
    result = {}
    for row in read_rows(path):
        if row["kind"] == "observable" and row["name"] in ("A", "H"):
            stdev = float(row["stdev"])
            result[row["name"]] = (float(row["mean"]), stdev if math.isfinite(stdev) else 0.0)
    return result


def exact_reference(stage, n):
    run(["clang++", "-O2", "-std=c++11", "-framework", "Accelerate", "-o", "exact_split_real",
         "exact_split_real.cpp"], stage, "exact_build.log")
    values = {}
    for beta, gamma in PATH:
        output = subprocess.check_output(["./exact_split_real", str(n), "H_fixed.txt", "H_gamma.txt",
                                          "transverse_magnetization.txt", str(beta), str(gamma)], cwd=stage,
                                         text=True)
        _, _, energy, magnetization = output.split()
        values[(beta, gamma)] = {"E": float(energy), "M": float(magnetization)}
    return values


def run_qcpt(stage, args, absolute):
    write_parameters(stage / "parameters.hpp", PATH[0][0], PATH[0][1], args, absolute, args.qmc_seed)
    run(["mpicxx", "-O2", "-std=c++11", "-DPMR_QCPT", "-o", "PMRQMC_qcpt_mpi.bin",
         "PMRQMC_qcpt_mpi.cpp"], stage, "compile.log")
    (stage / "qcpt_schedule.txt").write_text("# beta gamma\n" + "".join(
        f"{beta:.17g} {gamma:.17g}\n" for beta, gamma in PATH))
    command = ["mpirun"] + (["--oversubscribe"] if args.oversubscribe else []) + ["-n", str(len(PATH)),
              "./PMRQMC_qcpt_mpi.bin", "--schedule", "qcpt_schedule.txt", "--updates-per-exchange",
              str(args.updates_per_exchange), "--output-prefix", "qcpt", "--timeseries-prefix", "trace.csv"]
    run(command, stage, "run.log")
    rows = read_rows(stage / "trace.csv")
    estimates = {}
    for slot, point in enumerate(PATH):
        slot_rows = [row for row in rows if int(row["slot"]) == slot]
        estimates[point] = {"M": fixture.ratio_stats(slot_rows, "obs_0"),
                            "E": fixture.ratio_stats(slot_rows, "obs_1")}
    return estimates


def diagnostics(stage, metadata, estimates, exact):
    swaps = read_rows(stage / "qcpt_swaps.csv")
    flow = read_rows(stage / "qcpt_flow.csv")
    summary = read_summary(stage / "qcpt_observables.csv")
    max_round_trips = max(int(row["round_trips"]) for row in flow)
    qmax_hit = any(int(row["qmax_achieved"]) != 0 for row in flow)
    edge_attempts = [int(row["attempts"]) for row in swaps]
    comparisons = []
    for point in PATH:
        for name, summary_name in (("E", "H"), ("M", "A")):
            estimate = estimates[point][name]
            mean, reported_stdev = summary[summary_name]
            stdev = reported_stdev if math.isfinite(reported_stdev) else estimate["stdev"]
            tolerance = max(0.025, 6.0 * stdev) if math.isfinite(stdev) else 0.025
            comparisons.append({"point": point, "observable": name, "exact": exact[point][name],
                                "estimate": estimate["value"], "stdev": stdev,
                                "absolute_error": abs(estimate["value"] - exact[point][name]),
                                "tolerance": tolerance,
                                "passed": abs(estimate["value"] - exact[point][name]) <= tolerance})
    degrees = [0] * int(metadata["n"])
    for left, _, right, _ in metadata["clauses"]:
        degrees[left] += 1; degrees[right] += 1
    checks = {"all_edges_attempted": len(metadata["clauses"]) == 3 * int(metadata["n"]) // 2,
              "3_regular_clause_degrees": degrees == [3] * int(metadata["n"]),
              "all_swap_edges_attempted": all(value > 0 for value in edge_attempts),
              "at_least_one_round_trip": max_round_trips >= 1,
              "finite_sign_reweighted_estimates": all(math.isfinite(item["estimate"]) for item in comparisons),
              "no_qmax_hit": not qmax_hit}
    return {"comparisons": comparisons, "checks": checks,
            "diagnostics": {"swap_edges": swaps, "flow": flow, "summary": summary},
            "passed": all(checks.values()) and all(item["passed"] for item in comparisons)}


def one_run(root, instance_seed, qmc_seed, args):
    stage = root / f"instance_{instance_seed}_qmc_{qmc_seed}"
    stage.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1]
    for name in ("prepare.cpp", "mainqmc.hpp", "divdiff.hpp", "pt_schedule.hpp", "beta_anneal.hpp",
                 "PMRQMC_pt_mpi.cpp", "PMRQMC_qcpt_mpi.cpp"):
        shutil.copy2(source_root / name, stage / name)
    shutil.copy2(source_root / "experiments/exact_split_real.cpp", stage / "exact_split_real.cpp")
    metadata = make_instance(stage, args.n, instance_seed)
    prepare(stage)
    exact = exact_reference(stage, args.n)
    args.qmc_seed = qmc_seed
    result = diagnostics(stage, metadata, run_qcpt(stage, args, args.absolute_weights), exact)
    result.update({"instance_seed": instance_seed, "qmc_seed": qmc_seed,
                   "artifact": str(stage), "exact": {str(point): value for point, value in exact.items()}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--instance-seeds", default="11,12")
    parser.add_argument("--qmc-seeds", default="1000,2000")
    parser.add_argument("--Tsteps", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--nbins", type=int, default=20)
    parser.add_argument("--qmax", type=int, default=256)
    parser.add_argument("--updates-per-exchange", type=int, default=10)
    parser.add_argument("--oversubscribe", action="store_true")
    parser.add_argument("--absolute-weights", action="store_true")
    args = parser.parse_args()
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=True)
    results = [one_run(root, int(instance), int(qmc), args)
               for instance in args.instance_seeds.split(",") if instance.strip()
               for qmc in args.qmc_seeds.split(",") if qmc.strip()]
    report = {"n": args.n, "path": PATH, "results": results,
              "passed": all(result["passed"] for result in results)}
    (root / "qcpt_max2sat_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
