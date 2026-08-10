"""Equal-rank n=20 MAX2SAT pilot: fixed, beta-only PT, and QCPT."""

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import pt_convergence_gap as benchmark
import validate_qcpt as fixture


PATH = ((0.25, 1.50), (0.70, 0.75), (1.60, 0.25), (3.50, 0.00))


def run(command, directory, log_name):
    with (Path(directory) / log_name).open("w") as log:
        subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)


def write_parameters(path, beta, gamma, args, seed):
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
""")


def prepare_instance(root, n, seed, target_gamma):
    terms, observable, metadata = benchmark.write_max2sat_instance(root, n, seed)
    fixed, gamma = [], []
    for term in terms:
        (gamma if " X" in (" " + term) else fixed).append(term)
    (root / "H_fixed.txt").write_text("\n".join(fixed) + "\n")
    (root / "H_gamma.txt").write_text("\n".join(gamma) + "\n")
    (root / "transverse_magnetization.txt").write_text("\n".join(observable) + "\n")
    combined = []
    for term in fixed + gamma:
        tokens = term.split()
        if "X" in tokens:
            tokens[0] = f"{float(tokens[0]) * target_gamma:.17g}"
        combined.append(" ".join(tokens))
    (root / "H.txt").write_text("\n".join(combined) + "\n")
    (root / "instance.json").write_text(json.dumps(metadata, indent=2) + "\n")


def stage_sources(stage, source_root):
    stage.mkdir(parents=True, exist_ok=True)
    for name in ("prepare.cpp", "mainqmc.hpp", "divdiff.hpp", "pt_schedule.hpp", "beta_anneal.hpp",
                 "PMRQMC_mpi.cpp", "PMRQMC_pt_mpi.cpp", "PMRQMC_qcpt_mpi.cpp"):
        shutil.copy2(source_root / name, stage / name)


def prepare(stage, split):
    run(["g++", "-O2", "-std=c++11", "-o", "prepare.bin", "prepare.cpp"], stage, "prepare_build.log")
    if split:
        command = ["./prepare.bin", "--hamiltonian-fixed", "H_fixed.txt", "--hamiltonian-gamma",
                   "H_gamma.txt", "transverse_magnetization.txt"]
    else:
        command = ["./prepare.bin", "H.txt", "transverse_magnetization.txt"]
    run(command, stage, "prepare.log")


def mpi(args, ranks):
    return ["mpirun"] + (["--oversubscribe"] if args.oversubscribe else []) + ["-n", str(ranks)]


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def trace_stats(path, slot=None):
    data = rows(path)
    if slot is not None:
        data = [row for row in data if int(row.get("slot", row.get("temperature"))) == slot]
    return {"energy": fixture.ratio_stats(data, "obs_1"),
            "magnetization": fixture.ratio_stats(data, "obs_0"),
            "sign": sum(float(row["sign"]) for row in data) / len(data) if data else float("nan"),
            "measurements": len(data), "rows": data}


def pt_diagnostics(stage, prefix, ranks, ladders, trace, wall):
    swaps = rows(stage / f"{prefix}_swaps.csv")
    flow = rows(stage / f"{prefix}_flow.csv")
    attempts = [int(row["attempts"]) for row in swaps]
    acceptances = [float(row["acceptance"]) for row in swaps]
    cold = trace_stats(trace, len(PATH) - 1)
    iat = benchmark.integrated_autocorrelation_time(
        [(int(row["updates"]), float(row["signed_obs_1"]), float(row["sign"])) for row in cold["rows"]]
    )
    effective = cold["measurements"] * ladders / (iat or 1.0)
    core_hours = wall * ranks / 3600.0
    return {"wall_seconds": wall, "ranks": ranks, "independent_ladders": ladders,
            "energy": {key: value for key, value in cold["energy"].items() if key != "measurements"},
            "magnetization": {key: value for key, value in cold["magnetization"].items() if key != "measurements"},
            "average_sign": cold["sign"], "effective_samples": effective,
            "effective_samples_per_core_hour": effective / core_hours if core_hours else float("nan"),
            "integrated_autocorrelation_measurements": iat,
            "adjacent_acceptance": acceptances, "attempts": attempts,
            "path_occupancy": {row["temperature"]: int(row["visits"]) for row in flow},
            "endpoint_visits": max(int(row["endpoint_visits"]) for row in flow),
            "round_trips": max(int(row["round_trips"]) for row in flow),
            "mean_q": float(flow[0]["mean_q"]), "max_q": float(flow[0]["max_q"]),
            "qmax_hit": any(int(row["qmax_achieved"]) for row in flow),
            "crossed_weight_seconds": float(flow[0]["crossed_weight_seconds"]),
            "crossed_weight_fraction": float(flow[0]["crossed_weight_seconds"]) / wall if wall else float("nan")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--instance-seed", type=int, default=2019)
    parser.add_argument("--qmc-seed", type=int, default=1000)
    parser.add_argument("--Tsteps", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--nbins", type=int, default=20)
    parser.add_argument("--qmax", type=int, default=256)
    parser.add_argument("--updates-per-exchange", type=int, default=10)
    parser.add_argument("--oversubscribe", action="store_true")
    args = parser.parse_args()
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1]
    instance = root / "instance"; instance.mkdir(exist_ok=True)
    prepare_instance(instance, args.n, args.instance_seed, PATH[1][1])
    results = {}
    ranks = len(PATH) * 2

    fixed = root / "fixed_control"; stage_sources(fixed, source_root)
    for name in ("H.txt", "transverse_magnetization.txt", "instance.json"):
        shutil.copy2(instance / name, fixed / name)
    write_parameters(fixed / "parameters.hpp", PATH[-1][0], PATH[1][1], args, args.qmc_seed)
    prepare(fixed, False)
    run(["mpicxx", "-O2", "-std=c++11", "-o", "PMRQMC_mpi.bin", "PMRQMC_mpi.cpp"], fixed, "compile.log")
    started = time.perf_counter(); run(mpi(args, ranks) + ["./PMRQMC_mpi.bin", "--timeseries-prefix", "trace.csv"], fixed, "run.log")
    fixed_wall = time.perf_counter() - started
    fixed_stats = trace_stats(fixed / "trace.csv")
    results["fixed_control"] = {"wall_seconds": fixed_wall, "ranks": ranks,
                                 "average_sign": fixed_stats["sign"],
                                 "energy": fixed_stats["energy"], "magnetization": fixed_stats["magnetization"]}

    beta = root / "beta_only_pt"; stage_sources(beta, source_root)
    for name in ("H.txt", "transverse_magnetization.txt", "instance.json"):
        shutil.copy2(instance / name, beta / name)
    write_parameters(beta / "parameters.hpp", PATH[0][0], PATH[1][1], args, args.qmc_seed)
    (beta / "schedule.txt").write_text("# beta tau\n" + "".join(f"{b} {b/2}\n" for b, _ in PATH))
    prepare(beta, False)
    run(["mpicxx", "-O2", "-std=c++11", "-o", "PMRQMC_pt_mpi.bin", "PMRQMC_pt_mpi.cpp"], beta, "compile.log")
    started = time.perf_counter(); run(mpi(args, ranks) + ["./PMRQMC_pt_mpi.bin", "--schedule", "schedule.txt",
             "--updates-per-exchange", str(args.updates_per_exchange), "--independent-ladders", "2",
             "--output-prefix", "beta", "--timeseries-prefix", "trace.csv"], beta, "run.log")
    results["beta_only_pt"] = pt_diagnostics(beta, "beta", ranks, 2, beta / "trace.csv", time.perf_counter() - started)

    qcpt = root / "mixed_qcpt"; stage_sources(qcpt, source_root)
    for name in ("H_fixed.txt", "H_gamma.txt", "transverse_magnetization.txt", "instance.json"):
        shutil.copy2(instance / name, qcpt / name)
    write_parameters(qcpt / "parameters.hpp", PATH[0][0], PATH[0][1], args, args.qmc_seed)
    prepare(qcpt, True)
    run(["mpicxx", "-O2", "-std=c++11", "-DPMR_QCPT", "-o", "PMRQMC_qcpt_mpi.bin",
         "PMRQMC_qcpt_mpi.cpp"], qcpt, "compile.log")
    (qcpt / "schedule.txt").write_text("# beta gamma\n" + "".join(f"{b} {g}\n" for b, g in PATH))
    started = time.perf_counter(); run(mpi(args, ranks) + ["./PMRQMC_qcpt_mpi.bin", "--schedule", "schedule.txt",
             "--updates-per-exchange", str(args.updates_per_exchange), "--independent-ladders", "2",
             "--output-prefix", "qcpt", "--timeseries-prefix", "trace.csv"], qcpt, "run.log")
    results["mixed_qcpt"] = pt_diagnostics(qcpt, "qcpt", ranks, 2, qcpt / "trace.csv", time.perf_counter() - started)
    report = {"n": args.n, "instance_seed": args.instance_seed, "qmc_seed": args.qmc_seed,
              "path": PATH, "equal_ranks": ranks, "results": results,
              "files": {"instance": str(instance), "root": str(root)}}
    (root / "pilot_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
