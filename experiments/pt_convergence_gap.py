"""Compare fixed-beta and beta-tempered convergence on a larger instance.

The two runs use the same generated Hamiltonian, observable, update budget, and
MPI size.  The fixed-beta run puts every rank at the target beta.  The PT run
uses the same ranks as independent beta ladders and measures the cold slot.
Both MPI binaries emit one row per measurement, allowing this script to report
the first sustained entry into a tolerance band around a long-run PT estimate.

This is an experiment harness, not a claim that every instance benefits from
tempering.  Repeat it over several seeds and inspect ``comparison_report.json``
alongside the raw traces before drawing a performance conclusion.
"""

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_BETAS = (0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0)
SOURCE_FILES = (
    "prepare.cpp", "PMRQMC_mpi.cpp", "PMRQMC_pt_mpi.cpp", "mainqmc.hpp",
    "divdiff.hpp", "pt_schedule.hpp", "beta_anneal.hpp",
)


def random_3regular_graph(n, seed):
    if n < 4 or n % 2:
        raise ValueError("n must be an even integer >= 4")
    rng = random.Random(seed)
    for _ in range(1000):
        stubs = [vertex for vertex in range(n) for _ in range(3)]
        rng.shuffle(stubs)
        edges = set()
        valid = True
        for left, right in zip(stubs[::2], stubs[1::2]):
            edge = tuple(sorted((left, right)))
            if left == right or edge in edges:
                valid = False
                break
            edges.add(edge)
        if valid:
            return sorted(edges)
    raise RuntimeError("could not generate a simple 3-regular graph")


def write_pauli_term(coefficient, sites, operators):
    tokens = [f"{coefficient:.17g}"]
    for site, operator in zip(sites, operators):
        tokens.extend((str(site + 1), operator))
    return " ".join(tokens)


def write_tfim_instance(directory, n, gamma, seed):
    edges = random_3regular_graph(n, seed)
    hamiltonian = [write_pauli_term(-1.0, edge, ("Z", "Z")) for edge in edges]
    hamiltonian += [write_pauli_term(-gamma, (vertex,), ("X",)) for vertex in range(n)]
    observable = [write_pauli_term(1.0 / n, (vertex,), ("X",)) for vertex in range(n)]
    metadata = {"model": "random_3_regular_transverse_field_ising", "n": n,
                "gamma": gamma, "seed": seed, "edges": edges}
    return hamiltonian, observable, metadata


def write_max2sat_instance(directory, n, seed):
    rng = random.Random(seed)
    for _ in range(1000):
        stubs = [vertex for vertex in range(n) for _ in range(3)]
        rng.shuffle(stubs)
        if any(left == right for left, right in zip(stubs[::2], stubs[1::2])):
            continue
        clauses = [(left, rng.choice((-1, 1)), right, rng.choice((-1, 1)))
                   for left, right in zip(stubs[::2], stubs[1::2])]
        terms = []
        for left, sign_left, right, sign_right in clauses:
            terms.extend((write_pauli_term(0.25, (), ()),
                          write_pauli_term(-0.25 * sign_left, (left,), ("Z",)),
                          write_pauli_term(-0.25 * sign_right, (right,), ("Z",)),
                          write_pauli_term(0.25 * sign_left * sign_right, (left, right), ("Z", "Z"))))
        terms += [write_pauli_term(-0.25, (vertex,), ("X",)) for vertex in range(n)]
        observable = [write_pauli_term(1.0 / n, (vertex,), ("X",)) for vertex in range(n)]
        metadata = {"model": "random_3_regular_max2sat", "n": n, "seed": seed,
                    "clauses": clauses}
        return terms, observable, metadata
    raise RuntimeError("could not generate a 3-regular MAX2SAT instance")


def write_parameters(directory, beta, Tsteps, steps, steps_per_measurement, qmax, nbins,
                     rng_seed_offset):
    text = f"""
#define Tsteps {int(Tsteps)}
#define steps {int(steps)}
#define stepsPerMeasurement {int(steps_per_measurement)}
#define beta {float(beta):.17g}
#define tau {float(beta) / 2.0:.17g}
#define parity_cond 0
#define qmax {int(qmax)}
#define Nbins {int(nbins)}
#define EXHAUSTIVE_CYCLE_SEARCH
#define GAPS_GEOMETRIC_PARAMETER 0.8
#define COMPOSITE_UPDATE_BREAK_PROBABILITY 0.9
#define EXACTLY_REPRODUCIBLE
#define RNG_SEED_OFFSET {int(rng_seed_offset)}
"""
    (Path(directory) / "parameters.hpp").write_text(text)


def stage_run_directory(directory, source_root, instance_directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copy2(Path(source_root) / name, directory / name)
    for name in ("H.txt", "transverse_magnetization.txt", "instance.json"):
        shutil.copy2(Path(instance_directory) / name, directory / name)


def write_schedule(path, betas):
    Path(path).write_text("# beta tau\n" + "".join(f"{b:.17g} {b / 2.0:.17g}\n" for b in betas))


def run_checked(command, directory, log_name):
    directory = Path(directory)
    with (directory / log_name).open("w") as log:
        subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)


def read_trace(path, temperature=None, observable="obs_0"):
    with Path(path).open(newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if temperature is not None:
        rows = [row for row in rows if int(row["temperature"]) == temperature]
    signed_observable = f"signed_{observable}"
    if rows and signed_observable not in rows[0]:
        raise ValueError(
            f"trace is missing {signed_observable}; regenerate it with the corrected "
            "signed-numerator time-series format"
        )
    return [(int(row["updates"]), float(row[signed_observable]), float(row["sign"]))
            for row in rows]


def cumulative_ratio_trace(trace):
    result = []
    numerator = 0.0
    denominator = 0.0
    for updates, signed_value, sign in trace:
        numerator += signed_value
        denominator += sign
        result.append((updates, numerator / denominator if denominator else float("nan")))
    return result


def block_ratio_values(trace, block_measurements):
    """Non-overlapping ratio estimates used for a rough tail scale."""
    values = []
    for start in range(0, len(trace), block_measurements):
        block = trace[start:start + block_measurements]
        if len(block) < block_measurements:
            continue
        numerator = sum(row[1] for row in block)
        denominator = sum(row[2] for row in block)
        if denominator:
            values.append(numerator / denominator)
    return values


def block_standard_error(trace, block_measurements):
    values = block_ratio_values(trace, block_measurements)
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return (variance / len(values)) ** 0.5


def integrated_autocorrelation_time(trace):
    """Initial-positive-sequence IAT for the linearized ratio estimator."""
    if len(trace) < 4:
        return None
    estimate = cumulative_ratio_trace(trace)[-1][1]
    values = [signed_value - estimate * sign for _, signed_value, sign in trace]
    mean = sum(values) / len(values)
    values = [value - mean for value in values]
    variance = sum(value * value for value in values) / len(values)
    if variance <= 0.0:
        return 1.0
    correlations = []
    # Cap the pure-Python calculation; a positive sequence reaching this cap is
    # itself a warning that the trace or block length should be increased.
    max_lag = min(len(values) // 2, 1000)
    for lag in range(1, max_lag + 1):
        covariance = sum(values[index] * values[index + lag]
                         for index in range(len(values) - lag)) / (len(values) - lag)
        correlations.append(covariance / variance)
    included = []
    for start in range(0, len(correlations) - 1, 2):
        pair = correlations[start] + correlations[start + 1]
        if pair <= 0.0:
            break
        included.extend(correlations[start:start + 2])
    return max(1.0, 1.0 + 2.0 * sum(included))


def convergence_update(trace, target, tolerance, block_measurements, stability_blocks):
    """Return the first update with a sustained cumulative-mean tolerance hit."""
    if not trace:
        return None
    ratios = cumulative_ratio_trace(trace)
    checkpoints = []
    for index, (updates, value) in enumerate(ratios, start=1):
        if index % block_measurements == 0 or index == len(trace):
            checkpoints.append((updates, value))
    for start in range(0, len(checkpoints) - stability_blocks + 1):
        # Require at least stability_blocks remaining and no later escape from
        # the band.  A brief early crossing is not convergence.
        if all(abs(mean - target) <= tolerance for _, mean in checkpoints[start:]):
            return checkpoints[start][0]
    return None


def summarize_trace(trace, target, tolerance, args, independent_streams):
    if not trace:
        return {"measurements": 0, "final_mean": None, "convergence_updates": None}
    ratio_trace = cumulative_ratio_trace(trace)
    final_mean = ratio_trace[-1][1]
    standard_error = block_standard_error(trace, args.block_measurements)
    autocorrelation = integrated_autocorrelation_time(trace)
    effective_samples = (len(trace) * independent_streams / autocorrelation
                         if autocorrelation else None)
    return {
        "measurements": len(trace),
        "final_mean": final_mean,
        "mean_sign": sum(row[2] for row in trace) / len(trace),
        "block_standard_error": standard_error,
        "integrated_autocorrelation_measurements": autocorrelation,
        "effective_samples": effective_samples,
        "accuracy_tolerance": tolerance,
        "convergence_updates": convergence_update(
            trace, target, tolerance, args.block_measurements, args.stability_blocks
        ),
        "last_update": trace[-1][0],
    }


def make_instance(directory, model, n, gamma, seed):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if model == "tfim":
        hamiltonian, observable, metadata = write_tfim_instance(directory, n, gamma, seed)
    else:
        hamiltonian, observable, metadata = write_max2sat_instance(directory, n, seed)
    (directory / "H.txt").write_text("\n".join(hamiltonian) + "\n")
    (directory / "transverse_magnetization.txt").write_text("\n".join(observable) + "\n")
    (directory / "instance.json").write_text(json.dumps(metadata, indent=2) + "\n")


def run_experiment(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1]
    instance = output / "instance"
    make_instance(instance, args.model, args.n, args.gamma, args.seed)

    fixed = output / "fixed_beta"
    pt = output / "beta_tempered"
    for directory in (fixed, pt):
        stage_run_directory(directory, source_root, instance)

    betas = tuple(float(value) for value in args.betas.split(",") if value.strip())
    if len(betas) < 2 or any(left >= right for left, right in zip(betas, betas[1:])):
        raise ValueError("--betas must contain at least two strictly increasing values")
    target_beta = betas[-1]
    ranks = len(betas) * args.ladders
    write_schedule(pt / "tempering_schedule.txt", betas)
    write_parameters(fixed, target_beta, args.Tsteps, args.steps, args.steps_per_measurement,
                     args.qmax, args.nbins, args.rng_seed_offset)
    write_parameters(pt, betas[0], args.Tsteps, args.steps, args.steps_per_measurement,
                     args.qmax, args.nbins, args.rng_seed_offset)

    run_checked(["g++", "-O3", "-std=c++11", "-o", "prepare.bin", "prepare.cpp"], fixed, "prepare.log")
    run_checked(["./prepare.bin", "H.txt", "transverse_magnetization.txt"], fixed, "prepare_observable.log")
    shutil.copy2(fixed / "hamiltonian.hpp", pt / "hamiltonian.hpp")

    run_checked(["mpicxx", "-O3", "-std=c++11", "-o", "PMRQMC_mpi.bin", "PMRQMC_mpi.cpp"], fixed, "compile.log")
    shutil.copy2(fixed / "hamiltonian.hpp", pt / "hamiltonian.hpp")
    run_checked(["g++", "-O3", "-std=c++11", "-o", "prepare.bin", "prepare.cpp"], pt, "prepare.log")
    run_checked(["./prepare.bin", "H.txt", "transverse_magnetization.txt"], pt, "prepare_observable.log")
    run_checked(["mpicxx", "-O3", "-std=c++11", "-o", "PMRQMC_pt_mpi.bin", "PMRQMC_pt_mpi.cpp"], pt, "compile.log")

    fixed_trace = fixed / "trace.csv"
    pt_trace = pt / "trace.csv"
    mpirun_prefix = ["mpirun"] + (["--oversubscribe"] if args.oversubscribe else [])
    fixed_command = mpirun_prefix + ["-n", str(ranks), "./PMRQMC_mpi.bin", "--timeseries-prefix", str(fixed_trace)]
    pt_command = mpirun_prefix + ["-n", str(ranks), "./PMRQMC_pt_mpi.bin", "--schedule", str(pt / "tempering_schedule.txt"),
                  "--updates-per-exchange", str(args.updates_per_exchange), "--independent-ladders", str(args.ladders),
                  "--output-prefix", "pt", "--timeseries-prefix", str(pt_trace)]

    started = time.perf_counter()
    run_checked(fixed_command, fixed, "run.log")
    fixed_wall = time.perf_counter() - started
    started = time.perf_counter()
    run_checked(pt_command, pt, "run.log")
    pt_wall = time.perf_counter() - started

    fixed_data = read_trace(fixed_trace)
    cold_data = read_trace(pt_trace, temperature=len(betas) - 1)
    cold_tail = cold_data[max(0, len(cold_data) * 3 // 4):]
    tail = block_ratio_values(cold_tail, args.block_measurements)
    if not tail:
        tail = [cumulative_ratio_trace(cold_tail)[-1][1]]
    pt_tail_reference = sum(tail) / len(tail)
    tail_variance = sum((value - pt_tail_reference) ** 2 for value in tail) / max(1, len(tail) - 1)
    tail_stdev = tail_variance ** 0.5
    exact_reference = None
    reference_source = "pt_cold_slot_tail"
    reference = pt_tail_reference
    if args.exact_reference:
        if args.n > 12:
            raise ValueError("--exact-reference is restricted to n <= 12 for dense diagonalization")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from exact_diagonalization import exact_expectation
        exact_reference = exact_expectation(
            instance / "H.txt", instance / "transverse_magnetization.txt", args.n, target_beta
        )
        reference = exact_reference["exact_mean"]
        reference_source = "exact_diagonalization"
        (output / "exact_reference.json").write_text(json.dumps({
            "model": json.loads((instance / "instance.json").read_text())["model"],
            "n": args.n,
            "observable": "transverse_magnetization",
            "values": [exact_reference],
            "cold_beta": target_beta,
            "cold_exact_mean": reference,
        }, indent=2) + "\n")
    tail_standard_error = tail_stdev / (len(tail) ** 0.5)
    pt_standard_error = block_standard_error(cold_data, args.block_measurements)
    fixed_standard_error = block_standard_error(fixed_data, args.block_measurements)
    pt_tolerance = max(args.absolute_tolerance,
                       args.tolerance_sigma * (pt_standard_error or 0.0))
    fixed_tolerance = max(args.absolute_tolerance,
                          args.tolerance_sigma * (fixed_standard_error or 0.0))
    report = {
        "model": args.model,
        "n": args.n,
        "gamma": args.gamma,
        "seed": args.seed,
        "rng_seed_offset": args.rng_seed_offset,
        "target_beta": target_beta,
        "betas": betas,
        "independent_ladders": args.ladders,
        "mpi_ranks_each_run": ranks,
        "Tsteps": args.Tsteps,
        "steps": args.steps,
        "qmax": args.qmax,
        "reference": {"source": reference_source, "value": reference,
                      "cold_pt_tail_mean": pt_tail_reference, "cold_pt_tail_stdev": tail_stdev,
                      "cold_pt_tail_standard_error": tail_standard_error,
                      "tolerance": pt_tolerance, "exact": exact_reference},
        "fixed_beta": {"wall_seconds": fixed_wall,
                       **summarize_trace(fixed_data, reference, fixed_tolerance, args, ranks)},
        "beta_tempered_cold_slot": {"wall_seconds": pt_wall,
                                    **summarize_trace(cold_data, reference, pt_tolerance, args, args.ladders)},
        "files": {"fixed_trace": str(fixed_trace), "pt_trace": str(pt_trace),
                  "swaps": str(pt / "pt_swaps.csv"), "flow": str(pt / "pt_flow.csv")},
        "interpretation": "Convergence is the first sustained cumulative-mean entry into the reported tolerance band; repeat seeds before treating the ratio as a general speedup.",
    }
    fixed_time = report["fixed_beta"]["convergence_updates"]
    pt_time = report["beta_tempered_cold_slot"]["convergence_updates"]
    report["fixed_beta"]["effective_samples_per_second"] = (
        report["fixed_beta"]["effective_samples"] / fixed_wall)
    report["beta_tempered_cold_slot"]["effective_samples_per_second"] = (
        report["beta_tempered_cold_slot"]["effective_samples"] / pt_wall)
    if fixed_time is not None and pt_time is not None:
        report["convergence_update_ratio_fixed_over_pt"] = fixed_time / pt_time if pt_time else None
    (output / "comparison_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--model", choices=("tfim", "max2sat"), default="tfim")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--rng-seed-offset", type=int, default=1000,
                        help="first deterministic QMC RNG seed; rank is added to this value")
    parser.add_argument("--betas", default=",".join(str(value) for value in DEFAULT_BETAS))
    parser.add_argument("--ladders", type=int, default=2)
    parser.add_argument("--Tsteps", type=int, default=100000)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--steps-per-measurement", type=int, default=100)
    parser.add_argument("--updates-per-exchange", type=int, default=10)
    parser.add_argument("--oversubscribe", action="store_true",
                        help="pass OpenMPI's --oversubscribe flag (useful on a workstation)")
    parser.add_argument("--qmax", type=int, default=1000)
    parser.add_argument("--nbins", type=int, default=100)
    parser.add_argument("--block-measurements", type=int, default=10)
    parser.add_argument("--stability-blocks", type=int, default=10)
    parser.add_argument("--tolerance-sigma", type=float, default=0.5)
    parser.add_argument("--absolute-tolerance", type=float, default=0.01)
    parser.add_argument("--exact-reference", action="store_true",
                        help="use dense exact diagonalization for the cold-beta reference (n <= 12)")
    args = parser.parse_args()
    if args.ladders < 1 or args.steps < args.nbins * args.steps_per_measurement:
        parser.error("require --ladders >= 1 and steps >= nbins * steps-per-measurement")
    run_experiment(args)


if __name__ == "__main__":
    main()
