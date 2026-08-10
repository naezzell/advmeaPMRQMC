"""Generate and compare a tiny 3-regular MAX2SAT PMR example.

The generated Hamiltonian is the MAX2SAT cost function plus a small transverse
field, so the PMR chain has nontrivial off-diagonal updates.  The comparison is
deliberately modest: it demonstrates temperature mobility and mixing signals,
not a publication-quality speed benchmark.
"""

import argparse
import csv
import json
import random
import re
import shutil
import subprocess
from pathlib import Path


DEFAULT_CLAUSES = [
    (0, 1, 1, -1), (2, 1, 3, 1), (4, -1, 5, 1),
    (0, -1, 2, 1), (1, 1, 4, 1), (3, -1, 5, -1),
    (0, 1, 3, -1), (1, -1, 5, 1), (2, -1, 4, -1),
]


def make_clauses(seed=20260805, n=6):
    """Return a deterministic simple 3-regular MAX2SAT clause list."""
    if n != 6:
        raise ValueError("the checked-in small demonstration uses n=6")
    # Keep the example human-readable while retaining a seed in metadata.
    clauses = list(DEFAULT_CLAUSES)
    rng = random.Random(seed)
    signs = [rng.choice((-1, 1)) for _ in range(len(clauses))]
    return [(left, sl * signs[i], right, sr) for i, (left, sl, right, sr) in enumerate(clauses)]


def _add(terms, key, coefficient):
    terms[key] = terms.get(key, 0.0) + coefficient


def write_instance(directory, seed=20260805, gamma=0.25):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    clauses = make_clauses(seed)
    terms = {}
    for left, sign_left, right, sign_right in clauses:
        # (1-s_i Z_i)(1-s_j Z_j)/4 is the unsatisfied-clause projector.
        _add(terms, (), 0.25)
        _add(terms, ((left,), ("Z",)), -0.25 * sign_left)
        _add(terms, ((right,), ("Z",)), -0.25 * sign_right)
        _add(terms, ((left, right), ("Z", "Z")), 0.25 * sign_left * sign_right)
    for vertex in range(6):
        _add(terms, ((vertex,), ("X",)), -gamma)

    def line(key, coefficient):
        if key == ():
            return f"{coefficient:.8f}"
        sites, operators = key
        tokens = [f"{coefficient:.8f}"]
        for site, operator in zip(sites, operators):
            tokens += [str(site + 1), operator]
        return " ".join(tokens)

    (directory / "H.txt").write_text("\n".join(line(key, value) for key, value in terms.items() if abs(value) > 1e-14) + "\n")
    mx = {( (vertex,), ("X",)): 1.0 / 6.0 for vertex in range(6)}
    (directory / "transverse_magnetization.txt").write_text(
        "\n".join(line(key, value) for key, value in mx.items()) + "\n"
    )
    metadata = {"n": 6, "seed": seed, "gamma": gamma, "clauses": clauses, "model": "3-regular-max2sat-plus-transverse-field"}
    (directory / "instance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def write_parameters(directory, beta, Tsteps, steps, steps_per_measurement, qmax=200, nbins=20):
    text = f"""
#define Tsteps {int(Tsteps)}
#define steps {int(steps)}
#define stepsPerMeasurement {int(steps_per_measurement)}
#define beta {float(beta):.17g}
#define tau {float(beta)/2:.17g}
#define parity_cond 0
#define MEASURE_H
#define MEASURE_H2
#define MEASURE_Z_MAGNETIZATION
#define qmax {int(qmax)}
#define Nbins {int(nbins)}
#define EXHAUSTIVE_CYCLE_SEARCH
#define GAPS_GEOMETRIC_PARAMETER 0.8
#define COMPOSITE_UPDATE_BREAK_PROBABILITY 0.9
#define EXACTLY_REPRODUCIBLE
"""
    (Path(directory) / "parameters.hpp").write_text(text)


def stage_sources(directory, source_root):
    directory = Path(directory)
    source_root = Path(source_root)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("prepare.cpp", "PMRQMC.cpp", "PMRQMC_pt_mpi.cpp", "mainqmc.hpp", "divdiff.hpp", "pt_schedule.hpp", "beta_anneal.hpp"):
        shutil.copy2(source_root / name, directory / name)


def _number(pattern, text, default=float("nan")):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else default


def run_comparison(directory, source_root, Tsteps=1000, steps=20000, steps_per_measurement=10):
    directory = Path(directory)
    stage_sources(directory, source_root)
    write_instance(directory)
    write_parameters(directory, beta=2.5, Tsteps=Tsteps, steps=steps,
                     steps_per_measurement=steps_per_measurement)
    subprocess.run(["g++", "-O2", "-std=c++11", "-o", "prepare.bin", "prepare.cpp"], cwd=directory, check=True)
    subprocess.run(["./prepare.bin", "H.txt", "transverse_magnetization.txt"], cwd=directory, check=True)

    subprocess.run(["g++", "-O2", "-std=c++11", "-o", "PMRQMC.bin", "PMRQMC.cpp"], cwd=directory, check=True)
    with (directory / "fixed_beta.log").open("w") as output:
        subprocess.run(["./PMRQMC.bin"], cwd=directory, stdout=output, stderr=subprocess.STDOUT, check=True)

    schedule = directory / "tempering_schedule.txt"
    schedule.write_text("# beta tau\n0.1 0.05\n0.3 0.15\n0.7 0.35\n1.5 0.75\n2.5 1.25\n")
    subprocess.run(["mpicxx", "-O2", "-std=c++11", "-o", "PMRQMC_pt_mpi.bin", "PMRQMC_pt_mpi.cpp"], cwd=directory, check=True)
    subprocess.run(["mpirun", "-n", "5", "./PMRQMC_pt_mpi.bin", "--schedule", str(schedule),
                    "--updates-per-exchange", "10", "--output-prefix", "pt"], cwd=directory, check=True)

    fixed_text = (directory / "fixed_beta.log").read_text()
    fixed = {
        "mean_q": _number(r"mean\(q\) = ([^\n]+)", fixed_text),
        "max_q": _number(r"max\(q\) = ([^\n]+)", fixed_text),
        "mean_sign": _number(r"mean\(sgn\(W\)\) = ([^\n]+)", fixed_text),
    }
    with (directory / "pt_observables.csv").open(newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    cold_rows = [row for row in rows if row["temperature"] == "4"]
    with (directory / "pt_swaps.csv").open(newline="") as input_file:
        swaps = list(csv.DictReader(input_file))
    with (directory / "pt_flow.csv").open(newline="") as input_file:
        flow = list(csv.DictReader(input_file))
    report = {
        "fixed_beta": fixed,
        "tempering_cold_beta": {"beta": 2.5, "observables": cold_rows},
        "swap_acceptance": [float(row["acceptance"]) for row in swaps],
        "round_trips": float(flow[0]["round_trips"]) if flow else float("nan"),
        "cold_slot_visits": sum(int(row["visits"]) for row in flow if row["temperature"] == "4"),
        "note": "PT advantage is shown by temperature mobility and round trips; use longer runs for statistical speed claims.",
    }
    (directory / "comparison_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="isolated working directory for generated files and outputs")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--Tsteps", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=20000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.generate_only:
        print(json.dumps(write_instance(args.directory), indent=2))
    else:
        run_comparison(args.directory, root, args.Tsteps, args.steps)
