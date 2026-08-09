"""Validation of the remaining QCPT edge cases from the follow-on plan.

This driver deliberately exercises separate concerns so that a passing exact
two-spin campaign cannot hide a missing path/layout or preparation feature:

* a QCPT path with constant gamma is compared with legacy beta-only PT;
* a fixed-beta uniform transverse-field gamma path is compared with exact
  energies and the expansion-order identity q = -beta*gamma*<H_gamma>;
* two- and three-point paths run with two independent ladders;
* cancellation and repeated-energy preparation/runtime smokes are checked;
* target-weight immutability is checked in-process, including the RNG and
  active divided-difference cache.
"""

import argparse
import csv
import importlib.util
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXED_MIXED = ((-0.90, (0, 1), "ZZ"), (0.20, (0,), "Z"),
               (-0.35, (0,), "X"), (-0.10, (0, 1), "XX"))
GAMMA_MIXED = ((0.25, (1,), "Z"), (-0.15, (0, 1), "ZZ"),
               (-0.55, (1,), "X"), (-0.20, (0, 1), "XX"))
FIXED_TF = ((-0.90, (0, 1), "ZZ"), (0.20, (0,), "Z"))
GAMMA_TF = ((-0.35, (0,), "X"), (-0.35, (1,), "X"))
PURE_BETA = ((0.25, 0.75), (0.70, 0.75), (1.60, 0.75), (3.50, 0.75))
PURE_GAMMA = ((0.90, 0.00), (0.90, 0.50), (0.90, 1.00), (0.90, 1.50))


def load_qcpt_module():
    specification = importlib.util.spec_from_file_location("validate_qcpt", ROOT / "experiments/validate_qcpt.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


QCPT = load_qcpt_module()


def run(command, directory, log=None):
    if log is None:
        subprocess.run(command, cwd=directory, check=True)
        return
    with (directory / log).open("w") as stream:
        subprocess.run(command, cwd=directory, stdout=stream, stderr=subprocess.STDOUT, check=True)


def mpi(oversubscribe, ranks):
    command = ["mpirun"]
    if oversubscribe:
        command.append("--oversubscribe")
    return command + ["-n", str(ranks)]


def term_text(term):
    coefficient, sites, paulis = term
    return " ".join([f"{coefficient:.17g}"] +
                     [token for site, token in zip(sites, paulis) for token in (str(site + 1), token)])


def write_terms(path, terms):
    path.write_text("\n".join(term_text(term) for term in terms) + "\n")


def write_observables(directory):
    (directory / "z1.txt").write_text("1 1 Z\n")
    (directory / "xavg.txt").write_text("0.5 1 X\n0.5 2 X\n")


def write_parameters(directory, beta, gamma, tsteps, steps, nbins):
    (directory / "parameters.hpp").write_text(f"""
#define Tsteps {tsteps}
#define steps {steps}
#define stepsPerMeasurement 10
#define beta {beta:.17g}
#define tau {beta / 2:.17g}
#define gamma {gamma:.17g}
#define parity_cond 0
#define qmax 256
#define Nbins {nbins}
#define EXHAUSTIVE_CYCLE_SEARCH
#define GAPS_GEOMETRIC_PARAMETER 0.8
#define COMPOSITE_UPDATE_BREAK_PROBABILITY 0.9
#define EXACTLY_REPRODUCIBLE
#define RNG_SEED_OFFSET 8100
#define MEASURE_H
""")


def stage_sources(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("prepare.cpp", "mainqmc.hpp", "divdiff.hpp", "pt_schedule.hpp",
                 "PMRQMC_mpi.cpp", "PMRQMC_pt_mpi.cpp", "PMRQMC_qcpt_mpi.cpp",
                 "target_weight_immutability.cpp", "split_component_probe.cpp"):
        shutil.copy2(ROOT / "experiments" / name if name.endswith(".cpp") and name in {
            "target_weight_immutability.cpp", "split_component_probe.cpp"} else ROOT / name,
                     directory / name)


def prepare_split(directory, fixed, gamma, with_observables=True):
    write_terms(directory / "H_fixed.txt", fixed)
    write_terms(directory / "H_gamma.txt", gamma)
    write_observables(directory)
    run(["g++", "-O1", "-std=c++11", "-o", "prepare.bin", "prepare.cpp"], directory)
    command = ["./prepare.bin", "--hamiltonian-fixed", "H_fixed.txt",
               "--hamiltonian-gamma", "H_gamma.txt"]
    if with_observables:
        command += ["z1.txt"] * 5 + ["xavg.txt"]
    run(command, directory)


def compile_binaries(directory, beta=0.7, gamma=0.75, tsteps=5000, steps=100000, nbins=50):
    write_parameters(directory, beta, gamma, tsteps, steps, nbins)
    run(["mpicxx", "-O1", "-std=c++11", "-o", "PMRQMC_mpi.bin", "PMRQMC_mpi.cpp"], directory, "compile_fixed.log")
    run(["mpicxx", "-O1", "-std=c++11", "-o", "PMRQMC_pt_mpi.bin", "PMRQMC_pt_mpi.cpp"], directory, "compile_pt.log")
    run(["mpicxx", "-O1", "-std=c++11", "-DPMR_QCPT", "-o", "PMRQMC_qcpt_mpi.bin", "PMRQMC_qcpt_mpi.cpp"], directory, "compile_qcpt.log")


def write_schedule(path, points, qcpt):
    if qcpt:
        path.write_text("# beta gamma\n" + "".join(f"{beta:.17g} {gamma:.17g}\n" for beta, gamma in points))
    else:
        path.write_text("# beta tau\n" + "".join(f"{beta:.17g} {beta / 2:.17g}\n" for beta, _ in points))


def read_observables(path):
    values = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["kind"] == "observable":
                values[(int(row["slot"] if "slot" in row else row["temperature"]), row["name"])] = {
                    "mean": float(row["mean"]), "stdev": float(row["stdev"])}
    return values


def exact_values(fixed, gamma_terms, beta, gamma):
    h = [[0.0] * 4 for _ in range(4)]
    hg = [[0.0] * 4 for _ in range(4)]
    for coefficient, sites, paulis in fixed:
        QCPT.add_scaled(h, QCPT.operator(2, sites, paulis), coefficient)
    for coefficient, sites, paulis in gamma_terms:
        term = QCPT.operator(2, sites, paulis)
        QCPT.add_scaled(h, term, gamma * coefficient)
        QCPT.add_scaled(hg, term, coefficient)
    xavg = [[0.0] * 4 for _ in range(4)]
    QCPT.add_scaled(xavg, QCPT.operator(2, (0,), "X"), 0.5)
    QCPT.add_scaled(xavg, QCPT.operator(2, (1,), "X"), 0.5)
    energies, vectors = QCPT.jacobi_eigh(h)
    shift = min(energies)
    weights = [math.exp(-beta * (energy - shift)) for energy in energies]

    def expectation(operator):
        values = []
        for column in range(4):
            values.append(sum(vectors[i][column] * operator[i][j] * vectors[j][column]
                              for i in range(4) for j in range(4)))
        return sum(weight * value for weight, value in zip(weights, values)) / sum(weights)

    return {"E": expectation(h), "Xavg": expectation(xavg),
            "q": -beta * gamma * expectation(hg)}


def assert_close(value, expected, tolerance, label):
    error = abs(value - expected)
    if not math.isfinite(value) or error > tolerance:
        raise RuntimeError(f"{label}: value={value} expected={expected} error={error} tolerance={tolerance}")
    return error


def run_path(directory, points, qcpt, prefix, ranks, oversubscribe):
    schedule = directory / (prefix + "_schedule.txt")
    write_schedule(schedule, points, qcpt)
    binary = "PMRQMC_qcpt_mpi.bin" if qcpt else "PMRQMC_pt_mpi.bin"
    command = mpi(oversubscribe, ranks) + ["./" + binary, "--schedule", schedule.name,
        "--updates-per-exchange", "10", "--independent-ladders", str(ranks // len(points)),
        "--output-prefix", prefix]
    run(command, directory, prefix + ".log")
    return read_observables(directory / (prefix + "_observables.csv"))


def run_fixed(directory, beta, gamma, fixed, gamma_terms, oversubscribe):
    write_parameters(directory, beta, gamma, 5000, 50000, 50)
    run(["mpicxx", "-O1", "-std=c++11", "-o", "PMRQMC_mpi.bin", "PMRQMC_mpi.cpp"], directory,
        f"compile_fixed_{gamma:g}.log")
    command = mpi(oversubscribe, 1) + ["./PMRQMC_mpi.bin"]
    run(command, directory, f"fixed_{gamma:g}.log")
    text = (directory / f"fixed_{gamma:g}.log").read_text()
    match = re.search(r"mean\(q\) = ([^\s]+)", text)
    if not match:
        raise RuntimeError("fixed run did not report mean(q)")
    return float(match.group(1)), text


def test_pure_beta(directory, oversubscribe):
    points = PURE_BETA
    qcpt_values = run_path(directory, points, True, "pure_beta_qcpt", len(points), oversubscribe)
    pt_values = run_path(directory, points, False, "pure_beta_pt", len(points), oversubscribe)
    rows = []
    for slot, (beta, gamma) in enumerate(points):
        exact = exact_values(FIXED_MIXED, GAMMA_MIXED, beta, gamma)
        for name in ("H", "B"):
            key = (slot, name)
            qcpt_name = "E" if name == "H" else "Xavg"
            expected = exact[qcpt_name]
            q = qcpt_values[key]["mean"]
            p = pt_values[key]["mean"]
            assert_close(q, expected, max(0.10, 6 * qcpt_values[key]["stdev"]), f"pure-beta QCPT {slot} {name}")
            assert_close(p, expected, max(0.10, 6 * pt_values[key]["stdev"]), f"pure-beta PT {slot} {name}")
            assert_close(q, p, max(0.10, 6 * qcpt_values[key]["stdev"] + 6 * pt_values[key]["stdev"]), f"pure-beta equivalence {slot} {name}")
            rows.append({"slot": slot, "beta": beta, "name": name, "exact": expected,
                         "qcpt": q, "pt": p})
    return rows


def test_pure_gamma(directory, oversubscribe):
    qcpt_values = run_path(directory, PURE_GAMMA, True, "pure_gamma_qcpt", len(PURE_GAMMA), oversubscribe)
    rows = []
    for slot, (beta, gamma) in enumerate(PURE_GAMMA):
        qmc_q, _ = run_fixed(directory, beta, gamma, FIXED_TF, GAMMA_TF, oversubscribe)
        exact = exact_values(FIXED_TF, GAMMA_TF, beta, gamma)
        qcpt_energy = qcpt_values[(slot, "H")]
        assert_close(qcpt_energy["mean"], exact["E"],
                     max(0.10, 6 * qcpt_energy["stdev"]), f"pure-gamma QCPT energy gamma={gamma}")
        rows.append({"beta": beta, "gamma": gamma, "exact_q": exact["q"], "qmc_q": qmc_q,
                     "exact_energy": exact["E"], "qcpt_energy": qcpt_energy["mean"]})
        assert_close(qmc_q, exact["q"], 0.50, f"pure-gamma expansion order gamma={gamma}")
    return rows


def test_layouts(directory, oversubscribe):
    results = []
    for points in (((0.4, 0.7), (1.2, 0.3)),
                   ((0.3, 0.9), (0.7, 0.6), (1.5, 0.2))):
        prefix = "layout_" + str(len(points))
        values = run_path(directory, points, True, prefix, len(points) * 2, oversubscribe)
        swaps = list(csv.DictReader((directory / (prefix + "_swaps.csv")).open(newline="")))
        flow = list(csv.DictReader((directory / (prefix + "_flow.csv")).open(newline="")))
        if len(swaps) != len(points) - 1 or any(int(row["attempts"]) <= 0 for row in swaps):
            raise RuntimeError(f"{prefix}: incomplete swap coverage")
        if len(flow) != len(points) * len(points) or not values:
            raise RuntimeError(f"{prefix}: incomplete layout output")
        results.append({"points": len(points), "ranks": len(points) * 2,
                        "swap_attempts": [int(row["attempts"]) for row in swaps],
                        "flow_rows": len(flow)})
    return results


def test_preparation_edge_cases(directory):
    cancellation = directory / "cancellation"
    stage_sources(cancellation)
    prepare_split(cancellation,
                  ((1.0, (0,), "Z"), (0.4, (0, 1), "ZZ"), (-0.2, (0,), "X")),
                  ((-1.0, (0,), "Z"), (-0.4, (0, 1), "ZZ"), (-0.2, (1,), "X")),
                  with_observables=False)
    run(["g++", "-O1", "-std=c++11", "-o", "split_component_probe", "split_component_probe.cpp"], cancellation)
    probe = [float(value) for value in subprocess.check_output(["./split_component_probe"], cwd=cancellation, text=True).split()]
    if probe != [1.0, -1.0, 0.4, -0.4]:
        raise RuntimeError(f"component cancellation probe mismatch: {probe}")

    repeated = directory / "repeated"
    stage_sources(repeated)
    prepare_split(repeated, ((0.4, (0, 1), "ZZ"),), ((-0.3, (0,), "X"),), with_observables=False)
    compile_binaries(repeated, beta=0.8, gamma=0.5, tsteps=300, steps=4000, nbins=10)
    repeated_points = ((0.8, 0.0), (0.8, 0.5))
    values = run_path(repeated, repeated_points, True, "repeated", 2, True)
    if not values or any(not math.isfinite(row["mean"]) for row in values.values()):
        raise RuntimeError("repeated-energy QCPT smoke produced non-finite output")
    for slot, (beta, gamma) in enumerate(repeated_points):
        expected = exact_values(((0.4, (0, 1), "ZZ"),), ((-0.3, (0,), "X"),), beta, gamma)["E"]
        row = values[(slot, "H")]
        assert_close(row["mean"], expected, max(0.20, 6 * row["stdev"]),
                     f"repeated-energy exact comparison gamma={gamma}")

    return {"cancellation": probe, "repeated_energy": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--oversubscribe", action="store_true")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    mixed = root / "mixed"
    stage_sources(mixed)
    prepare_split(mixed, FIXED_MIXED, GAMMA_MIXED)
    compile_binaries(mixed)
    report = {
        "pure_beta": test_pure_beta(mixed, args.oversubscribe),
        "layouts": test_layouts(mixed, args.oversubscribe),
        "preparation_edge_cases": test_preparation_edge_cases(root),
    }
    tf = root / "uniform_transverse_field"
    stage_sources(tf)
    prepare_split(tf, FIXED_TF, GAMMA_TF, with_observables=False)
    compile_binaries(tf, beta=0.9, gamma=0.0)
    report["pure_gamma"] = test_pure_gamma(tf, args.oversubscribe)
    immutability = root / "immutability"
    stage_sources(immutability)
    prepare_split(immutability, FIXED_MIXED, GAMMA_MIXED)
    write_parameters(immutability, 0.7, 0.75, 100, 1000, 10)
    run(["g++", "-O1", "-std=c++11", "-o", "target_weight_immutability", "target_weight_immutability.cpp"], immutability)
    run(["./target_weight_immutability"], immutability, "immutability.log")
    report["immutability"] = {"passed": True}
    (root / "edge_case_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
