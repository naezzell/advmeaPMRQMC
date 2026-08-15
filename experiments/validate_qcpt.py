"""End-to-end QCPT validation for the fully mixed two-spin fixture.

The campaign is intentionally self-contained: it stages the sources into an
artifact directory, builds both split and explicitly combined Hamiltonians,
runs fixed-parameter controls and QCPT, compares E/Z1/(X1+X2)/2 with direct
4x4 diagonalization, and records deterministic negative controls for the two
common incomplete target-weight implementations.
"""

import argparse
import csv
import json
import math
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path


PATH = ((0.25, 1.50), (0.70, 0.75), (1.60, 0.25), (3.50, 0.00))
FIXED = ((-0.90, (0, 1), "ZZ"), (0.20, (0,), "Z"),
         (-0.35, (0,), "X"), (-0.10, (0, 1), "XX"))
GAMMA = ((0.25, (1,), "Z"), (-0.15, (0, 1), "ZZ"),
         (-0.55, (1,), "X"), (-0.20, (0, 1), "XX"))
ABSOLUTE_TOLERANCE = 0.025
SIGMA_LIMIT = 6.0


def matmul(left, right):
    return [[sum(left[i][k] * right[k][j] for k in range(len(right)))
             for j in range(len(right[0]))] for i in range(len(left))]


def jacobi_eigh(matrix, tolerance=1e-14):
    n = len(matrix)
    a = [list(row) for row in matrix]
    vectors = [[float(i == j) for j in range(n)] for i in range(n)]
    for _ in range(100 * n * n):
        p, q = max(((i, j) for i in range(n) for j in range(i + 1, n)),
                   key=lambda pair: abs(a[pair[0]][pair[1]]))
        if abs(a[p][q]) < tolerance:
            return [a[i][i] for i in range(n)], vectors
        angle = 0.5 * math.atan2(2.0 * a[p][q], a[q][q] - a[p][p])
        c, s = math.cos(angle), math.sin(angle)
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        for i in range(n):
            if i in (p, q):
                continue
            aip, aiq = a[i][p], a[i][q]
            a[i][p] = a[p][i] = c * aip - s * aiq
            a[i][q] = a[q][i] = s * aip + c * aiq
        a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq
        a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq
        a[p][q] = a[q][p] = 0.0
        for i in range(n):
            vip, viq = vectors[i][p], vectors[i][q]
            vectors[i][p] = c * vip - s * viq
            vectors[i][q] = s * vip + c * viq
    raise RuntimeError("Jacobi diagonalization did not converge")


def pauli_matrix(pauli):
    if pauli == "I":
        return ((1.0, 0.0), (0.0, 1.0))
    if pauli == "X":
        return ((0.0, 1.0), (1.0, 0.0))
    if pauli == "Z":
        return ((1.0, 0.0), (0.0, -1.0))
    raise ValueError(pauli)


def kron(a, b):
    return tuple(tuple(a[i][j] * b[k][l] for j in range(len(a[0]))
                       for l in range(len(b[0])))
                 for i in range(len(a)) for k in range(len(b)))


def operator(n, sites, paulis):
    result = ((1.0, 0.0), (0.0, 1.0))
    site_map = dict(zip(sites, paulis))
    for spin in range(n):
        result = kron(result, pauli_matrix(site_map.get(spin, "I")))
    return result


def add_scaled(target, source, scale):
    for i in range(len(target)):
        for j in range(len(target)):
            target[i][j] += scale * source[i][j]


def exact_point(beta, gamma):
    h = [[0.0] * 4 for _ in range(4)]
    for coefficient, sites, paulis in FIXED:
        add_scaled(h, operator(2, sites, paulis), coefficient)
    for coefficient, sites, paulis in GAMMA:
        add_scaled(h, operator(2, sites, paulis), gamma * coefficient)
    z1 = operator(2, (0,), "Z")
    xavg = [[0.0] * 4 for _ in range(4)]
    add_scaled(xavg, operator(2, (0,), "X"), 0.5)
    add_scaled(xavg, operator(2, (1,), "X"), 0.5)
    energies, vectors = jacobi_eigh(h)
    weights = [math.exp(-beta * (e - min(energies))) for e in energies]
    values = {}
    for name, observable in (("E", h), ("Z1", z1), ("Xavg", xavg)):
        diagonal = []
        for col in range(4):
            diagonal.append(sum(vectors[i][col] * observable[i][j] * vectors[j][col]
                                for i in range(4) for j in range(4)))
        values[name] = sum(w * v for w, v in zip(weights, diagonal)) / sum(weights)
    return values


def term_text(term):
    coefficient, sites, paulis = term
    tokens = [f"{coefficient:.17g}"]
    for site, pauli in zip(sites, paulis):
        tokens += [str(site + 1), pauli]
    return " ".join(tokens)


def write_fixture(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "H_fixed.txt").write_text("\n".join(term_text(t) for t in FIXED) + "\n")
    (directory / "H_gamma.txt").write_text("\n".join(term_text(t) for t in GAMMA) + "\n")
    (directory / "z1.txt").write_text("1 1 Z\n")
    (directory / "xavg.txt").write_text("0.5 1 X\n0.5 2 X\n")
    # The PMR custom-observable interface reserves the first five slots for A
    # and the sixth for B.
    (directory / "qcpt_schedule.txt").write_text(
        "# beta gamma\n" + "".join(f"{b:.17g} {g:.17g}\n" for b, g in PATH))
    (directory / "H_combined.txt").write_text(
        "\n".join((term_text(t) for t in (
            (-1.0125, (0, 1), "ZZ"), (0.2, (0,), "Z"),
            (0.1875, (1,), "Z"), (-0.35, (0,), "X"),
            (-0.4125, (1,), "X"), (-0.25, (0, 1), "XX")))) + "\n")


def write_parameters(path, beta, gamma, tsteps, steps, interval, nbins, absolute):
    path.write_text(f"""
#define Tsteps {tsteps}
#define steps {steps}
#define stepsPerMeasurement {interval}
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
#define RNG_SEED_OFFSET 7300
#define MEASURE_H
""" + ("#define ABS_WEIGHTS\n" if absolute else ""))


def copy_sources(stage, source_root):
    stage.mkdir(parents=True, exist_ok=True)
    for name in ("prepare.cpp", "mainqmc.hpp", "divdiff.hpp", "pt_schedule.hpp", "beta_anneal.hpp",
                 "PMRQMC_mpi.cpp", "PMRQMC_pt_mpi.cpp", "PMRQMC_qcpt_mpi.cpp"):
        shutil.copy2(source_root / name, stage / name)


def prepare_split(stage):
    subprocess.run(["g++", "-O1", "-std=c++11", "-o", "prepare.bin", "prepare.cpp"],
                   cwd=stage, check=True, stdout=subprocess.DEVNULL)
    obs = ["z1.txt"] * 5 + ["xavg.txt"]
    subprocess.run(["./prepare.bin", "--hamiltonian-fixed", "H_fixed.txt",
                    "--hamiltonian-gamma", "H_gamma.txt", *obs], cwd=stage,
                   check=True, stdout=subprocess.DEVNULL)


def prepare_combined(stage):
    subprocess.run(["g++", "-O1", "-std=c++11", "-o", "prepare.bin", "prepare.cpp"],
                   cwd=stage, check=True, stdout=subprocess.DEVNULL)
    obs = ["z1.txt"] * 5 + ["xavg.txt"]
    subprocess.run(["./prepare.bin", "H_combined.txt", *obs], cwd=stage,
                   check=True, stdout=subprocess.DEVNULL)


def mpi_command(oversubscribe, ranks):
    command = ["mpirun"]
    if oversubscribe:
        command.append("--oversubscribe")
    return command + ["-n", str(ranks)]


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def ratio(rows, observable):
    numerator = sum(float(row[f"signed_{observable}"]) for row in rows)
    denominator = sum(float(row["sign"]) for row in rows)
    return numerator / denominator if denominator else float("nan")


def ratio_stats(rows, observable, blocks=50):
    if not rows:
        return {"value": float("nan"), "stdev": float("nan"), "measurements": 0}
    value = ratio(rows, observable)
    block_size = max(1, len(rows) // blocks)
    block_values = []
    for start in range(0, len(rows), block_size):
        block = rows[start:start + block_size]
        if len(block) < block_size:
            continue
        block_value = ratio(block, observable)
        if math.isfinite(block_value):
            block_values.append(block_value)
    if len(block_values) > 1:
        mean = sum(block_values) / len(block_values)
        stdev = math.sqrt(sum((item - mean) ** 2 for item in block_values) /
                           (len(block_values) * (len(block_values) - 1)))
    else:
        stdev = float("nan")
    return {"value": value, "stdev": stdev, "measurements": len(rows)}


def final_observables(path):
    result = {}
    for row in read_rows(path):
        if row["kind"] == "observable" and row["name"] in ("A", "B", "H"):
            result[(float(row.get("beta", row.get("slot"))), row["name"])] = (
                float(row["mean"]), float(row["stdev"]))
    return result


def run_fixed(source_root, output, beta, gamma, args, absolute=False, combined=False):
    stage = output / (f"fixed_{beta:g}_{gamma:g}" + ("_combined" if combined else ""))
    copy_sources(stage, source_root)
    write_fixture(stage)
    write_parameters(stage / "parameters.hpp", beta, gamma, args.Tsteps, args.steps,
                     args.interval, args.nbins, absolute)
    if combined:
        prepare_combined(stage)
    else:
        prepare_split(stage)
    subprocess.run(["mpicxx", "-O1", "-std=c++11", "-o", "PMRQMC_mpi.bin", "PMRQMC_mpi.cpp"],
                   cwd=stage, check=True, stdout=subprocess.DEVNULL)
    command = mpi_command(args.oversubscribe, 1) + ["./PMRQMC_mpi.bin",
               "--timeseries-prefix", "trace.csv"]
    if args.beta_anneal:
        command += ["--beta-anneal", "--anneal-interval", str(args.anneal_interval)]
    subprocess.run(command, cwd=stage, check=True,
                   stdout=(stage / "run.log").open("w"), stderr=subprocess.STDOUT)
    rows = read_rows(stage / "trace.csv")
    return stage, {name: ratio_stats(rows, field) for name, field in
                   (("Z1", "obs_0"), ("Xavg", "obs_5"), ("E", "obs_6"))}


def run_qcpt(source_root, output, args, absolute=False):
    stage = output / ("qcpt_abs" if absolute else "qcpt_signed")
    copy_sources(stage, source_root)
    write_fixture(stage)
    write_parameters(stage / "parameters.hpp", PATH[0][0], PATH[0][1], args.Tsteps,
                     args.steps, args.interval, args.nbins, absolute)
    prepare_split(stage)
    subprocess.run(["mpicxx", "-O1", "-std=c++11", "-DPMR_QCPT", "-o",
                    "PMRQMC_qcpt_mpi.bin", "PMRQMC_qcpt_mpi.cpp"], cwd=stage,
                   check=True, stdout=subprocess.DEVNULL)
    command = mpi_command(args.oversubscribe, len(PATH)) + ["./PMRQMC_qcpt_mpi.bin",
               "--schedule", "qcpt_schedule.txt", "--updates-per-exchange",
               str(args.updates_per_exchange), "--output-prefix", "qcpt",
               "--timeseries-prefix", "trace.csv"]
    if args.beta_anneal:
        command += ["--beta-anneal", "--anneal-interval", str(args.anneal_interval)]
    subprocess.run(command, cwd=stage, check=True,
                   stdout=(stage / "run.log").open("w"), stderr=subprocess.STDOUT)
    rows = read_rows(stage / "trace.csv")
    estimates = {}
    for slot, (beta, gamma) in enumerate(PATH):
        slot_rows = [row for row in rows if int(row["slot"]) == slot]
        estimates[(beta, gamma)] = {name: ratio_stats(slot_rows, field) for name, field in
                                    (("Z1", "obs_0"), ("Xavg", "obs_5"), ("E", "obs_6"))}
    return stage, estimates, final_observables(stage / "qcpt_observables.csv")


def complete_homogeneous(nodes, degree):
    coefficients = [1.0] + [0.0] * degree
    for node in nodes:
        updated = [0.0] * (degree + 1)
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                updated[i + j] += coefficients[i] * node ** j
        coefficients = updated
    return coefficients[degree]


def divided_difference_exp(nodes):
    # Complete homogeneous polynomial series for exp[x_0,...,x_q].
    q = len(nodes) - 1
    return sum(complete_homogeneous(nodes, m - q) / math.factorial(m)
               for m in range(q, q + 80))


def pmr_weight(beta, gamma, base_state, operators, freeze_diagonal=False,
               freeze_offdiagonal=False):
    state = base_state
    energies = []
    products = []
    for operator_index in range(len(operators) + 1):
        energies.append(exact_path_energy(state, gamma if not freeze_diagonal else 0.25))
        if operator_index < len(operators):
            op = operators[operator_index]
            state ^= 1 << op
            products.append(exact_path_matrix_element(op, state,
                                                      gamma if not freeze_offdiagonal else 0.25))
    dd = divided_difference_exp([-beta * value for value in energies])
    factor = (-beta) ** len(operators) / math.factorial(len(operators))
    return dd * factor * math.prod(products)


def exact_path_energy(state, gamma):
    value = 0.0
    for coefficient, sites, paulis in FIXED:
        sign = 1.0
        for site in sites:
            if "Z" in paulis and (state & (1 << site)):
                sign *= -1
        value -= sign * coefficient
    for coefficient, sites, paulis in GAMMA:
        sign = 1.0
        for site in sites:
            if "Z" in paulis and (state & (1 << site)):
                sign *= -1
        value -= gamma * sign * coefficient
    return value


def exact_path_matrix_element(operator_index, state, gamma):
    value = 0.0
    for coefficient, sites, paulis in FIXED + tuple((c, s, p) for c, s, p in GAMMA):
        if "X" not in paulis:
            continue
        if sum(1 << site for site in sites) != (1 << operator_index):
            continue
        coefficient *= 1.0 if (coefficient, sites, paulis) in FIXED else gamma
        value -= coefficient
    return value


def negative_controls():
    # q=0 detects diagonal freezing; q=2 with X1 repeated detects local
    # off-diagonal-product reuse.  The target point is deliberately different
    # from the local point used by the incomplete evaluators.
    target = (0.70, 0.75)
    local = (0.70, 0.25)
    q0_full = pmr_weight(target[0], target[1], 0, ())
    q0_frozen = pmr_weight(target[0], local[1], 0, (), freeze_diagonal=True)
    # Spin 2 is present only in H_gamma's X term in this fixture.  Repeating
    # it makes the target-dependent local product observable.
    q2_full = pmr_weight(target[0], target[1], 0, (1, 1))
    q2_local_product = pmr_weight(target[0], target[1], 0, (1, 1), freeze_offdiagonal=True)
    return {"q0_diagonal_freeze_disagreement": abs(q0_full - q0_frozen),
            "q2_offdiagonal_reuse_disagreement": abs(q2_full - q2_local_product),
            "passed": abs(q0_full - q0_frozen) > 1e-8 and abs(q2_full - q2_local_product) > 1e-8}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--Tsteps", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--nbins", type=int, default=50)
    parser.add_argument("--updates-per-exchange", type=int, default=10)
    parser.add_argument("--beta-anneal", action="store_true")
    parser.add_argument("--anneal-interval", type=int, default=10)
    parser.add_argument("--oversubscribe", action="store_true")
    parser.add_argument("--absolute-weights", action="store_true")
    args = parser.parse_args()
    if args.steps < args.interval * args.nbins:
        parser.error("steps must be at least interval * nbins")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1]
    controls = negative_controls()
    fixed = {}
    for beta, gamma in PATH:
        _, fixed[(beta, gamma)] = run_fixed(source_root, output, beta, gamma, args)
    _, combined = run_fixed(source_root, output, PATH[1][0], PATH[1][1], args, combined=True)
    _, qcpt, summaries = run_qcpt(source_root, output, args, args.absolute_weights)
    exact = {(beta, gamma): exact_point(beta, gamma) for beta, gamma in PATH}
    comparisons = []
    summary_names = {"E": "H", "Z1": "A", "Xavg": "B"}
    for point in PATH:
        for name, exact_name in (("E", "E"), ("Z1", "Z1"), ("Xavg", "Xavg")):
            exact_value = exact[point][exact_name]
            qcpt_stats = qcpt[point][name]
            fixed_stats = fixed[point][name]
            summary = summaries.get((point[0], summary_names[name]), (float("nan"), float("nan")))
            qcpt_stdev = summary[1] if math.isfinite(summary[1]) else qcpt_stats["stdev"]
            fixed_stdev = fixed_stats["stdev"]
            qcpt_tolerance = max(ABSOLUTE_TOLERANCE, SIGMA_LIMIT * qcpt_stdev) if math.isfinite(qcpt_stdev) else ABSOLUTE_TOLERANCE
            fixed_tolerance = max(ABSOLUTE_TOLERANCE, SIGMA_LIMIT * fixed_stdev) if math.isfinite(fixed_stdev) else ABSOLUTE_TOLERANCE
            comparisons.extend((
                {"point": point, "observable": name, "source": "fixed", "exact": exact_value,
                 "estimate": fixed_stats["value"], "stdev": fixed_stdev,
                 "absolute_error": abs(fixed_stats["value"] - exact_value),
                 "tolerance": fixed_tolerance,
                 "passed": abs(fixed_stats["value"] - exact_value) <= fixed_tolerance},
                {"point": point, "observable": name, "source": "qcpt", "exact": exact_value,
                 "estimate": qcpt_stats["value"], "stdev": qcpt_stdev,
                 "absolute_error": abs(qcpt_stats["value"] - exact_value),
                 "tolerance": qcpt_tolerance,
                 "passed": abs(qcpt_stats["value"] - exact_value) <= qcpt_tolerance}))
    split = fixed[PATH[1]]
    combined_errors = {name: abs(split[name]["value"] - combined[name]["value"]) for name in split}
    report = {"path": PATH, "exact": {str(k): v for k, v in exact.items()},
              "fixed": {str(k): v for k, v in fixed.items()}, "qcpt": {str(k): v for k, v in qcpt.items()},
              "qcpt_summary": {str(k): v for k, v in summaries.items()},
              "split_vs_combined_at_0.70_0.75": combined_errors,
              "negative_controls": controls, "comparisons": comparisons,
              "passed": (controls["passed"] and
                         all(error <= ABSOLUTE_TOLERANCE for error in combined_errors.values()) and
                         all(item["passed"] for item in comparisons))}
    (output / "qcpt_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
