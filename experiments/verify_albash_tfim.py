"""Verify the N=12 regular TFIM against exact thermal diagonalization.

The Albash-style path uses the paper fixture ``fixture_12`` and
Gamma(beta) = (10 beta)^(-1/2).  Each beta is run independently because the
field changes with beta.  The output is a JSON report containing energy and
specific heat per site for exact diagonalization and PMR-QMC.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

try:
    from experiments.exact_diagonalization import dense_pauli_matrix, read_pauli_terms
    from experiments.paper_tfim_3regular import write_instance
except ModuleNotFoundError:  # direct ``python experiments/verify_...py`` invocation
    from exact_diagonalization import dense_pauli_matrix, read_pauli_terms
    from paper_tfim_3regular import write_instance


DEFAULT_BETAS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
ROOT = Path(__file__).parents[1]


def gamma_for_beta(beta: float) -> float:
    if beta <= 0:
        raise ValueError("beta must be positive")
    return (10.0 * beta) ** -0.5


def write_combined_hamiltonian(fixed: Path, transverse: Path, output: Path,
                               gamma: float) -> None:
    lines = [line for line in fixed.read_text().splitlines() if line.strip()]
    for line in transverse.read_text().splitlines():
        tokens = line.split()
        if tokens and not tokens[0].startswith("#"):
            tokens[0] = format(gamma * float(tokens[0]), ".17g")
            lines.append(" ".join(tokens))
    output.write_text("\n".join(lines) + "\n")


def exact_values(hamiltonian: Path, n: int, beta: float) -> dict:
    """Exact thermal values, using the global spin-flip parity blocks."""
    terms = read_pauli_terms(hamiltonian)
    if n != 12:
        matrix = dense_pauli_matrix(n, terms, np)
        energies = np.linalg.eigvalsh(matrix).real
    else:
        # The regular TFIM commutes with X^(tensor n).  Pair each Z-basis
        # state with its global complement and build the +/- blocks directly.
        dimension = 1 << (n - 1)
        blocks = [np.zeros((dimension, dimension), dtype=np.float64) for _ in (1, -1)]
        all_bits = (1 << n) - 1
        for coefficient, operators in terms:
            if len(operators) == 2 and all(op == "Z" for _, op in operators):
                left, right = (site for site, _ in operators)
                for state in range(dimension):
                    sign = (1 if ((state >> left) ^ (state >> right)) & 1 == 0 else -1)
                    blocks[0][state, state] += coefficient * sign
                    blocks[1][state, state] += coefficient * sign
            elif len(operators) == 1 and operators[0][1] == "X":
                site = operators[0][0]
                for state in range(dimension):
                    flipped = state ^ (1 << site)
                    representative = min(flipped, all_bits ^ flipped)
                    complement_phase = 1 if flipped == representative else -1
                    for block_index, parity in enumerate((1, -1)):
                        blocks[block_index][representative, state] += coefficient * parity ** (complement_phase == -1)
            else:
                raise ValueError("the symmetry-block exact path expects ZZ and X terms")
        energies = np.concatenate([np.linalg.eigvalsh(block) for block in blocks])
    weights = np.exp(-beta * (energies - energies.min()))
    weights /= weights.sum()
    mean = float(np.dot(weights, energies))
    variance = float(np.dot(weights, (energies - mean) ** 2))
    return {"energy_per_site": mean / n, "specific_heat_per_site": beta * beta * variance / n,
            "ground_energy": float(energies[0])}


def qmc_values(log: Path, n: int) -> dict:
    text = log.read_text()
    blocks = re.findall(r"Observable #\d+: ([^\n]+)\nmean\(O\) = ([^\n]+)\n"
                       r"std\.dev\.\(O\) = ([^\n]+)", text)
    values = {name.strip(): (float(mean), float(error)) for name, mean, error in blocks}
    if "H" not in values or "H^2" not in values:
        raise RuntimeError(f"could not parse H/H^2 from {log}; found {sorted(values)}")
    derived = re.search(r"Derived observable: specific heat\nmean\(O\) = ([^\n]+)\n"
                        r"std\.dev\.\(O\) = ([^\n]+)", text)
    if derived is None:
        raise RuntimeError(f"could not parse specific-heat uncertainty from {log}")
    energy, energy_error = values["H"]
    h2, h2_error = values["H^2"]
    # The C++ derived observable uses beta^2 ( <H^2> - <H>^2 ).
    beta = float(re.search(r"Parameters: beta = ([^,]+)", text).group(1))
    return {"energy_per_site": energy / n, "energy_error_per_site": energy_error / n,
            "specific_heat_per_site": beta * beta * (h2 - energy * energy) / n,
            "specific_heat_error_per_site": float(derived.group(2)) / n}


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    instance = output / "instance"
    if not instance.exists():
        write_instance(instance, 12, args.seed)
    rows = []
    for index, beta in enumerate(args.betas):
        gamma = gamma_for_beta(beta)
        stage = output / f"beta_{index:02d}_{beta:g}"
        stage.mkdir(exist_ok=True)
        hamiltonian = stage / "H.txt"
        write_combined_hamiltonian(instance / "H_fixed.txt", instance / "H_gamma.txt",
                                   hamiltonian, gamma)
        parameters = f"""#define Tsteps {args.tsteps}
#define steps {args.steps}
#define stepsPerMeasurement {args.steps_per_measurement}
#define beta {beta:.17g}
#define tau {beta / 2:.17g}
#define gamma 1.0
#define parity_cond 0
#define qmax {args.qmax}
#define Nbins {args.nbins}
#define EXHAUSTIVE_CYCLE_SEARCH
#define GAPS_GEOMETRIC_PARAMETER 0.8
#define COMPOSITE_UPDATE_BREAK_PROBABILITY 0.9
#define EXACTLY_REPRODUCIBLE
#define RNG_SEED_OFFSET {args.seed + index}
#define MEASURE_H
#define MEASURE_H2
"""
        (stage / "parameters.hpp").write_text(parameters)
        for source in ("PMRQMC.cpp", "mainqmc.hpp", "beta_anneal.hpp", "divdiff.hpp",
                       "prepare.cpp"):
            shutil.copy2(ROOT / source, stage / source)
        subprocess.run(["g++", "-O3", "-std=c++11", "-o", "prepare.bin", "prepare.cpp"],
                       cwd=stage, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["./prepare.bin", "H.txt"], cwd=stage, check=True,
                       stdout=subprocess.DEVNULL)
        subprocess.run(["g++", "-O3", "-std=c++11", "-o", "PMRQMC.bin", "PMRQMC.cpp"],
                       cwd=stage, check=True, stdout=subprocess.DEVNULL)
        log = stage / "qmc.log"
        with log.open("w") as stream:
            subprocess.run(["./PMRQMC.bin"], cwd=stage, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        exact = exact_values(hamiltonian, 12, beta)
        qmc = qmc_values(log, 12)
        rows.append({"beta": beta, "gamma": gamma, "exact": exact, "qmc": qmc,
                     "energy_abs_error": abs(qmc["energy_per_site"] - exact["energy_per_site"]),
                     "specific_heat_abs_error": abs(qmc["specific_heat_per_site"] - exact["specific_heat_per_site"])})
    report = {"model": "paper_tfim_3regular", "n": 12, "seed": args.seed,
              "betas": list(args.betas), "gamma_rule": "(10 beta)^(-1/2)", "rows": rows}
    (output / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--betas", default=",".join(map(str, DEFAULT_BETAS)),
                        help="comma-separated sparse beta values")
    parser.add_argument("--seed", type=int, default=1012)
    parser.add_argument("--tsteps", type=int, default=20000)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--steps-per-measurement", type=int, default=20)
    parser.add_argument("--nbins", type=int, default=100)
    parser.add_argument("--qmax", type=int, default=1000)
    args = parser.parse_args()
    args.betas = tuple(float(value) for value in args.betas.split(",") if value.strip())
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
