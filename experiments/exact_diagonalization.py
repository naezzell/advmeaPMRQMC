"""Exact thermal expectations for a generated PMR benchmark instance.

This is intended for small systems only. Dense diagonalization is practical
for the n=10 benchmark and becomes expensive quickly as n grows.
"""

import argparse
import json
from pathlib import Path


def read_pauli_terms(path):
    terms = []
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        coefficient = float(tokens[0])
        if (len(tokens) - 1) % 2:
            raise ValueError(f"invalid Pauli term in {path}: {raw_line}")
        operators = []
        for index in range(1, len(tokens), 2):
            operators.append((int(tokens[index]) - 1, tokens[index + 1]))
        terms.append((coefficient, operators))
    return terms


def dense_pauli_matrix(n, terms, numpy):
    dimension = 1 << n
    matrix = numpy.zeros((dimension, dimension), dtype=numpy.complex128)
    for coefficient, operators in terms:
        for state in range(dimension):
            target = state
            phase = 1.0 + 0.0j
            for site, operator in operators:
                bit = (state >> site) & 1
                if operator == "X":
                    target ^= 1 << site
                elif operator == "Y":
                    phase *= 1j if bit == 0 else -1j
                    target ^= 1 << site
                elif operator == "Z":
                    phase *= 1.0 if bit == 0 else -1.0
                else:
                    raise ValueError(f"unsupported Pauli operator: {operator}")
            matrix[target, state] += coefficient * phase
    return matrix


def exact_expectation(hamiltonian_path, observable_path, n, beta):
    try:
        import numpy
    except ImportError as error:
        raise SystemExit("numpy is required: python3 -m pip install numpy") from error

    hamiltonian = dense_pauli_matrix(n, read_pauli_terms(hamiltonian_path), numpy)
    observable = dense_pauli_matrix(n, read_pauli_terms(observable_path), numpy)
    energies, eigenvectors = numpy.linalg.eigh(hamiltonian)
    shifted_weights = numpy.exp(-beta * (energies - energies.min()))
    observable_in_eigenbasis = numpy.einsum(
        "ij,ij->j", eigenvectors.conj(), observable @ eigenvectors
    ).real
    expectation = float(numpy.dot(shifted_weights, observable_in_eigenbasis) /
                       numpy.sum(shifted_weights))
    return {
        "beta": float(beta),
        "exact_mean": expectation,
        "dimension": 1 << n,
        "ground_energy": float(energies[0]),
        "energy_max": float(energies[-1]),
    }


def compute_benchmark_reference(benchmark_directory):
    root = Path(benchmark_directory)
    metadata = json.loads((root / "instance" / "instance.json").read_text())
    report = json.loads((root / "comparison_report.json").read_text())
    n = int(metadata["n"])
    if n > 12:
        raise ValueError("dense exact diagonalization is restricted to n <= 12")
    betas = [float(beta) for beta in report["betas"]]
    values = [exact_expectation(root / "instance" / "H.txt",
                                root / "instance" / "transverse_magnetization.txt",
                                n, beta) for beta in betas]
    result = {
        "model": metadata["model"],
        "n": n,
        "observable": "transverse_magnetization",
        "values": values,
        "cold_beta": values[-1]["beta"],
        "cold_exact_mean": values[-1]["exact_mean"],
    }
    (root / "exact_reference.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_directory")
    args = parser.parse_args()
    result = compute_benchmark_reference(args.benchmark_directory)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
