"""Deterministic instance and schedule presets for the 2019 PMR benchmark.

This keeps instance generation explicit: the seed, graph, clauses, and PMR
Hamiltonian are saved beside every run so benchmark data is auditable.
"""

import json
import random
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
sys.path.append(str(Path(__file__).resolve().parent))
from pauli_manipulations import PauliH, PauliTerm
from pt_driver import generate_tempering_schedule
from paper_tfim_3regular import (
    random_3_regular_graph as canonical_random_3_regular_graph,
    write_instance as write_paper_tfim_instance,
)


# The eleven-point beta ladder used by this preset.  tau defaults to beta/2.
PUBLISHED_BETA_LADDER = (0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0)
GAMMA_PRESETS = (0.1, 0.4)


def random_3_regular_graph(n, seed):
    return canonical_random_3_regular_graph(n, seed)


def make_random_3regular_tfim(n, gamma, seed, output_dir):
    """Generate the split paper TFIM; gamma is retained only for API compatibility."""
    del gamma
    return write_paper_tfim_instance(output_dir, n, seed)


def random_3regular_max2sat(n, seed):
    """Return clauses with three occurrences of every variable."""
    if n < 4 or n % 2:
        raise ValueError("a 3-regular MAX2SAT instance requires an even n >= 4")
    rng = random.Random(seed)
    for _ in range(1000):
        stubs = [vertex for vertex in range(n) for _ in range(3)]
        rng.shuffle(stubs)
        clauses = []
        if any(left == right for left, right in zip(stubs[::2], stubs[1::2])):
            continue
        for left, right in zip(stubs[::2], stubs[1::2]):
            clauses.append((left, rng.choice((-1, 1)), right, rng.choice((-1, 1))))
        return clauses
    raise RuntimeError("could not generate a 3-regular MAX2SAT instance")


def make_random_3regular_max2sat(n, seed, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clauses = random_3regular_max2sat(n, seed)
    terms = []
    for left, sign_left, right, sign_right in clauses:
        # Unsatisfied-literal projectors: (1-s_i Z_i)(1-s_j Z_j)/4.
        terms.append(PauliTerm(0.25, [], [], n))
        terms.append(PauliTerm(-0.25 * sign_left, [left + 1], ["Z"], n))
        terms.append(PauliTerm(-0.25 * sign_right, [right + 1], ["Z"], n))
        terms.append(PauliTerm(0.25 * sign_left * sign_right,
                               [left + 1, right + 1], ["Z", "Z"], n))
    hamiltonian = PauliH(n, terms)
    magnetization = PauliH(n, [PauliTerm(1.0 / n, [vertex + 1], ["X"], n) for vertex in range(n)])
    (output_dir / "H.txt").write_text(hamiltonian.to_pmr_str())
    (output_dir / "transverse_magnetization.txt").write_text(magnetization.to_pmr_str())
    (output_dir / "instance.json").write_text(json.dumps({
        "model": "random_3_regular_max2sat", "n": n, "seed": seed, "clauses": clauses,
    }, indent=2) + "\n")
    return output_dir / "H.txt"


def make_benchmark_preset(output_dir, n=20, seed=2019, gamma=0.1, model="tfim"):
    if gamma not in GAMMA_PRESETS:
        raise ValueError(f"gamma must be one of {GAMMA_PRESETS}")
    output_dir = Path(output_dir)
    if model == "tfim":
        make_random_3regular_tfim(n, gamma, seed, output_dir)
    elif model == "max2sat":
        make_random_3regular_max2sat(n, seed, output_dir)
    else:
        raise ValueError("model must be 'tfim' or 'max2sat'")
    generate_tempering_schedule(output_dir / "tempering_schedule.txt", PUBLISHED_BETA_LADDER)
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--gamma", type=float, choices=GAMMA_PRESETS, default=0.1)
    parser.add_argument("--model", choices=("tfim", "max2sat"), default="tfim")
    args = parser.parse_args()
    make_benchmark_preset(args.output_dir, args.n, args.seed, args.gamma, args.model)
