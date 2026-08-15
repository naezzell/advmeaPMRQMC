"""Generate the paper-faithful random 3-regular TFIM ensemble.

The instance Hamiltonian is

    H(Gamma) = sum_(i,j in E) Z_i Z_j - Gamma sum_i X_i,

where E is a simple 3-regular graph.  Generated instances are split into
H_fixed and H_gamma so that Gamma can be selected at simulation time.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


REFERENCE_GAMMAS = (0.1, 0.4)
REFERENCE_GAMMA_NOTE = (
    "Reference transverse-field strengths used in the papers; H_fixed and "
    "H_gamma define the model for arbitrary Gamma."
)


def random_3_regular_graph(n: int, seed: int, max_attempts: int = 10000) -> List[Tuple[int, int]]:
    """Return a deterministic simple 3-regular graph using rejection sampling."""
    if n < 4 or n % 2:
        raise ValueError("a simple 3-regular graph requires even n >= 4")
    rng = random.Random(seed)
    for _ in range(max_attempts):
        stubs = [vertex for vertex in range(n) for _ in range(3)]
        rng.shuffle(stubs)
        edges = set()
        valid = True
        for left, right in zip(stubs[::2], stubs[1::2]):
            edge = tuple(sorted((left, right)))
            if edge[0] == edge[1] or edge in edges:
                valid = False
                break
            edges.add(edge)
        if not valid or len(edges) != 3 * n // 2:
            continue
        degree = [0] * n
        for left, right in edges:
            degree[left] += 1
            degree[right] += 1
        if degree == [3] * n:
            return sorted(edges)
    raise RuntimeError(f"could not generate a simple 3-regular graph for n={n}, seed={seed}")


def validate_edges(n: int, edges: Sequence[Sequence[int]]) -> None:
    """Validate the canonical simple 3-regular graph representation."""
    canonical = [tuple(edge) for edge in edges]
    if len(canonical) != 3 * n // 2:
        raise ValueError("incorrect edge count")
    if canonical != sorted(canonical):
        raise ValueError("edges must be lexicographically sorted")
    if any(len(edge) != 2 or edge[0] >= edge[1] for edge in canonical):
        raise ValueError("edges must contain distinct increasing endpoints")
    if len(set(canonical)) != len(canonical):
        raise ValueError("duplicate edge")
    if any(vertex < 0 or vertex >= n for edge in canonical for vertex in edge):
        raise ValueError("edge endpoint out of range")
    degree = [0] * n
    for left, right in canonical:
        degree[left] += 1
        degree[right] += 1
    if degree != [3] * n:
        raise ValueError(f"invalid degree sequence: {degree}")


def _pmr_term(coefficient: float, sites: Iterable[int], operators: Iterable[str]) -> str:
    tokens = [f"{coefficient:.17g}"]
    for site, operator in zip(sites, operators):
        tokens.extend((str(site + 1), operator))
    return " ".join(tokens)


def hamiltonian_terms(n: int, edges: Sequence[Sequence[int]]) -> Tuple[List[str], List[str]]:
    """Return PMR lines for H_fixed and H_gamma."""
    validate_edges(n, edges)
    fixed = [_pmr_term(1.0, edge, ("Z", "Z")) for edge in edges]
    gamma = [_pmr_term(-1.0, (vertex,), ("X",)) for vertex in range(n)]
    return fixed, gamma


def instance_metadata(n: int, seed: int, edges: Sequence[Sequence[int]]) -> dict:
    validate_edges(n, edges)
    return {
        "schema_version": 1,
        "model": "paper_tfim_3regular",
        "n": n,
        "seed": seed,
        "degree": 3,
        "coupling": 1.0,
        "edges": [list(edge) for edge in edges],
        "validation": {
            "edge_count": len(edges),
            "degree_sequence": [3] * n,
            "simple": True,
        },
        "reference_parameters": {
            "gamma_values": list(REFERENCE_GAMMAS),
            "gamma_values_note": REFERENCE_GAMMA_NOTE,
        },
    }


def write_instance(output_dir: Path | str, n: int, seed: int) -> Path:
    """Generate one instance directory and return its path."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    edges = random_3_regular_graph(n, seed)
    fixed, gamma = hamiltonian_terms(n, edges)
    (output / "H_fixed.txt").write_text("\n".join(fixed) + "\n")
    (output / "H_gamma.txt").write_text("\n".join(gamma) + "\n")
    (output / "instance.json").write_text(
        json.dumps(instance_metadata(n, seed, edges), indent=2) + "\n"
    )
    return output


def generate_catalog(catalog: Path | str, output_root: Path | str) -> List[Path]:
    """Generate all entries in a catalog JSON file."""
    catalog_data = json.loads(Path(catalog).read_text())
    written = []
    for entry in catalog_data["instances"]:
        target = Path(output_root) / f"N{entry['n']:03d}" / entry["id"]
        written.append(write_instance(target, entry["n"], entry["seed"]))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    one = subparsers.add_parser("one", help="generate one instance")
    one.add_argument("output_dir", type=Path)
    one.add_argument("--n", type=int, required=True)
    one.add_argument("--seed", type=int, required=True)

    many = subparsers.add_parser("catalog", help="generate a catalog locally")
    many.add_argument("catalog", type=Path)
    many.add_argument("output_root", type=Path)

    args = parser.parse_args()
    if args.command == "one":
        write_instance(args.output_dir, args.n, args.seed)
    else:
        for path in generate_catalog(args.catalog, args.output_root):
            print(path)


if __name__ == "__main__":
    main()
