"""Pinned ALPS ``dirloop_sse`` input generation for matched TFIM controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ALPS_TAG = "v2.3.4"
ALPS_COMMIT = "97914eba01fb8eae1b96d460b577cb62a8f7ba94"


def pauli_to_alps(coupling: float, gamma: float):
    """Map -J sigma_z sigma_z - Gamma sigma_x to ALPS spin-1/2 units.

    ALPS uses S=sigma/2 and its spin model parameters represent
    Jz*S_z*S_z - Gamma_ALPS*S_x.  Therefore Jz=-4J and
    Gamma_ALPS=2*Gamma preserve the Hamiltonian without an energy rescaling.
    """
    return {"Jxy": 0.0, "Jz": -4.0 * coupling, "Gamma": 2.0 * gamma}


def tfim_parameters(length: int, gamma: float, beta: float, periodic: bool,
                    thermalization: int, sweeps: int, seed: int, skip: int = 1):
    if length < 2 or beta <= 0 or thermalization < 0 or sweeps <= 0 or skip <= 0:
        raise ValueError("invalid SSE TFIM parameters")
    parameters = {
        "LATTICE": "square lattice" if periodic else "open square lattice",
        "MODEL": "spin", "local_S": 0.5, "L": length, "W": length,
        **pauli_to_alps(1.0, gamma),
        "T": 1.0 / beta, "THERMALIZATION": thermalization, "SWEEPS": sweeps,
        "SKIP": skip, "SEED": seed, "WHICH_LOOP_TYPE": "minbounce",
    }
    return parameters


def parameter_text(parameters) -> str:
    def render(value):
        if isinstance(value, str):
            return json.dumps(value)
        return f"{value:.17g}" if isinstance(value, float) else str(value)
    return "\n".join(f"{key}={render(value)}" for key, value in parameters.items()) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--thermalization", type=int, default=10000)
    parser.add_argument("--sweeps", type=int, default=100000)
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    parameters = tfim_parameters(args.L, args.gamma, args.beta, not args.open,
                                 args.thermalization, args.sweeps, args.seed, args.skip)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(parameter_text(parameters))


if __name__ == "__main__":
    main()
