"""Small, model-independent driver for the beta-only PMR tempering binary.

The model-specific drivers can continue to generate H.txt and observables as
before, then call :func:`run_parallel_tempering` with the desired schedule.
"""

import os
import re
import sys
import subprocess
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from ioscripts import make_all_stand_param_fstr


def generate_tempering_schedule(path, beta_list, tau_list=None):
    """Write the two-column schedule consumed by ``PMRQMC_pt_mpi.bin``."""
    beta_list = [float(value) for value in beta_list]
    if tau_list is None:
        tau_list = [value / 2.0 for value in beta_list]
    tau_list = [float(value) for value in tau_list]
    if len(beta_list) == 0 or len(beta_list) != len(tau_list):
        raise ValueError("beta_list and tau_list must be non-empty and have equal length")
    if any(beta <= 0 for beta in beta_list):
        raise ValueError("all beta values must be positive")
    if any(beta <= previous for previous, beta in zip(beta_list, beta_list[1:])):
        raise ValueError("beta values must be strictly increasing")
    if any(tau < 0 or tau > beta for beta, tau in zip(beta_list, tau_list)):
        raise ValueError("each tau must satisfy 0 <= tau <= beta")
    path = Path(path)
    path.write_text("# beta tau\n" + "".join(f"{beta:.17g} {tau:.17g}\n" for beta, tau in zip(beta_list, tau_list)))
    return path


def run_parallel_tempering(directory, beta_list, tau_list=None, updates_per_exchange=10,
                           independent_ladders=1, nt=None, qmax=1000, Tsteps=100000,
                           steps=1000000, stepsPerMeasurement=10, parity=0,
                           output_prefix="pmrqmc_pt", resume=False, checkpoint_every=0,
                           observables=None):
    """Prepare, compile, and launch a beta-only tempering run.

    ``directory`` must contain ``H.txt`` and may contain observable files.  The
    MPI size is always ``len(beta_list) * independent_ladders`` unless an
    explicit, matching ``nt`` is supplied.
    """
    directory = Path(directory)
    beta_list = list(beta_list)
    expected_nt = len(beta_list) * int(independent_ladders)
    if nt is None:
        nt = expected_nt
    if int(nt) != expected_nt:
        raise ValueError("nt must equal len(beta_list) * independent_ladders")
    if updates_per_exchange < 1:
        raise ValueError("updates_per_exchange must be positive")

    schedule = generate_tempering_schedule(directory / "tempering_schedule.txt", beta_list, tau_list)
    parameters = make_all_stand_param_fstr(beta_list[0], (tau_list or [beta_list[0] / 2])[0],
                                           Tsteps, steps, stepsPerMeasurement, parity, True, resume)
    parameters = re.sub(r"#define\s+qmax\s+\d+", f"#define qmax {int(qmax)}", parameters)
    (directory / "parameters.hpp").write_text(parameters)

    subprocess.run(["g++", "-O3", "-std=c++11", "-o", "prepare.bin", "prepare.cpp"],
                   cwd=directory, check=True)
    files = ["H.txt"] + list(observables or [])
    subprocess.run(["./prepare.bin"] + files, cwd=directory, check=True)
    subprocess.run(["mpicxx", "-O3", "-std=c++11", "-o", "PMRQMC_pt_mpi.bin", "PMRQMC_pt_mpi.cpp"],
                   cwd=directory, check=True)
    command = ["mpirun", "-n", str(expected_nt), "./PMRQMC_pt_mpi.bin",
               "--schedule", str(schedule), "--updates-per-exchange", str(updates_per_exchange),
               "--output-prefix", output_prefix]
    if independent_ladders != 1:
        command += ["--independent-ladders", str(independent_ladders)]
    if checkpoint_every:
        command += ["--checkpoint-every", str(checkpoint_every)]
    if resume:
        command.append("--resume")
    return subprocess.run(command, cwd=directory, check=True)
