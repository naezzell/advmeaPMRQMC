"""Run fresh fixed-parameter PMR simulations for a list of target betas."""

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from ioscripts import make_all_stand_param_fstr


def parse_values(text):
    return [float(value) for value in text.split(",") if value.strip()]


def parse_schedule_mapping(values):
    result = {}
    for value in values:
        try:
            separator = ":" if ":" in value else "="
            beta_text, path_text = value.split(separator, 1)
            beta = float(beta_text)
        except ValueError as error:
            raise ValueError("absolute schedules must use BETA:FILE") from error
        if beta in result:
            raise ValueError(f"duplicate absolute schedule for beta {beta}")
        result[beta] = Path(path_text).resolve()
    return result


def target_directory_name(index, beta):
    token = re.sub(r"[^A-Za-z0-9_.-]", "_", format(beta, ".17g"))
    return f"target_{index:03d}_beta_{token}"


def annealing_plan_hash(beta, tau, tsteps, interval, factor, schedule):
    description = {
        "target_beta": beta, "target_tau": tau, "Tsteps": tsteps,
        "anneal_interval": interval if interval is not None else "N",
        "start_factor": None if schedule else factor,
        "schedule_sha256": (hashlib.sha256(schedule.read_bytes()).hexdigest()
                            if schedule else None),
    }
    encoded = json.dumps(description, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), description["schedule_sha256"]


def build(directory, beta, tau, tsteps, steps, measurement_interval, qmax,
          observables):
    parameters = make_all_stand_param_fstr(
        beta, tau, tsteps, steps, measurement_interval, 0, False, False)
    parameters = re.sub(r"#define\s+qmax\s+\d+", f"#define qmax {int(qmax)}", parameters)
    (directory / "parameters.hpp").write_text(parameters)
    subprocess.run(["g++", "-O3", "-std=c++11", "-o", "prepare.bin", "prepare.cpp"],
                   cwd=directory, check=True)
    subprocess.run(["./prepare.bin", "H.txt", *observables], cwd=directory, check=True)
    subprocess.run(["g++", "-O3", "-std=c++11", "-o", "PMRQMC.bin", "PMRQMC.cpp"],
                   cwd=directory, check=True)
    subprocess.run(["mpicxx", "-O3", "-std=c++11", "-o", "PMRQMC_mpi.bin", "PMRQMC_mpi.cpp"],
                   cwd=directory, check=True)


def run_fixed_beta_anneals(directory, beta_list, tau_list=None, mpi_ranks=1,
                           start_factor=0.001, anneal_interval=None,
                           absolute_schedules=None, tsteps=1000000,
                           steps=1000000, measurement_interval=100,
                           qmax=1000, observables=None, output="beta_anneals",
                           oversubscribe=False, use_single=False):
    directory = Path(directory).resolve()
    beta_list = [float(value) for value in beta_list]
    if not beta_list or any(beta <= 0 for beta in beta_list):
        raise ValueError("beta_list must contain positive values")
    tau_list = ([beta / 2 for beta in beta_list] if tau_list is None
                else [float(value) for value in tau_list])
    if len(tau_list) != len(beta_list):
        raise ValueError("tau_list must have one entry per beta")
    if any(tau < 0 or tau > beta for beta, tau in zip(beta_list, tau_list)):
        raise ValueError("every tau must satisfy 0 <= tau <= beta")
    if mpi_ranks < 1 or (use_single and mpi_ranks != 1):
        raise ValueError("mpi_ranks must be positive and must equal one for --single")
    if not (0 < start_factor <= 1):
        raise ValueError("start_factor must satisfy 0 < factor <= 1")
    if anneal_interval is not None and anneal_interval <= 0:
        raise ValueError("anneal_interval must be positive")
    schedules = dict(absolute_schedules or {})
    unknown = set(schedules) - set(beta_list)
    if unknown:
        raise ValueError(f"absolute schedules have no matching target: {sorted(unknown)}")
    for path in schedules.values():
        if not path.is_file():
            raise ValueError(f"absolute schedule does not exist: {path}")

    output_root = (directory / output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    build(directory, beta_list[0], tau_list[0], tsteps, steps,
          measurement_interval, qmax, list(observables or []))
    binary = (directory / ("PMRQMC.bin" if use_single else "PMRQMC_mpi.bin")).resolve()
    records = []
    for index, (target_beta, target_tau) in enumerate(zip(beta_list, tau_list)):
        target_dir = output_root / target_directory_name(index, target_beta)
        target_dir.mkdir(parents=True, exist_ok=True)
        command = [str(binary), "--target-beta", format(target_beta, ".17g"),
                   "--target-tau", format(target_tau, ".17g")]
        schedule = schedules.get(target_beta)
        if schedule:
            command += ["--beta-anneal-schedule", str(schedule)]
        else:
            command += ["--beta-anneal", "--anneal-start-factor", format(start_factor, ".17g")]
        if anneal_interval is not None:
            command += ["--anneal-interval", str(int(anneal_interval))]
        trace = target_dir / "trace.csv"
        if not use_single:
            command += ["--timeseries-prefix", str(trace)]
            prefix = ["mpirun"]
            if oversubscribe:
                prefix.append("--oversubscribe")
            command = prefix + ["-n", str(int(mpi_ranks))] + command
        started = time.perf_counter()
        with (target_dir / "stdout.log").open("w") as log:
            subprocess.run(command, cwd=target_dir, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        log_text = (target_dir / "stdout.log").read_text()
        seeds = [int(value) for value in re.findall(r"RNG seed = (\d+)", log_text)]
        plan_hash, schedule_hash = annealing_plan_hash(
            target_beta, target_tau, tsteps, anneal_interval, start_factor, schedule)
        records.append({
            "index": index, "beta": target_beta, "tau": target_tau,
            "directory": str(target_dir), "schedule": str(schedule) if schedule else None,
            "start_factor": None if schedule else start_factor,
            "annealing_plan_sha256": plan_hash,
            "schedule_sha256": schedule_hash,
            "anneal_interval": anneal_interval if anneal_interval is not None else "N",
            "Tsteps": tsteps, "steps": steps,
            "rng_seeds": seeds,
            "wall_seconds": time.perf_counter() - started,
            "command": command,
        })
    manifest = {"fresh_run_per_target": True, "targets": records}
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--betas", required=True)
    parser.add_argument("--taus")
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument("--start-factor", type=float, default=0.001)
    parser.add_argument("--anneal-interval", type=int)
    parser.add_argument("--absolute-schedule", action="append", default=[], metavar="BETA:FILE")
    parser.add_argument("--Tsteps", type=int, default=1000000)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--steps-per-measurement", type=int, default=100)
    parser.add_argument("--qmax", type=int, default=1000)
    parser.add_argument("--observable", action="append", default=[])
    parser.add_argument("--output", default="beta_anneals")
    parser.add_argument("--oversubscribe", action="store_true")
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()
    try:
        schedules = parse_schedule_mapping(args.absolute_schedule)
        manifest = run_fixed_beta_anneals(
            args.directory, parse_values(args.betas),
            parse_values(args.taus) if args.taus else None,
            args.mpi_ranks, args.start_factor, args.anneal_interval,
            schedules, args.Tsteps, args.steps, args.steps_per_measurement,
            args.qmax, args.observable, args.output, args.oversubscribe,
            args.single)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
