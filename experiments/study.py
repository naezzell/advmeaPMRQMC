"""Reproducible desktop-to-cluster PMR-QMC benchmark campaigns.

The CLI deliberately separates immutable planning from execution:

    python3 experiments/study.py plan --config CONFIG --output CAMPAIGN
    python3 experiments/study.py run --plan CAMPAIGN/plan.csv --resume
    python3 experiments/study.py analyze --output CAMPAIGN
    python3 experiments/study.py promote --output CAMPAIGN --backend slurm

Only compact plans and summaries belong in Git.  Run directories normally live
under ``benchmarking_tests/artifacts`` and are content-addressed by their full
scientific and computational specification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
TRACE_SCHEMA_VERSION = 3
LARGE_ARTIFACT_BYTES = 5 * 1024 * 1024
SOURCE_FILES = (
    "prepare.cpp", "mainqmc.hpp", "divdiff.hpp", "pt_schedule.hpp",
    "PMRQMC_mpi.cpp", "PMRQMC_pt_mpi.cpp", "PMRQMC_qcpt_mpi.cpp",
)


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()


def git_is_dirty() -> bool:
    return bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPOSITORY, text=True
    ).strip())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def pauli_term(coefficient: float, operators: Sequence[Tuple[int, str]]) -> str:
    tokens = [f"{coefficient:.17g}"]
    for site, operator in operators:
        tokens.extend((str(site), operator))
    return " ".join(tokens)


def square_edges(length: int, periodic: bool) -> List[Tuple[int, int]]:
    """Return unique nearest-neighbour bonds with one-indexed row-major sites."""
    if length < 2:
        raise ValueError("square TFIM requires L >= 2")
    edges = set()
    for row in range(length):
        for column in range(length):
            site = row * length + column + 1
            for drow, dcolumn in ((1, 0), (0, 1)):
                other_row, other_column = row + drow, column + dcolumn
                if periodic:
                    other_row %= length
                    other_column %= length
                elif other_row >= length or other_column >= length:
                    continue
                other = other_row * length + other_column + 1
                edges.add(tuple(sorted((site, other))))
    return sorted(edges)


@dataclass(frozen=True)
class HamiltonianSpec:
    family: str
    length: int
    spins: int
    coupling: float
    periodic: bool
    representation: str
    parity: int
    fixed_terms: Tuple[str, ...]
    lambda_terms: Tuple[str, ...]
    observables: Mapping[str, Tuple[str, ...]]
    supported_moves: Tuple[str, ...]

    def combined_terms(self) -> Tuple[str, ...]:
        combined = list(self.fixed_terms)
        for term in self.lambda_terms:
            tokens = term.split()
            tokens[0] = f"{float(tokens[0]) * self.coupling:.17g}"
            combined.append(" ".join(tokens))
        return tuple(combined)

    def metadata(self) -> Dict:
        data = asdict(self)
        data["fixed_terms"] = list(self.fixed_terms)
        data["lambda_terms"] = list(self.lambda_terms)
        data["observables"] = {key: list(value) for key, value in self.observables.items()}
        return data


class ModelFamily:
    name = "abstract"

    def build(self, length: int, coupling: float, periodic: bool,
              representation: str, parity: int = 0) -> HamiltonianSpec:
        raise NotImplementedError


class SquareTFIM(ModelFamily):
    name = "square_tfim"

    def build(self, length: int, coupling: float, periodic: bool = True,
              representation: str = "standard", parity: int = 0) -> HamiltonianSpec:
        edges = square_edges(length, periodic)
        spins = length * length
        if representation == "standard":
            fixed = tuple(pauli_term(-1.0, ((a, "Z"), (b, "Z"))) for a, b in edges)
            varying = tuple(pauli_term(-1.0, ((site, "X"),)) for site in range(1, spins + 1))
            transverse = tuple(pauli_term(-1.0, ((site, "X"),)) for site in range(1, spins + 1))
        elif representation == "rotated_parity":
            fixed = tuple(pauli_term(-1.0, ((a, "X"), (b, "X"))) for a, b in edges)
            varying = tuple(pauli_term(-1.0, ((site, "Z"),)) for site in range(1, spins + 1))
            transverse = tuple(pauli_term(-1.0, ((site, "Z"),)) for site in range(1, spins + 1))
            if parity == 0:
                parity = 1
        else:
            raise ValueError(f"unknown TFIM representation: {representation}")
        magnetization = tuple(
            pauli_term(1.0 / spins, ((site, "X" if representation == "standard" else "Z"),))
            for site in range(1, spins + 1)
        )
        return HamiltonianSpec(
            family=self.name, length=length, spins=spins, coupling=coupling,
            periodic=periodic, representation=representation, parity=parity,
            fixed_terms=fixed, lambda_terms=varying,
            observables={"driving_term": transverse, "transverse_magnetization": magnetization},
            supported_moves=("none", "global_z2") if representation == "standard" else ("none",),
        )


MODEL_FAMILIES = {SquareTFIM.name: SquareTFIM()}


def geometric_beta_schedule(target_beta: float, points: int = 4) -> List[float]:
    if target_beta <= 0 or points < 2:
        raise ValueError("target beta must be positive and points >= 2")
    beta_min = target_beta / 8.0
    return [beta_min * (target_beta / beta_min) ** (index / (points - 1))
            for index in range(points)]


def qcpt_schedule(name: str, target_beta: float, coupling: float) -> List[Dict[str, float]]:
    betas = geometric_beta_schedule(target_beta)
    if name == "pure_beta":
        gammas = [coupling] * 4
    elif name.startswith("diagonal_p"):
        power = float(name.split("p", 1)[1].replace("_", "."))
        gammas = [coupling * (index / 3.0) ** power for index in range(4)]
    elif name == "classical_dogleg":
        betas = [target_beta / 8.0, target_beta / 8.0, target_beta / 2.0, target_beta]
        gammas = [coupling, 0.0, 0.0, coupling]
    else:
        raise ValueError(f"unknown QCPT schedule: {name}")
    return [{"beta": beta, "gamma": gamma, "tau": beta / 2.0}
            for beta, gamma in zip(betas, gammas)]


def beta_schedule(target_beta: float) -> List[Dict[str, float]]:
    return [{"beta": beta, "tau": beta / 2.0}
            for beta in geometric_beta_schedule(target_beta)]


def run_identity(specification: Mapping) -> str:
    return sha256_bytes(canonical_json(specification).encode())[:16]


PLAN_FIELDS = (
    "run_id", "schema_version", "source_commit", "campaign", "model", "method",
    "protocol", "L", "lambda", "beta", "beta_over_L", "periodic",
    "representation", "parity", "move", "seed", "tuning", "Tsteps", "steps",
    "steps_per_measurement", "qmax", "nbins", "schedule_name", "schedule_json",
    "max_wall_seconds", "status",
)


def normalize_row(row: Mapping) -> Dict[str, str]:
    return {field: str(row.get(field, "")) for field in PLAN_FIELDS}


def make_plan_row(commit: str, campaign: str, method: str, protocol: str,
                  length: int, coupling: float, beta: float, periodic: bool,
                  representation: str, parity: int, seed: int, tuning: bool,
                  simulation: Mapping, schedule_name: str = "", move: str = "none") -> Dict[str, str]:
    if method == "qcpt":
        schedule = qcpt_schedule(schedule_name, beta, coupling)
    elif method == "beta_pt":
        schedule = beta_schedule(beta)
        schedule_name = "geometric_beta"
    else:
        schedule = []
        schedule_name = ""
    identity = {
        "schema_version": SCHEMA_VERSION, "source_commit": commit,
        "campaign": campaign, "model": "square_tfim", "method": method,
        "protocol": protocol, "L": length, "lambda": coupling, "beta": beta,
        "periodic": periodic, "representation": representation, "parity": parity,
        "move": move, "seed": seed, "tuning": tuning,
        "Tsteps": int(simulation["Tsteps"]), "steps": int(simulation["steps"]),
        "steps_per_measurement": int(simulation["steps_per_measurement"]),
        "qmax": int(simulation["qmax"]), "nbins": int(simulation["nbins"]),
        "schedule_name": schedule_name, "schedule": schedule,
        "max_wall_seconds": int(simulation["max_wall_seconds"]),
    }
    row = dict(identity)
    row["run_id"] = run_identity(identity)
    row["beta_over_L"] = beta / length
    row["schedule_json"] = canonical_json(schedule)
    row.pop("schedule")
    row["status"] = "planned"
    return normalize_row(row)


def expand_matrix(config: Mapping, commit: str) -> List[Dict[str, str]]:
    campaign = str(config["campaign"])
    simulation = config["simulation"]
    methods = list(config.get("methods", ["current_fixed"]))
    rows = []
    for matrix in config["matrices"]:
        lengths = matrix["L"] if isinstance(matrix["L"], list) else [matrix["L"]]
        couplings = matrix["lambda"] if isinstance(matrix["lambda"], list) else [matrix["lambda"]]
        periodic = bool(matrix.get("periodic", True))
        representation = matrix.get("representation", "standard")
        parity = int(matrix.get("parity", 0))
        tuning = bool(matrix.get("tuning", False))
        matrix_methods = matrix.get("methods", methods)
        moves = matrix.get("moves", ["none"])
        protocols = matrix.get("protocols", ["cheap", "advanced"])
        seeds = matrix.get("seeds", config.get("seeds", [1000, 2000, 3000, 4000]))
        for length in lengths:
            if "beta" in matrix:
                betas = matrix["beta"] if isinstance(matrix["beta"], list) else [matrix["beta"]]
            else:
                ratios = matrix.get("beta_over_L", [0.5, 1.0, 2.0])
                betas = [float(ratio) * int(length) for ratio in ratios]
            for coupling in couplings:
                for beta in betas:
                    for method in matrix_methods:
                        if method == "qcpt":
                            schedule_names = matrix.get("schedules", ["pure_beta"])
                        else:
                            # Beta-only PT has one generated geometric ladder;
                            # fixed methods have no schedule.  QCPT candidate
                            # names in a mixed-method matrix must not duplicate
                            # either control.
                            schedule_names = [""]
                        for schedule_name in schedule_names:
                            for protocol in protocols:
                                for seed in seeds:
                                    for move in moves:
                                        if move == "global_z2" and (representation != "standard" or parity != 0):
                                            raise ValueError("global_z2 is valid only for the standard unrestricted TFIM")
                                        if move not in ("none", "global_z2"):
                                            raise ValueError(f"unknown model-specific move: {move}")
                                        rows.append(make_plan_row(
                                            commit, campaign, method, protocol, int(length),
                                            float(coupling), float(beta), periodic,
                                            representation, parity, int(seed), tuning,
                                            simulation, schedule_name=schedule_name, move=move,
                                        ))
    rows.sort(key=lambda row: tuple(row[field] for field in PLAN_FIELDS[3:-1]))
    duplicate_ids = len(rows) - len({row["run_id"] for row in rows})
    if duplicate_ids:
        raise ValueError(f"configuration generated {duplicate_ids} duplicate runs")
    return rows


def plan_command(args) -> None:
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    commit = source_commit()
    rows = expand_matrix(config, commit)
    write_csv(output / "plan.csv", rows)
    write_json(output / "campaign_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "campaign": config["campaign"],
        "source_commit": commit,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config": config,
        "runs": len(rows),
        "created_unix": time.time(),
    })
    print(f"wrote {len(rows)} runs to {output / 'plan.csv'}")


def parameter_text(row: Mapping[str, str]) -> str:
    protocol = row["protocol"]
    macros = []
    if protocol == "cheap":
        macros = ["MEASURE_H", "MEASURE_H2", "MEASURE_HDIAG", "MEASURE_HOFFDIAG"]
    elif protocol == "advanced":
        macros = ["MEASURE_H", "MEASURE_H2", "MEASURE_HOFFDIAG",
                  "MEASURE_HOFFDIAG_EINT", "MEASURE_HOFFDIAG_FINT"]
    elif protocol == "advanced_slow":
        macros = ["MEASURE_H", "MEASURE_H2", "MEASURE_HOFFDIAG",
                  "MEASURE_HOFFDIAG_EINT", "MEASURE_HOFFDIAG_FINT",
                  "USE_SLOW_FS_ESTIMATOR"]
    elif protocol == "es_only":
        macros = ["MEASURE_H", "MEASURE_H2", "MEASURE_HOFFDIAG",
                  "MEASURE_HOFFDIAG_EINT"]
    elif protocol == "fs_only":
        macros = ["MEASURE_H", "MEASURE_H2", "MEASURE_HOFFDIAG",
                  "MEASURE_HOFFDIAG_FINT"]
    elif protocol == "all":
        macros = ["MEASURE_H", "MEASURE_H2", "MEASURE_HDIAG", "MEASURE_HDIAG_EINT",
                  "MEASURE_HDIAG_FINT", "MEASURE_HOFFDIAG", "MEASURE_HOFFDIAG_EINT",
                  "MEASURE_HOFFDIAG_FINT"]
    elif protocol != "none":
        raise ValueError(f"unknown measurement protocol: {protocol}")
    definitions = "\n".join(f"#define {macro}" for macro in macros)
    move_definition = "#define TFIM_GLOBAL_Z2_MOVE" if row.get("move", "none") == "global_z2" else ""
    return f"""#define Tsteps {int(row['Tsteps'])}
#define steps {int(row['steps'])}
#define stepsPerMeasurement {int(row['steps_per_measurement'])}
#define beta {float(row['beta']):.17g}
#define tau {float(row['beta']) / 2.0:.17g}
#define gamma {float(row['lambda']):.17g}
#define parity_cond {int(row['parity'])}
#define qmax {int(row['qmax'])}
#define Nbins {int(row['nbins'])}
#define EXHAUSTIVE_CYCLE_SEARCH
#define GAPS_GEOMETRIC_PARAMETER 0.8
#define COMPOSITE_UPDATE_BREAK_PROBABILITY 0.9
#define EXACTLY_REPRODUCIBLE
#define RNG_SEED_OFFSET {int(row['seed'])}
#define SAVE_UNFINISHED_CALCULATION
#define RESUME_CALCULATION
#define HURRY_ON_SIGTERM
{move_definition}
{definitions}
"""


def stage_sources(run_directory: Path) -> None:
    for name in SOURCE_FILES:
        shutil.copy2(REPOSITORY / name, run_directory / name)


def write_model_files(run_directory: Path, spec: HamiltonianSpec) -> None:
    (run_directory / "H_fixed.txt").write_text("\n".join(spec.fixed_terms) + "\n")
    (run_directory / "H_gamma.txt").write_text("\n".join(spec.lambda_terms) + "\n")
    (run_directory / "H.txt").write_text("\n".join(spec.combined_terms()) + "\n")
    for name, terms in spec.observables.items():
        (run_directory / f"{name}.txt").write_text("\n".join(terms) + "\n")
    write_json(run_directory / "model.json", spec.metadata())


def write_schedule_file(run_directory: Path, method: str, schedule: Sequence[Mapping]) -> Path:
    path = run_directory / "schedule.txt"
    if method == "qcpt":
        body = "# beta gamma tau\n" + "".join(
            f"{point['beta']:.17g} {point['gamma']:.17g} {point['tau']:.17g}\n"
            for point in schedule
        )
    else:
        body = "# beta tau\n" + "".join(
            f"{point['beta']:.17g} {point['tau']:.17g}\n" for point in schedule
        )
    path.write_text(body)
    return path


def command_environment() -> Dict:
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo.read_text(), re.MULTILINE)
        if match:
            cpu_model = match.group(1)
    return {
        "hostname": platform.node(), "platform": platform.platform(),
        "python": sys.version, "cpu_count": os.cpu_count(), "cpu_model": cpu_model,
        "compiler": subprocess.run(["g++", "--version"], text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.splitlines()[0],
        "mpi_compiler": subprocess.run(["mpicxx", "--version"], text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.splitlines()[0],
    }


def run_logged(command: Sequence[str], directory: Path, log_name: str,
               timeout: Optional[int] = None) -> float:
    started = time.perf_counter()
    with (directory / log_name).open("w") as log:
        subprocess.run(list(command), cwd=directory, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=timeout)
    return time.perf_counter() - started


def execute_row(row: Mapping[str, str], campaign_root: Path, resume: bool,
                oversubscribe: bool, dry_run: bool) -> Dict:
    run_directory = campaign_root / "runs" / row["run_id"]
    summary_path = run_directory / "summary.json"
    if resume and summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing.get("status") == "complete":
            return existing
    if source_commit() != row["source_commit"]:
        raise RuntimeError(
            f"run {row['run_id']} was planned at {row['source_commit']} but HEAD is {source_commit()}; re-plan"
        )
    run_directory.mkdir(parents=True, exist_ok=True)
    family = MODEL_FAMILIES[row["model"]]
    spec = family.build(int(row["L"]), float(row["lambda"]), row["periodic"] == "True",
                        row["representation"], int(row["parity"]))
    schedule = json.loads(row["schedule_json"])
    manifest = {
        "schema_version": SCHEMA_VERSION, "trace_schema_version": TRACE_SCHEMA_VERSION,
        "run_id": row["run_id"], "planned": dict(row), "source_commit": source_commit(),
        "dirty_worktree": git_is_dirty(), "environment": command_environment(),
        "archive_root": str(campaign_root), "created_unix": time.time(),
    }
    write_json(run_directory / "manifest.json", manifest)
    stage_sources(run_directory)
    write_model_files(run_directory, spec)
    (run_directory / "parameters.hpp").write_text(parameter_text(row))
    method = row["method"]
    prepare = ["./prepare.bin"]
    if method == "qcpt":
        prepare += ["--hamiltonian-fixed", "H_fixed.txt", "--hamiltonian-gamma", "H_gamma.txt"]
    else:
        prepare += ["H.txt"]
    prepare += ["driving_term.txt", "transverse_magnetization.txt"]
    ranks = 4
    if method == "current_fixed":
        binary, source, defines = "PMRQMC_mpi.bin", "PMRQMC_mpi.cpp", []
        launch = ["mpirun"] + (["--oversubscribe"] if oversubscribe else []) + [
            "-n", str(ranks), "./PMRQMC_mpi.bin", "--timeseries-prefix", "trace.csv",
            "--stream-timeseries-prefix", "trace_stream"
        ]
    elif method in ("beta_pt", "qcpt"):
        binary = "PMRQMC_qcpt_mpi.bin" if method == "qcpt" else "PMRQMC_pt_mpi.bin"
        source = "PMRQMC_qcpt_mpi.cpp" if method == "qcpt" else "PMRQMC_pt_mpi.cpp"
        defines = ["-DPMR_QCPT"] if method == "qcpt" else []
        write_schedule_file(run_directory, method, schedule)
        launch = ["mpirun"] + (["--oversubscribe"] if oversubscribe else []) + [
            "-n", str(ranks), f"./{binary}", "--schedule", "schedule.txt",
            "--updates-per-exchange", "10", "--independent-ladders", "1",
            "--output-prefix", "tempered", "--timeseries-prefix", "trace.csv",
            "--stream-timeseries-prefix", "trace_stream",
            "--checkpoint-every", str(max(1, min(100000, int(row["steps"]) // 10))),
        ]
        checkpoint_suffix = ".qcptckpt" if method == "qcpt" else ".ptckpt"
        if resume and len(list(run_directory.glob(f"tempered.rank*{checkpoint_suffix}"))) == ranks:
            launch.append("--resume")
    else:
        raise ValueError(f"execution adapter not implemented for method {method}")
    commands = {
        "build_prepare": ["g++", "-O3", "-std=c++11", "-o", "prepare.bin", "prepare.cpp"],
        "prepare": prepare,
        "build_simulator": ["mpicxx", "-O3", "-std=c++11"] + defines + ["-o", binary, source],
        "launch": launch,
    }
    write_json(run_directory / "commands.json", commands)
    if dry_run:
        return {"run_id": row["run_id"], "status": "dry_run", "run_directory": str(run_directory)}
    summary = {"run_id": row["run_id"], "status": "running", "run_directory": str(run_directory)}
    write_json(summary_path, summary)
    try:
        timings = {}
        for key in ("build_prepare", "prepare", "build_simulator"):
            timings[key] = run_logged(commands[key], run_directory, f"{key}.log")
        timings["simulation"] = run_logged(
            commands["launch"], run_directory, "simulation.log",
            timeout=int(row["max_wall_seconds"]),
        )
        artifacts = []
        for path in sorted(run_directory.iterdir()):
            if path.is_file():
                artifacts.append({
                    "name": path.name, "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "large": path.stat().st_size > LARGE_ARTIFACT_BYTES,
                })
        summary.update({"status": "complete", "timings": timings, "artifacts": artifacts})
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        summary.update({"status": "failed", "error": repr(error)})
    write_json(summary_path, summary)
    return summary


def run_command(args) -> None:
    plan_path = Path(args.plan).resolve()
    campaign_root = plan_path.parent
    rows = read_csv(plan_path)
    if args.run_id:
        requested = set(args.run_id)
        rows = [row for row in rows if row["run_id"] in requested]
        missing = requested - {row["run_id"] for row in rows}
        if missing:
            raise SystemExit("unknown run IDs: " + ", ".join(sorted(missing)))
    summaries = []
    for index, row in enumerate(rows, start=1):
        deadline = getattr(args, "deadline", None)
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= float(row["max_wall_seconds"]):
                print(f"desktop campaign budget reached with {len(rows)-index+1} runs remaining")
                break
        print(f"[{index}/{len(rows)}] {row['run_id']} {row['method']} L={row['L']}")
        summaries.append(execute_row(row, campaign_root, args.resume,
                                     args.oversubscribe, args.dry_run))
        write_json(campaign_root / "execution_summary.json", summaries)


def promote_command(args) -> None:
    root = Path(args.output).resolve()
    plan = read_csv(root / "plan.csv")
    summaries = {}
    for path in (root / "runs").glob("*/summary.json") if (root / "runs").exists() else []:
        data = json.loads(path.read_text())
        summaries[data["run_id"]] = data
    promoted = []
    for row in plan:
        summary = summaries.get(row["run_id"])
        if summary and summary.get("status") == "complete":
            candidate = dict(row)
            candidate["status"] = "promoted"
            promoted.append(candidate)
    write_csv(root / "cluster_plan.csv", promoted)
    if args.backend != "slurm":
        raise ValueError("only the slurm promotion backend is currently supported")
    script = """#!/bin/bash
# Generated only; site-specific account, partition, time and modules are required.
set -euo pipefail
PLAN=${1:?cluster plan path required}
INDEX=${SLURM_ARRAY_TASK_ID:?run as a Slurm array}
RUN_ID=$(python3 -c 'import csv,sys; r=list(csv.DictReader(open(sys.argv[1]))); print(r[int(sys.argv[2])]["run_id"])' "$PLAN" "$INDEX")
python3 experiments/study.py run --plan "$PLAN" --run-id "$RUN_ID" --resume
"""
    (root / "slurm_array.sh").write_text(script)
    (root / "slurm_array.sh").chmod(0o755)
    print(f"promoted {len(promoted)} completed runs")


def analyze_command(args) -> None:
    # The statistical implementation is imported lazily so planning remains
    # usable on minimal cluster login nodes.
    from study_stats import analyze_campaign
    analyze_campaign(Path(args.output).resolve())


def pilot_command(args) -> None:
    root = Path(args.output).resolve()
    plan_path = root / "plan.csv"
    if not plan_path.exists():
        plan_args = argparse.Namespace(config=args.config, output=str(root))
        plan_command(plan_args)
    deadline = time.monotonic() + args.budget_hours * 3600.0
    run_args = argparse.Namespace(
        plan=str(plan_path), resume=args.resume, oversubscribe=args.oversubscribe,
        dry_run=args.dry_run, run_id=None, deadline=deadline,
    )
    run_command(run_args)
    if not args.dry_run:
        analyze_command(argparse.Namespace(output=str(root)))
        for _ in range(args.adaptive_rounds):
            extension_path = root / "extension_plan.csv"
            extensions = read_csv(extension_path) if extension_path.exists() and extension_path.stat().st_size else []
            plan_rows = read_csv(plan_path)
            existing = {row["run_id"] for row in plan_rows}
            new_rows = [row for row in extensions if row["run_id"] not in existing]
            if not new_rows or time.monotonic() >= deadline:
                break
            write_csv(plan_path, plan_rows + new_rows)
            run_args.run_id = [row["run_id"] for row in new_rows]
            run_command(run_args)
            analyze_command(argparse.Namespace(output=str(root)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--config", required=True)
    plan.add_argument("--output", required=True)
    plan.set_defaults(function=plan_command)
    run = subparsers.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--run-id", action="append")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--oversubscribe", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(function=run_command)
    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--config", required=True)
    pilot.add_argument("--output", required=True)
    pilot.add_argument("--resume", action="store_true")
    pilot.add_argument("--oversubscribe", action="store_true")
    pilot.add_argument("--dry-run", action="store_true")
    pilot.add_argument("--budget-hours", type=float, default=24.0)
    pilot.add_argument("--adaptive-rounds", type=int, default=3)
    pilot.set_defaults(function=pilot_command)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--output", required=True)
    analyze.set_defaults(function=analyze_command)
    promote = subparsers.add_parser("promote")
    promote.add_argument("--output", required=True)
    promote.add_argument("--backend", choices=("slurm",), default="slurm")
    promote.set_defaults(function=promote_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
