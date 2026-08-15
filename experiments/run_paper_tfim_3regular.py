"""Plan and run paper-matched PMR-QMC experiments.

Generated instances and run outputs are local artifacts.  Use ``plan`` to
inspect a preset without compiling or running MPI, ``generate`` to materialize
the catalog, and ``run`` for one selected instance/configuration.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from experiments.paper_tfim_3regular import generate_catalog, write_instance
except ModuleNotFoundError:  # direct ``python experiments/run_...py`` invocation
    from paper_tfim_3regular import generate_catalog, write_instance


BETA_2019 = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0)
GAMMA_2019 = (0.1, 0.4)
DEFAULT_CATALOG = Path(__file__).parents[1] / "instances" / "paper_tfim_3regular_catalog.json"
SOURCE_FILES = ("PMRQMC_pt_mpi.cpp", "PMRQMC_qcpt_mpi.cpp", "mainqmc.hpp", "divdiff.hpp", "pt_schedule.hpp", "beta_anneal.hpp")


def gamma_2017(beta: float) -> float:
    if beta <= 0:
        raise ValueError("beta must be positive")
    return (10.0 * beta) ** -0.5


def load_catalog(path: Path) -> dict:
    return json.loads(path.read_text())


def entries_for_preset(catalog: dict, preset: str) -> list[dict]:
    if preset == "2019":
        return [entry for entry in catalog["instances"] if entry["n"] in (96, 128)]
    if preset == "2017":
        return [entry for entry in catalog["instances"] if entry["id"] in ("fixture_12", "fixture_60")]
    raise ValueError(f"unknown preset: {preset}")


def schedule_for_preset(preset: str, gamma: float | None = None) -> list[tuple[float, float, float]]:
    if preset == "2019":
        selected_gamma = GAMMA_2019[0] if gamma is None else gamma
        return [(beta, selected_gamma, beta / 2.0) for beta in BETA_2019]
    if preset == "2017":
        return [(beta, gamma_2017(beta), beta / 2.0) for beta in BETA_2019]
    raise ValueError(f"unknown preset: {preset}")


def expand_plan(catalog: dict, preset: str) -> list[dict]:
    rows = []
    entries = entries_for_preset(catalog, preset)
    gammas = GAMMA_2019 if preset == "2019" else (None,)
    for entry in entries:
        for gamma in gammas:
            rows.append({
                "preset": preset,
                "instance_id": entry["id"],
                "n": entry["n"],
                "seed": entry["seed"],
                "gamma": gamma,
                "schedule": schedule_for_preset(preset, gamma),
            })
    return rows


def write_schedule(path: Path, preset: str, gamma: float | None) -> list[tuple[float, float, float]]:
    schedule = schedule_for_preset(preset, gamma)
    with path.open("w") as output:
        if preset == "2017":
            output.write("# beta gamma tau\n")
            for beta, schedule_gamma, tau in schedule:
                output.write(f"{beta:.17g} {schedule_gamma:.17g} {tau:.17g}\n")
        else:
            output.write("# beta tau\n")
            for beta, _, tau in schedule:
                output.write(f"{beta:.17g} {tau:.17g}\n")
    return schedule


def write_parameters(path: Path, *, beta: float, gamma: float, seed: int,
                     tsteps: int, steps: int, steps_per_measurement: int,
                     nbins: int, qmax: int) -> None:
    path.write_text(f"""#define Tsteps {tsteps}
#define steps {steps}
#define stepsPerMeasurement {steps_per_measurement}
#define beta {beta:.17g}
#define tau {beta / 2.0:.17g}
#define gamma {gamma:.17g}
#define parity_cond 0
#define qmax {qmax}
#define Nbins {nbins}
#define EXHAUSTIVE_CYCLE_SEARCH
#define GAPS_GEOMETRIC_PARAMETER 0.8
#define COMPOSITE_UPDATE_BREAK_PROBABILITY 0.9
#define EXACTLY_REPRODUCIBLE
#define RNG_SEED_OFFSET {seed}
#define MEASURE_H
#define MEASURE_HOFFDIAG
""")


def run_command(command: list[str], cwd: Path, log: Path) -> None:
    with log.open("w") as stream:
        subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)


def prepare_run_directory(run_dir: Path, instance_dir: Path, source_root: Path,
                          n: int, seed: int, gamma: float, preset: str,
                          tsteps: int, steps: int, steps_per_measurement: int,
                          nbins: int, qmax: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("H_fixed.txt", "H_gamma.txt", "instance.json"):
        shutil.copy2(instance_dir / name, run_dir / name)
    write_parameters(run_dir / "parameters.hpp", beta=0.1, gamma=gamma, seed=seed,
                     tsteps=tsteps, steps=steps, steps_per_measurement=steps_per_measurement,
                     nbins=nbins, qmax=qmax)
    run_command(["g++", "-O2", "-std=c++11", "-o", "prepare.bin", str(source_root / "prepare.cpp")],
                run_dir, run_dir / "prepare.log")
    run_command(["./prepare.bin", "--hamiltonian-fixed", "H_fixed.txt",
                 "--hamiltonian-gamma", "H_gamma.txt"], run_dir, run_dir / "prepare.log")


def run_one(run_dir: Path, source_root: Path, preset: str, gamma: float | None,
            updates_per_exchange: int, oversubscribe: bool, resume: bool) -> None:
    metadata = json.loads((run_dir / "instance.json").read_text())
    n = metadata["n"]
    selected_gamma = GAMMA_2019[0] if gamma is None else gamma
    schedule = write_schedule(run_dir / "schedule.txt", preset, gamma)
    for source in SOURCE_FILES:
        shutil.copy2(source_root / source, run_dir / source)
    is_qcpt = preset == "2017"
    program_source = "PMRQMC_qcpt_mpi.cpp" if is_qcpt else "PMRQMC_pt_mpi.cpp"
    binary = "PMRQMC_qcpt_mpi.bin" if is_qcpt else "PMRQMC_pt_mpi.bin"
    compile_command = ["mpicxx", "-O2", "-std=c++11"]
    if is_qcpt:
        compile_command.append("-DPMR_QCPT")
    compile_command += ["-o", binary, program_source]
    run_command(compile_command, run_dir, run_dir / "compile.log")
    command = ["mpirun"]
    if oversubscribe:
        command.append("--oversubscribe")
    command += ["-n", str(len(schedule)), f"./{binary}", "--schedule", "schedule.txt",
                "--updates-per-exchange", str(updates_per_exchange),
                "--output-prefix", "pmr", "--timeseries-prefix", "timeseries.csv"]
    if resume:
        command.append("--resume")
    run_command(command, run_dir, run_dir / "run.log")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "generate", "run"))
    parser.add_argument("--preset", choices=("2017", "2019"), default="2019")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--instances-root", type=Path, default=Path("generated_instances/paper_tfim_3regular"))
    parser.add_argument("--run-root", type=Path, default=Path("run_outputs/paper_tfim_3regular"))
    parser.add_argument("--instance-id")
    parser.add_argument("--n", type=int)
    parser.add_argument("--gamma", type=float, choices=GAMMA_2019)
    parser.add_argument("--updates-per-exchange", type=int, default=10)
    parser.add_argument("--tsteps", type=int, default=100000)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--steps-per-measurement", type=int, default=10)
    parser.add_argument("--nbins", type=int, default=100)
    parser.add_argument("--qmax", type=int, default=1000)
    parser.add_argument("--oversubscribe", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    catalog = load_catalog(args.catalog)
    source_root = Path(__file__).parents[1]

    if args.command == "plan":
        plan = expand_plan(catalog, args.preset)
        text = json.dumps({"preset": args.preset, "runs": plan}, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        else:
            print(text, end="")
        return

    if args.command == "generate":
        for path in generate_catalog(args.catalog, args.instances_root):
            print(path)
        return

    if not args.instance_id or args.n is None:
        parser.error("run requires --instance-id and --n")
    entry = next((item for item in catalog["instances"]
                  if item["id"] == args.instance_id and item["n"] == args.n), None)
    if entry is None:
        parser.error("instance ID and N are not present in the catalog")
    instance_dir = args.instances_root / f"N{args.n:03d}" / args.instance_id
    if not instance_dir.exists():
        write_instance(instance_dir, args.n, entry["seed"])
    selected_gamma = gamma_2017(0.1) if args.preset == "2017" else (args.gamma or GAMMA_2019[0])
    run_id = f"{args.preset}_{args.instance_id}_gamma_{selected_gamma:g}"
    run_dir = args.run_root / run_id
    if run_dir.exists() and not args.resume:
        raise SystemExit(f"run directory already exists: {run_dir}; use --resume or choose another output root")
    if not run_dir.exists():
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        prepare_run_directory(run_dir, instance_dir, source_root, args.n, entry["seed"], selected_gamma,
                              args.preset, args.tsteps, args.steps, args.steps_per_measurement,
                              args.nbins, args.qmax)
    manifest = {
        "preset": args.preset, "instance_id": args.instance_id, "n": args.n,
        "seed": entry["seed"], "gamma": selected_gamma,
        "updates_per_exchange": args.updates_per_exchange,
        "tsteps": args.tsteps, "steps": args.steps,
        "steps_per_measurement": args.steps_per_measurement,
        "nbins": args.nbins, "qmax": args.qmax,
        "schedule": schedule_for_preset(args.preset, args.gamma),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    run_one(run_dir, source_root, args.preset, args.gamma, args.updates_per_exchange,
            args.oversubscribe, args.resume)


if __name__ == "__main__":
    main()
