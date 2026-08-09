"""Exercise deterministic uninterrupted versus interrupted/resumed PT runs.

The test deliberately uses separate output prefixes and compares the restored
bin accumulators, swap counts, flow counts, and final observables.  Timing
fields are excluded because interruption changes wall-clock accounting.
"""

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import validate_qcpt as fixture


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def run(command, directory, log_name, check=True):
    with (Path(directory) / log_name).open("w") as log:
        return subprocess.run(command, cwd=directory, stdout=log,
                              stderr=subprocess.STDOUT, check=check)


def prepare_stage(stage, source_root, combined, absolute, args):
    fixture.copy_sources(stage, source_root)
    fixture.write_fixture(stage)
    fixture.write_parameters(stage / "parameters.hpp", fixture.PATH[0][0], fixture.PATH[0][1],
                             args.Tsteps, args.steps, args.interval, args.nbins, absolute)
    if combined:
        fixture.prepare_combined(stage)
    else:
        fixture.prepare_split(stage)


def build(stage, qcpt):
    source = "PMRQMC_qcpt_mpi.cpp" if qcpt else "PMRQMC_pt_mpi.cpp"
    binary = "PMRQMC_qcpt_mpi.bin" if qcpt else "PMRQMC_pt_mpi.bin"
    flags = ["-DPMR_QCPT"] if qcpt else []
    run(["mpicxx", "-O1", "-std=c++11", *flags, "-o", binary, source],
        stage, "build.log")
    return binary


def mpi_command(args, ranks):
    command = ["mpirun"]
    if args.oversubscribe:
        command.append("--oversubscribe")
    return command + ["-n", str(ranks)]


def execute(stage, binary, schedule, prefix, args, qcpt, resume=False,
            checkpoint_every=0, interrupt=False):
    command = mpi_command(args, len(fixture.PATH) if qcpt else len(fixture.PATH))
    command += [f"./{binary}", "--schedule", schedule,
                "--updates-per-exchange", str(args.updates_per_exchange),
                "--checkpoint-every", str(checkpoint_every),
                "--output-prefix", prefix,
                "--timeseries-prefix", prefix + "_trace.csv"]
    if args.beta_anneal:
        command += ["--beta-anneal", "--anneal-interval", str(args.anneal_interval)]
    if resume:
        command.append("--resume")
    if interrupt:
        command.append("--stop-after-checkpoint")
    if not interrupt:
        run(command, stage, prefix + ".log")
        return "completed"
    run(command, stage, prefix + ".log")
    checkpoint_suffix = "qcptckpt" if qcpt else "ptckpt"
    checkpoints = list(stage.glob(f"{prefix}.rank*.{checkpoint_suffix}"))
    if len(checkpoints) != len(fixture.PATH):
        raise RuntimeError("interrupted run did not publish a complete checkpoint set")
    return "interrupted"


def comparable_csv(path, ignored=("elapsed_seconds", "crossed_weight_seconds",
                                  "measurement_seconds")):
    rows = read_csv(path)
    for row in rows:
        for key in ignored:
            row.pop(key, None)
    return rows


def csv_equal(left, right, tolerance=1e-8):
    if len(left) != len(right):
        return False
    for left_row, right_row in zip(left, right):
        if left_row.keys() != right_row.keys():
            return False
        for key in left_row:
            try:
                if abs(float(left_row[key]) - float(right_row[key])) > tolerance:
                    return False
            except ValueError:
                if left_row[key] != right_row[key]:
                    return False
    return True


def compare_outputs(full, resumed, prefix):
    names = ("_observables.csv", "_swaps.csv", "_flow.csv")
    differences = {}
    for suffix in names:
        left = comparable_csv(Path(full) / ("full" + suffix))
        right = comparable_csv(Path(resumed) / (prefix + suffix))
        differences[suffix] = not csv_equal(left, right)
    return differences


def one_case(root, name, args, qcpt, absolute):
    case = root / name
    case.mkdir(parents=True, exist_ok=True)
    full = case / "full_run"
    resumed = case / "resumed_run"
    source_root = Path(__file__).resolve().parents[1]
    for stage in (full, resumed):
        stage.mkdir(parents=True, exist_ok=True)
        prepare_stage(stage, source_root, combined=not qcpt, absolute=absolute, args=args)
    binary_full = build(full, qcpt)
    binary_resumed = build(resumed, qcpt)
    schedule = "qcpt_schedule.txt" if qcpt else "tempering_schedule.txt"
    if not qcpt:
        (full / schedule).write_text("# beta tau\n" + "".join(
            f"{beta:.17g} {beta / 2:.17g}\n" for beta, _ in fixture.PATH))
        shutil.copy2(full / schedule, resumed / schedule)
    execute(full, binary_full, schedule, "full", args, qcpt)
    execute(resumed, binary_resumed, schedule, "resumed", args, qcpt,
            checkpoint_every=args.checkpoint_every, interrupt=True)
    execute(resumed, binary_resumed, schedule, "resumed", args, qcpt, resume=True)
    differences = compare_outputs(full, resumed, "resumed")
    return {"case": name, "qcpt": qcpt, "absolute_weights": absolute,
            "differences": differences, "passed": not any(differences.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--Tsteps", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--nbins", type=int, default=20)
    parser.add_argument("--updates-per-exchange", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--beta-anneal", action="store_true")
    parser.add_argument("--anneal-interval", type=int, default=10)
    parser.add_argument("--interrupt-timeout", type=float, default=120.0)
    parser.add_argument("--oversubscribe", action="store_true")
    parser.add_argument("--absolute-weights", action="store_true")
    args = parser.parse_args()
    if args.steps < args.interval * args.nbins:
        parser.error("steps must be at least interval * nbins")
    if args.beta_anneal and not (0 < args.checkpoint_every < args.Tsteps):
        parser.error("annealed resume validation requires 0 < checkpoint-every < Tsteps")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = [one_case(root, "qcpt", args, True, args.absolute_weights),
               one_case(root, "beta", args, False, args.absolute_weights)]
    report = {"results": results, "passed": all(result["passed"] for result in results)}
    (root / "checkpoint_resume.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
