"""Require byte-identical fixed-run state after a mid-anneal resume."""

import argparse
import json
import shutil
import signal
import subprocess
import time
from pathlib import Path

import validate_qcpt as fixture


def prepare(stage, source_root, args):
    fixture.copy_sources(stage, source_root)
    shutil.copy2(source_root / "PMRQMC.cpp", stage / "PMRQMC.cpp")
    fixture.write_fixture(stage)
    fixture.write_parameters(stage / "parameters.hpp", 1.6, 0.25,
                             args.Tsteps, args.steps, args.interval,
                             args.nbins, False)
    with (stage / "parameters.hpp").open("a") as parameters:
        parameters.write("\n#define SAVE_UNFINISHED_CALCULATION\n"
                         "#define SAVE_COMPLETED_CALCULATION\n"
                         "#define RESUME_CALCULATION\n"
                         "#define HURRY_ON_SIGTERM\n")
    fixture.prepare_split(stage)
    subprocess.run(["g++", "-O1", "-std=c++11", "-o", "PMRQMC.bin", "PMRQMC.cpp"],
                   cwd=stage, check=True, stdout=subprocess.DEVNULL)


def command(args):
    return ["./PMRQMC.bin", "--target-beta", "1.6", "--target-tau", "0.8",
            "--beta-anneal", "--anneal-interval", str(args.anneal_interval)]


def run_complete(stage, args, log_name):
    with (stage / log_name).open("w") as log:
        subprocess.run(command(args), cwd=stage, check=True, stdout=log,
                       stderr=subprocess.STDOUT)


def run_interrupted(stage, args):
    with (stage / "interrupted.log").open("w") as log:
        process = subprocess.Popen(command(args), cwd=stage, stdout=log,
                                   stderr=subprocess.STDOUT)
        time.sleep(args.interrupt_delay)
        if process.poll() is not None:
            raise RuntimeError("fixed run completed before the checkpoint signal")
        process.send_signal(signal.SIGTERM)
        result = process.wait(timeout=args.timeout)
    if result != 0 or not (stage / "qmc_data.dat").is_file():
        raise RuntimeError("fixed run did not save a mid-anneal checkpoint")
    run_complete(stage, args, "resumed.log")


def normalized_checkpoint(path):
    payload = bytearray(Path(path).read_bytes())
    # Dynamic fixed checkpoints end in elapsed_seconds followed by five
    # uint64/double fields: magic, identity, beta, tau, and gamma.
    if len(payload) < 48:
        raise RuntimeError("fixed checkpoint is unexpectedly short")
    payload[-48:-40] = b"\0" * 8
    return bytes(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--Tsteps", type=int, default=1000000)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--nbins", type=int, default=20)
    parser.add_argument("--anneal-interval", type=int, default=10)
    parser.add_argument("--interrupt-delay", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if args.steps < args.interval * args.nbins:
        parser.error("steps must be at least interval * nbins")
    if args.Tsteps <= args.anneal_interval:
        parser.error("Tsteps must exceed anneal-interval")
    root = Path(args.output).resolve()
    source_root = Path(__file__).resolve().parents[1]
    full, resumed = root / "full", root / "resumed"
    for stage in (full, resumed):
        stage.mkdir(parents=True, exist_ok=True)
        prepare(stage, source_root, args)
    run_complete(full, args, "full.log")
    run_interrupted(resumed, args)
    identical = normalized_checkpoint(full / "qmc_data.dat") == normalized_checkpoint(
        resumed / "qmc_data.dat")
    report = {"passed": identical, "mid_anneal_checkpoint": True,
              "compared": "complete fixed-run checkpoint excluding elapsed_seconds"}
    (root / "fixed_anneal_resume.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if identical else 1)


if __name__ == "__main__":
    main()
