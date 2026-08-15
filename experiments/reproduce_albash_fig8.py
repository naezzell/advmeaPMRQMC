"""Reproduce a local, Figure-8-style CPT/QPT/QCPT plot.

This harness deliberately separates the documented Albash settings from the
undocumented choices (instance seed, ladder, and path endpoints).  It runs the
repository's QCPT executable and writes machine-readable schedules, manifests,
timeseries, and a phase-plane plot.  A PIQMC reference may be supplied as a
CSV; this repository does not contain a PIQMC implementation.

Examples:
    python3 experiments/reproduce_albash_fig8.py plan --output /tmp/fig8-plan
    python3 experiments/reproduce_albash_fig8.py run --output /tmp/fig8 --oversubscribe
    python3 experiments/reproduce_albash_fig8.py plot --output /tmp/fig8
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

try:
    from experiments.paper_tfim_3regular import write_instance
except ModuleNotFoundError:  # direct ``python experiments/reproduce_...py``
    from paper_tfim_3regular import write_instance


N = 60
INSTANCE_ID = "fixture_60"
INSTANCE_SEED = 1060
# Figure 8 is plotted in the beta range 0..1.  beta=0 is not an admissible
# QMC schedule value, so use the positive logarithmic interval 0.1..1 as a
# finite-temperature proxy for the lower edge of the displayed plot.
DEFAULT_BETAS = tuple(10 ** (x / 8) for x in range(-8, 1))  # 0.1 ... 1
DEFAULT_GAMMAS = tuple(10 ** (x / 8) for x in range(-8, 1))  # 0.1 ... 1
SOURCE_FILES = (
    "prepare.cpp", "PMRQMC_qcpt_mpi.cpp", "PMRQMC_pt_mpi.cpp", "mainqmc.hpp", "divdiff.hpp",
    "pt_schedule.hpp", "beta_anneal.hpp",
)


def qcpt_paths(betas=DEFAULT_BETAS, gammas=DEFAULT_GAMMAS):
    if len(betas) != len(gammas):
        raise ValueError("beta and gamma ladders must have equal length")
    return {
        "cpt": [(beta, 0.0) for beta in betas],
        "qpt": [(max(betas), gamma) for gamma in reversed(gammas)],
        "qcpt": [(beta, (10.0 * beta) ** -0.5) for beta in betas],
    }


def write_parameters(path: Path, *, tsteps: int, steps: int,
                     interval: int, nbins: int, qmax: int, seed: int) -> None:
    path.write_text(f"""#define Tsteps {tsteps}
#define steps {steps}
#define stepsPerMeasurement {interval}
#define beta 1.0
#define tau 0.5
#define gamma 0.1
#define parity_cond 0
#define qmax {qmax}
#define Nbins {nbins}
#define EXHAUSTIVE_CYCLE_SEARCH
#define GAPS_GEOMETRIC_PARAMETER 0.8
#define COMPOSITE_UPDATE_BREAK_PROBABILITY 0.9
#define EXACTLY_REPRODUCIBLE
#define RNG_SEED_OFFSET {seed}
#define MEASURE_H
#define MEASURE_H2
#define MEASURE_HOFFDIAG
""")


def run_checked(command: list[str], cwd: Path, log_name: str) -> None:
    with (cwd / log_name).open("w") as log:
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def write_schedule(path: Path, points: list[tuple[float, float]]) -> None:
    path.write_text("# beta gamma tau\n" + "".join(
        f"{beta:.17g} {gamma:.17g} {beta / 2.0:.17g}\n"
        for beta, gamma in points
    ))


def make_manifest(args, paths):
    return {
        "model": "paper_tfim_3regular",
        "instance_id": INSTANCE_ID,
        "n": N,
        "instance_seed": INSTANCE_SEED,
        "paths": paths,
        "documented_paper_parameters": {
            "qcpt_swap_sweeps": args.tsteps,
            "local_qmc_sweeps_per_exchange": args.updates_per_exchange,
            "measurements": args.steps // args.interval,
            "sweeps_between_measurements": args.interval,
            "piqmc_sweeps": 1000000,
            "piqmc_trotter_slices": 5120,
        },
        "implementation_parameters": {
            "tsteps": args.tsteps,
            "steps": args.steps,
            "steps_per_measurement": args.interval,
            "nbins": args.nbins,
            "qmax": args.qmax,
            "independent_ladders": args.ladders,
            "compiler": "mpicxx -O2 -std=c++11 -DPMR_QCPT",
        },
        "host": platform.node(),
        "platform": platform.platform(),
    }


def prepare_common(root: Path, source_root: Path, args) -> None:
    instance = root / "instance"
    if not instance.exists():
        write_instance(instance, N, INSTANCE_SEED)
    for name in SOURCE_FILES:
        shutil.copy2(source_root / name, root / name)
    shutil.copy2(instance / "H_fixed.txt", root / "H_fixed.txt")
    shutil.copy2(instance / "H_gamma.txt", root / "H_gamma.txt")
    shutil.copy2(instance / "instance.json", root / "instance.json")
    write_parameters(root / "parameters.hpp", tsteps=args.tsteps, steps=args.steps,
                     interval=args.interval, nbins=args.nbins, qmax=args.qmax,
                     seed=args.qmc_seed)
    run_checked(["g++", "-O2", "-std=c++11", "-o", "prepare.bin", "prepare.cpp"], root, "prepare_build.log")
    run_checked(["./prepare.bin", "--hamiltonian-fixed", "H_fixed.txt",
                 "--hamiltonian-gamma", "H_gamma.txt"],
                root, "prepare.log")


def run_path(root: Path, points: list[tuple[float, float]], args) -> float:
    write_schedule(root / "schedule.txt", points)
    command = ["mpirun"]
    if args.oversubscribe:
        command.append("--oversubscribe")
    command += ["-n", str(len(points) * args.ladders), "./PMRQMC_qcpt_mpi.bin",
                "--schedule", "schedule.txt", "--updates-per-exchange",
                str(args.updates_per_exchange), "--independent-ladders", str(args.ladders),
                "--output-prefix", "pmr", "--timeseries-prefix", "timeseries.csv"]
    started = time.time()
    run_checked(command, root, "run.log")
    return time.time() - started


def read_timeseries(path: Path, path_name: str):
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            slot = int(row["slot"])
            sign = float(row["sign"])
            # With no custom observable files, the generated observable layout
            # is H=obs_0, H2=obs_1, and Hoffdiag=obs_4.  For Gamma>0,
            # Hoffdiag=-Gamma*N*m_x, so m_x is obtained in O(1) time.
            energy = float(row["obs_0"])
            energy2 = float(row["obs_1"])
            hoffdiag = float(row["obs_4"])
            gamma = float(row["gamma"])
            mx = -hoffdiag / (N * gamma) if gamma > 0.0 else 0.0
            rows.append({
                "path": path_name, "slot": slot,
                "measurement": int(row["measurement"]),
                "updates": int(row["updates"]),
                "beta": float(row["beta"]), "gamma": gamma,
                "mx": mx, "energy": energy / N, "energy2": energy2 / (N * N),
                "sign": sign,
                "elapsed_seconds": float(row["elapsed_seconds"]),
            })
    # The executable emits one row per slot per measurement.  Aggregate with
    # sign reweighting; this also remains valid for the repository's signed
    # estimator format.
    result = {}
    for row in rows:
        key = row["slot"]
        acc = result.setdefault(key, {"row": row, "den": 0.0, "num_mx": 0.0,
                                       "num_e": 0.0, "num_e2": 0.0,
                                       "first_measurement": row["measurement"],
                                       "last_measurement": row["measurement"],
                                       "first_update": row["updates"],
                                       "last_update": row["updates"],
                                       "first_elapsed": row["elapsed_seconds"],
                                       "last_elapsed": row["elapsed_seconds"]})
        acc["den"] += row["sign"]
        acc["num_mx"] += row["mx"] * row["sign"]
        acc["num_e"] += row["energy"] * row["sign"]
        acc["num_e2"] += row["energy2"] * row["sign"]
        acc["last_measurement"] = row["measurement"]
        acc["last_update"] = row["updates"]
        acc["last_elapsed"] = row["elapsed_seconds"]
    output = []
    for acc in result.values():
        if acc["den"] == 0:
            continue
        row = dict(acc["row"])
        row.update({"mx": acc["num_mx"] / acc["den"],
                    "energy": acc["num_e"] / acc["den"],
                    "energy2": acc["num_e2"] / acc["den"]})
        # These timestamps describe the shared QCPT process.  All replicas
        # run concurrently, so this is a wall-time window for the point's
        # samples, not an isolated per-replica CPU-time measurement.
        row["first_measurement"] = acc["first_measurement"]
        row["last_measurement"] = acc["last_measurement"]
        row["sample_count"] = acc["last_measurement"] - acc["first_measurement"] + 1
        row["first_update"] = acc["first_update"]
        row["last_update"] = acc["last_update"]
        row["update_span"] = acc["last_update"] - acc["first_update"]
        row["first_measurement_elapsed_seconds"] = acc["first_elapsed"]
        row["last_measurement_elapsed_seconds"] = acc["last_elapsed"]
        row["measurement_window_seconds"] = (
            row["last_measurement_elapsed_seconds"] -
            row["first_measurement_elapsed_seconds"]
        )
        row["specific_heat"] = row["beta"] ** 2 * N * (row["energy2"] - row["energy"] ** 2)
        output.append(row)
    timing = []
    for row in output:
        timing.append({key: row[key] for key in (
            "path", "slot", "beta", "gamma", "first_measurement", "last_measurement",
            "sample_count", "first_update", "last_update", "update_span",
            "first_measurement_elapsed_seconds", "last_measurement_elapsed_seconds",
            "measurement_window_seconds")})
    return output, timing


def run_campaign(args) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1]
    paths = qcpt_paths()
    if args.only_path:
        paths = {args.only_path: paths[args.only_path]}
    manifest = make_manifest(args, paths)
    manifest["artifact_contract"] = {
        "raw_timeseries": "<curve>/timeseries.csv",
        "raw_summary": "<curve>/pmr_observables.csv",
        "raw_swaps": "<curve>/pmr_swaps.csv",
        "raw_flow": "<curve>/pmr_flow.csv",
        "point_timing": "point_timing.csv",
        "curve_timing": "curve_timing.json",
        "rank_timing": "<curve>/pmr_rank_timing.csv",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    all_rows = []
    all_timing = []
    curve_timing = []
    for name, points in paths.items():
        path_root = output / name
        path_root.mkdir(exist_ok=True)
        prepare_started = time.perf_counter()
        prepare_common(path_root, source_root, args)
        prepare_seconds = time.perf_counter() - prepare_started
        compile_started = time.perf_counter()
        run_checked(["mpicxx", "-O2", "-std=c++11", "-DPMR_QCPT", "-o",
                     "PMRQMC_qcpt_mpi.bin", "PMRQMC_qcpt_mpi.cpp"], path_root, "compile.log")
        compile_seconds = time.perf_counter() - compile_started
        path_wall_seconds = run_path(path_root, points, args)
        rows, timing = read_timeseries(path_root / "timeseries.csv", name)
        for row in rows:
            row["path_wall_seconds"] = path_wall_seconds
        for row in timing:
            row["path_wall_seconds"] = path_wall_seconds
        all_timing.extend(timing)
        all_rows.extend(rows)
        curve_record = {
            "path": name, "replica_points": len(points),
            "mpi_ranks": len(points) * args.ladders,
            "prepare_seconds": prepare_seconds,
            "compile_seconds": compile_seconds,
            "mpi_wall_seconds": path_wall_seconds,
            "total_curve_seconds": prepare_seconds + compile_seconds + path_wall_seconds,
            "raw_files": sorted(path.name for path in path_root.iterdir() if path.is_file()),
        }
        curve_timing.append(curve_record)
        (path_root / "curve_timing.json").write_text(json.dumps(curve_record, indent=2) + "\n")
    with (output / "estimates.csv").open("w", newline="") as stream:
        fields = ["path", "slot", "measurement", "updates", "beta", "gamma", "mx", "energy", "energy2", "specific_heat",
                  "sign", "first_measurement", "last_measurement", "sample_count",
                  "first_update", "last_update", "update_span",
                  "elapsed_seconds", "first_measurement_elapsed_seconds",
                  "last_measurement_elapsed_seconds", "measurement_window_seconds",
                  "path_wall_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(all_rows)
    timing_fields = ["path", "slot", "beta", "gamma", "first_measurement", "last_measurement",
                     "sample_count", "first_update", "last_update", "update_span",
                     "first_measurement_elapsed_seconds", "last_measurement_elapsed_seconds",
                     "measurement_window_seconds", "path_wall_seconds"]
    with (output / "point_timing.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=timing_fields)
        writer.writeheader(); writer.writerows(all_timing)
    (output / "curve_timing.json").write_text(json.dumps(curve_timing, indent=2) + "\n")
    manifest["completed_curve_timing"] = curve_timing
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {output / 'estimates.csv'}")
    print(f"wrote {output / 'point_timing.csv'}")
    print(f"wrote {output / 'curve_timing.json'}")


def plot_campaign(args) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("matplotlib is required for plotting") from error
    rows = list(csv.DictReader((args.output / "estimates.csv").open(newline="")))
    if not rows:
        raise SystemExit("estimates.csv is empty")
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; registers 3-D projection
    figure = plt.figure(figsize=(18, 6))
    axis3d = figure.add_subplot(131, projection="3d")
    heat_axis = figure.add_subplot(132, projection="3d")
    timing_axis = figure.add_subplot(133)
    colors = {"cpt": "tab:red", "qpt": "tab:green", "qcpt": "tab:blue"}
    labels = {"cpt": "CPT", "qpt": "QPT", "qcpt": r"QCPT: $\Gamma=(10\beta)^{-1/2}$"}
    for name in ("cpt", "qpt", "qcpt"):
        selected = [r for r in rows if r["path"] == name]
        x = [float(r["beta"]) for r in selected]
        y = [float(r["gamma"]) for r in selected]
        c = [float(r["mx"]) for r in selected]
        heat = [float(r["specific_heat"]) for r in selected]
        axis3d.plot(x, y, c, "o-", color=colors[name], label=labels[name])
        axis3d.scatter(x, y, c, c=c, cmap="viridis", edgecolors="black", s=38)
        heat_axis.plot(x, y, heat, "o-", color=colors[name], label=labels[name])
        heat_axis.scatter(x, y, heat, c=heat, cmap="plasma", edgecolors="black", s=38)
        timing_axis.plot(x, [float(r["measurement_window_seconds"]) for r in selected],
                         "o-", color=colors[name], label=labels[name])
    if args.reference_csv:
        reference = list(csv.DictReader(args.reference_csv.open(newline="")))
        if not {"beta", "gamma", "mx"}.issubset(reference[0] if reference else {}):
            raise SystemExit("reference CSV must contain beta,gamma,mx columns")
        rx = [float(row["beta"]) for row in reference]
        ry = [float(row["gamma"]) for row in reference]
        rm = [float(row["mx"]) for row in reference]
        axis3d.scatter(rx, ry, rm, marker="x", color="black", label="reference")
        if "specific_heat" in reference[0]:
            rc = [float(row["specific_heat"]) for row in reference]
            heat_axis.scatter(rx, ry, rc, marker="x", color="black", label="reference")
    axis3d.set_xlabel(r"$\beta$")
    axis3d.set_ylabel(r"$\Gamma$")
    axis3d.set_zlabel(r"$m_x$")
    axis3d.set_title("3-D transverse magnetization")
    heat_axis.set_xlabel(r"$\beta$")
    heat_axis.set_ylabel(r"$\Gamma$")
    heat_axis.set_zlabel(r"$C/N$")
    heat_axis.set_title("3-D specific heat")
    timing_axis.set_xscale("log")
    timing_axis.set_xlabel(r"$\beta$")
    timing_axis.set_ylabel("measurement-window wall time (s)")
    timing_axis.set_title("Point-generation timing")
    timing_axis.legend(fontsize=8)
    figure.suptitle("Albash Figure 8-style N=60 3-regular MAX2SAT")
    figure.tight_layout()
    figure.savefig(args.output / "albash_fig8_style.png", dpi=180)
    print(f"wrote {args.output / 'albash_fig8_style.png'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "plot"))
    parser.add_argument("--output", type=Path, default=Path("run_outputs/albash_fig8"))
    parser.add_argument("--tsteps", type=int, default=100000)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--updates-per-exchange", type=int, default=10)
    parser.add_argument("--ladders", type=int, default=1)
    parser.add_argument("--nbins", type=int, default=100)
    parser.add_argument("--qmax", type=int, default=2000)
    parser.add_argument("--qmc-seed", type=int, default=8060)
    parser.add_argument("--reference-csv", type=Path,
                        help="optional PIQMC or exact-reference CSV with beta,gamma,mx columns")
    parser.add_argument("--only-path", choices=("cpt", "qpt", "qcpt"),
                        help="run only one curve; useful for smoke tests")
    parser.add_argument("--oversubscribe", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        args.output.mkdir(parents=True, exist_ok=True)
        paths = qcpt_paths()
        (args.output / "schedules.json").write_text(json.dumps(paths, indent=2) + "\n")
        print(json.dumps({"n": N, "instance_seed": INSTANCE_SEED, "paths": paths}, indent=2))
    elif args.command == "run":
        run_campaign(args)
    else:
        plot_campaign(args)


if __name__ == "__main__":
    main()
