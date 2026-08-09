"""Minimal end-to-end validity check for beta-only parallel tempering.

The test uses a two-spin transverse-field Ising Hamiltonian,

    H = -J Z_1 Z_2 - Gamma (X_1 + X_2),

and measures M_x = (X_1 + X_2)/2.  It compares every PT temperature with a
direct 4x4 diagonalization and independently compares a single-beta PMRQMC
run at the cold endpoint.  Only the Python standard library is required.
"""

import argparse
import csv
import html
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


BETAS = (0.25, 0.6, 1.3, 3.0)
J = 1.0
GAMMA = 0.7
ABSOLUTE_TOLERANCE = 0.025
SIGMA_LIMIT = 6.0


def jacobi_eigh(matrix, tolerance=1e-14):
    """Eigenvalues/vectors of a small real symmetric matrix."""
    size = len(matrix)
    a = [list(row) for row in matrix]
    vectors = [[float(i == j) for j in range(size)] for i in range(size)]
    for _ in range(100 * size * size):
        p, q = max(
            ((i, j) for i in range(size) for j in range(i + 1, size)),
            key=lambda pair: abs(a[pair[0]][pair[1]]),
        )
        if abs(a[p][q]) < tolerance:
            break
        angle = 0.5 * math.atan2(2.0 * a[p][q], a[q][q] - a[p][p])
        cosine, sine = math.cos(angle), math.sin(angle)
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        for i in range(size):
            if i in (p, q):
                continue
            aip, aiq = a[i][p], a[i][q]
            a[i][p] = a[p][i] = cosine * aip - sine * aiq
            a[i][q] = a[q][i] = sine * aip + cosine * aiq
        a[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq
        a[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq
        a[p][q] = a[q][p] = 0.0
        for i in range(size):
            vip, viq = vectors[i][p], vectors[i][q]
            vectors[i][p] = cosine * vip - sine * viq
            vectors[i][q] = sine * vip + cosine * viq
    else:
        raise RuntimeError("Jacobi diagonalization did not converge")
    return [a[i][i] for i in range(size)], vectors


def two_spin_matrices(coupling=J, gamma=GAMMA):
    hamiltonian = [[0.0] * 4 for _ in range(4)]
    observable = [[0.0] * 4 for _ in range(4)]
    for state in range(4):
        z0 = 1.0 if not state & 1 else -1.0
        z1 = 1.0 if not state & 2 else -1.0
        hamiltonian[state][state] = -coupling * z0 * z1
        for bit in (0, 1):
            target = state ^ (1 << bit)
            hamiltonian[target][state] += -gamma
            observable[target][state] += 0.5
    return hamiltonian, observable


def exact_expectation(beta):
    hamiltonian, observable = two_spin_matrices()
    energies, eigenvectors = jacobi_eigh(hamiltonian)
    minimum = min(energies)
    weights = [math.exp(-beta * (energy - minimum)) for energy in energies]
    diagonal = []
    for column in range(4):
        vector = [eigenvectors[row][column] for row in range(4)]
        diagonal.append(sum(
            vector[i] * observable[i][j] * vector[j]
            for i in range(4) for j in range(4)
        ))
    return sum(weight * value for weight, value in zip(weights, diagonal)) / sum(weights)


def write_inputs(directory, thermalization, steps, measurement_interval):
    (directory / "H.txt").write_text(
        f"{-J:.17g} 1 Z 2 Z\n"
        f"{-GAMMA:.17g} 1 X\n"
        f"{-GAMMA:.17g} 2 X\n"
    )
    (directory / "transverse_magnetization.txt").write_text("0.5 1 X\n0.5 2 X\n")
    (directory / "tempering_schedule.txt").write_text(
        "# beta tau\n" + "".join(f"{beta:.17g} {beta / 2:.17g}\n" for beta in BETAS)
    )
    (directory / "parameters.hpp").write_text(f"""
#define Tsteps {thermalization}
#define steps {steps}
#define stepsPerMeasurement {measurement_interval}
#define beta {BETAS[-1]:.17g}
#define tau {BETAS[-1] / 2:.17g}
#define parity_cond 0
#define qmax 128
#define Nbins 50
#define EXHAUSTIVE_CYCLE_SEARCH
#define GAPS_GEOMETRIC_PARAMETER 0.8
#define COMPOSITE_UPDATE_BREAK_PROBABILITY 0.9
#define EXACTLY_REPRODUCIBLE
#define RNG_SEED_OFFSET 7300
#define TFIM_GLOBAL_Z2_MOVE
""")


def run(command, directory, log_name):
    with (directory / log_name).open("w") as output:
        subprocess.run(command, cwd=directory, stdout=output,
                       stderr=subprocess.STDOUT, check=True)


def fixed_result(log_path):
    text = log_path.read_text()
    means = re.findall(r"^mean\(O\) = ([^\n]+)$", text, re.MULTILINE)
    errors = re.findall(r"^std\.dev\.\(O\) = ([^\n]+)$", text, re.MULTILINE)
    if not means or not errors:
        raise RuntimeError("could not find the custom observable in the fixed-beta output")
    return float(means[0]), float(errors[0])


def pt_results(path):
    with path.open(newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    results = {}
    for row in rows:
        if row["kind"] == "observable" and row["name"] == "A":
            results[float(row["beta"])] = (float(row["mean"]), float(row["stdev"]))
    if len(results) != len(BETAS):
        raise RuntimeError("PT output does not contain one A estimate per temperature")
    return results


def trace_rows(path, temperature=None):
    with path.open(newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if temperature is not None:
        rows = [row for row in rows if int(row["temperature"]) == temperature]
    return rows


def validate_stream_traces(output):
    fixed_paths = sorted(output.glob("fixed_stream.rank*.csv"))
    pt_paths = sorted(output.glob("pt_stream.rank*.csv"))
    if len(fixed_paths) != 1 or len(pt_paths) != len(BETAS):
        raise RuntimeError("independent stream trace count does not match MPI layout")
    required_fixed = {"stream", "rank", "sign", "signed_obs_0"}
    required_pt = required_fixed | {"ladder", "temperature", "trajectory"}
    for path, required in [(fixed_paths[0], required_fixed)] + [
            (path, required_pt) for path in pt_paths]:
        rows = trace_rows(path)
        if not rows or not required.issubset(rows[0]):
            raise RuntimeError(f"invalid independent stream schema: {path}")
    return [path.name for path in fixed_paths + pt_paths]


def running_trace(rows):
    numerator = 0.0
    denominator = 0.0
    result = []
    for row in rows:
        numerator += float(row["signed_obs_0"])
        denominator += float(row["sign"])
        result.append({
            "updates": int(row["updates"]),
            "seconds": float(row["elapsed_seconds"]),
            "mean": numerator / denominator if denominator else float("nan"),
        })
    return result


def write_convergence_csv(path, fixed, pt, exact):
    if len(fixed) != len(pt):
        raise RuntimeError("fixed and PT traces have different lengths")
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("measurement", "updates_per_replica", "fixed_seconds",
                         "fixed_running_mean", "pt_seconds", "pt_running_mean", "exact"))
        for index, (fixed_row, pt_row) in enumerate(zip(fixed, pt), start=1):
            writer.writerow((index, fixed_row["updates"], fixed_row["seconds"],
                             fixed_row["mean"], pt_row["seconds"], pt_row["mean"], exact))


def _polyline(series, x_value, x_min, x_max, y_min, y_max, left, top, width, height):
    stride = max(1, len(series) // 1500)
    sampled = series[::stride]
    if sampled[-1] is not series[-1]:
        sampled.append(series[-1])
    points = []
    for row in sampled:
        x = left + width * (x_value(row) - x_min) / (x_max - x_min)
        y = top + height * (y_max - row["mean"]) / (y_max - y_min)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def write_svg(path, fixed, pt, exact, x_key, x_label):
    width, height = 900, 560
    left, top, plot_width, plot_height = 85, 45, 780, 430
    x_value = lambda row: row[x_key]
    x_min = min(x_value(fixed[0]), x_value(pt[0]))
    x_max = max(x_value(fixed[-1]), x_value(pt[-1]))
    values = [row["mean"] for row in fixed + pt if math.isfinite(row["mean"])] + [exact]
    y_min, y_max = min(values), max(values)
    padding = max(0.02, 0.08 * (y_max - y_min))
    y_min -= padding
    y_max += padding
    tolerance = ABSOLUTE_TOLERANCE

    def sx(value):
        return left + plot_width * (value - x_min) / (x_max - x_min)

    def sy(value):
        return top + plot_height * (y_max - value) / (y_max - y_min)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<rect x="{left}" y="{sy(exact + tolerance):.2f}" width="{plot_width}" '
        f'height="{sy(exact - tolerance) - sy(exact + tolerance):.2f}" fill="#dddddd" opacity="0.65"/>',
    ]
    for tick in range(6):
        value = x_min + tick * (x_max - x_min) / 5.0
        x = sx(value)
        label = f"{value:.2f}" if x_key == "seconds" else f"{value:.0f}"
        elements += [
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#eeeeee"/>',
            f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" font-size="13">{label}</text>',
        ]
    for tick in range(6):
        value = y_min + tick * (y_max - y_min) / 5.0
        y = sy(value)
        elements += [
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#eeeeee"/>',
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="13">{value:.3f}</text>',
        ]
    elements += [
        f'<line x1="{left}" y1="{sy(exact):.2f}" x2="{left + plot_width}" y2="{sy(exact):.2f}" '
        'stroke="black" stroke-width="2" stroke-dasharray="7,5"/>',
        f'<polyline points="{_polyline(fixed, x_value, x_min, x_max, y_min, y_max, left, top, plot_width, plot_height)}" '
        'fill="none" stroke="#1f77b4" stroke-width="2"/>',
        f'<polyline points="{_polyline(pt, x_value, x_min, x_max, y_min, y_max, left, top, plot_width, plot_height)}" '
        'fill="none" stroke="#ff7f0e" stroke-width="2"/>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="black"/>',
        f'<text x="{width / 2}" y="27" text-anchor="middle" font-size="20">Two-spin cold-beta convergence</text>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="15">{html.escape(x_label)}</text>',
        f'<text x="20" y="{top + plot_height / 2}" text-anchor="middle" font-size="15" '
        f'transform="rotate(-90 20 {top + plot_height / 2})">running transverse magnetization</text>',
        f'<line x1="{left + 20}" y1="{top + 20}" x2="{left + 55}" y2="{top + 20}" stroke="#1f77b4" stroke-width="3"/>',
        f'<text x="{left + 63}" y="{top + 25}" font-size="14">fixed beta</text>',
        f'<line x1="{left + 160}" y1="{top + 20}" x2="{left + 195}" y2="{top + 20}" stroke="#ff7f0e" stroke-width="3"/>',
        f'<text x="{left + 203}" y="{top + 25}" font-size="14">PT cold slot</text>',
        f'<line x1="{left + 325}" y1="{top + 20}" x2="{left + 360}" y2="{top + 20}" stroke="black" '
        f'stroke-width="2" stroke-dasharray="7,5"/>',
        f'<text x="{left + 368}" y="{top + 25}" font-size="14">exact</text>',
        '</svg>',
    ]
    path.write_text("\n".join(elements) + "\n")


def optionally_render_png(svg_paths):
    converter = shutil.which("rsvg-convert")
    rendered = []
    if converter:
        for svg_path in svg_paths:
            png_path = svg_path.with_suffix(".png")
            subprocess.run([converter, str(svg_path), "-o", str(png_path)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            rendered.append(png_path.name)
    return rendered


def comparison(beta, method, mean, standard_error):
    exact = exact_expectation(beta)
    error = abs(mean - exact)
    tolerance = max(ABSOLUTE_TOLERANCE, SIGMA_LIMIT * standard_error)
    return {
        "method": method,
        "beta": beta,
        "exact": exact,
        "mean": mean,
        "standard_error": standard_error,
        "absolute_error": error,
        "tolerance": tolerance,
        "z_score": error / standard_error if standard_error > 0.0 else None,
        "passed": error <= tolerance,
    }


def print_report(comparisons, swaps, round_trips, output):
    print("method       beta       exact         QMC          SE      |error|   pass")
    for item in comparisons:
        print(f"{item['method']:<12} {item['beta']:>4.2f}  {item['exact']:>10.6f}  "
              f"{item['mean']:>10.6f}  {item['standard_error']:>9.6f}  "
              f"{item['absolute_error']:>9.6f}   {'yes' if item['passed'] else 'NO'}")
    print(f"PT swap acceptance: {', '.join(f'{value:.3f}' for value in swaps)}")
    print(f"PT round trips: {round_trips}")
    print(f"Artifacts: {output}")


def validate(output, source_root, thermalization, steps, measurement_interval,
             updates_per_exchange, oversubscribe, beta_anneal=False,
             anneal_interval=None):
    output.mkdir(parents=True, exist_ok=True)
    for name in ("prepare.cpp", "PMRQMC_mpi.cpp", "PMRQMC_pt_mpi.cpp", "mainqmc.hpp",
                 "divdiff.hpp", "pt_schedule.hpp", "beta_anneal.hpp"):
        shutil.copy2(source_root / name, output / name)
    write_inputs(output, thermalization, steps, measurement_interval)

    run(["g++", "-O1", "-std=c++11", "-o", "prepare.bin", "prepare.cpp"],
        output, "compile_prepare.log")
    run(["./prepare.bin", "H.txt", "transverse_magnetization.txt"],
        output, "prepare.log")
    run(["mpicxx", "-O1", "-std=c++11", "-o", "PMRQMC_mpi.bin", "PMRQMC_mpi.cpp"],
        output, "compile_fixed.log")
    run(["mpicxx", "-O1", "-std=c++11", "-o", "PMRQMC_pt_mpi.bin", "PMRQMC_pt_mpi.cpp"],
        output, "compile_pt.log")
    mpi_command = ["mpirun"]
    if oversubscribe:
        mpi_command.append("--oversubscribe")
    anneal_args = (["--beta-anneal"] if beta_anneal else [])
    if beta_anneal and anneal_interval is not None:
        anneal_args += ["--anneal-interval", str(anneal_interval)]
    run(mpi_command + ["-n", "1", "./PMRQMC_mpi.bin", "--timeseries-prefix", "fixed_trace.csv",
                       "--stream-timeseries-prefix", "fixed_stream"] + anneal_args,
        output, "fixed_beta.log")
    mpi_command += ["-n", str(len(BETAS)), "./PMRQMC_pt_mpi.bin",
                    "--schedule", "tempering_schedule.txt",
                    "--updates-per-exchange", str(updates_per_exchange),
                    "--independent-ladders", "1", "--output-prefix", "pt",
                    "--timeseries-prefix", "pt_trace.csv",
                    "--stream-timeseries-prefix", "pt_stream"] + anneal_args
    run(mpi_command, output, "pt.log")

    fixed_mean, fixed_error = fixed_result(output / "fixed_beta.log")
    pt = pt_results(output / "pt_observables.csv")
    fixed_trace = running_trace(trace_rows(output / "fixed_trace.csv"))
    cold_trace = running_trace(trace_rows(output / "pt_trace.csv", len(BETAS) - 1))
    cold_exact = exact_expectation(BETAS[-1])
    write_convergence_csv(output / "convergence.csv", fixed_trace, cold_trace, cold_exact)
    update_plot = output / "convergence_updates.svg"
    time_plot = output / "convergence_time.svg"
    write_svg(update_plot, fixed_trace, cold_trace, cold_exact,
              "updates", "QMC updates per replica")
    write_svg(time_plot, fixed_trace, cold_trace, cold_exact,
              "seconds", "elapsed wall time (seconds)")
    png_plots = optionally_render_png((update_plot, time_plot))
    comparisons = [comparison(BETAS[-1], "fixed", fixed_mean, fixed_error)]
    comparisons += [comparison(beta, "PT", *pt[beta]) for beta in BETAS]
    with (output / "pt_swaps.csv").open(newline="") as input_file:
        swaps = [float(row["acceptance"]) for row in csv.DictReader(input_file)]
    with (output / "pt_flow.csv").open(newline="") as input_file:
        flow = list(csv.DictReader(input_file))
    stream_trace_files = validate_stream_traces(output)
    round_trips = int(flow[0]["round_trips"]) if flow else 0
    passed = all(item["passed"] for item in comparisons) and round_trips > 0
    report = {
        "model": "two_spin_transverse_field_ising",
        "hamiltonian": {"J": J, "Gamma": GAMMA},
        "observable": "(X1 + X2) / 2",
        "thermalization_updates": thermalization,
        "beta_anneal": beta_anneal,
        "anneal_interval": anneal_interval,
        "sampling_updates": steps,
        "updates_per_exchange": updates_per_exchange,
        "acceptance": swaps,
        "round_trips": round_trips,
        "stream_trace_files": stream_trace_files,
        "comparisons": comparisons,
        "convergence_files": {
            "data": "convergence.csv",
            "updates_plot": "convergence_updates.svg",
            "time_plot": "convergence_time.svg",
            "png_plots": png_plots,
        },
        "passed": passed,
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print_report(comparisons, swaps, round_trips, output)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", help="artifact directory (default: a new /tmp directory)")
    parser.add_argument("--Tsteps", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--steps-per-measurement", type=int, default=10)
    parser.add_argument("--updates-per-exchange", type=int, default=1)
    parser.add_argument("--beta-anneal", action="store_true")
    parser.add_argument("--anneal-interval", type=int)
    parser.add_argument("--oversubscribe", action="store_true")
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve() if args.output else Path(
        tempfile.mkdtemp(prefix="pmrqmc_pt_validation_"))
    passed = validate(output, source_root, args.Tsteps, args.steps,
                      args.steps_per_measurement, args.updates_per_exchange,
                      args.oversubscribe, args.beta_anneal, args.anneal_interval)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
