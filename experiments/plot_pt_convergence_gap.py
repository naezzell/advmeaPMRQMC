"""Plot the convergence and tempering diagnostics from pt_convergence_gap.py.

Usage:
    python3 experiments/plot_pt_convergence_gap.py /tmp/pt_gap

The script writes convergence.png, wall_time.png, and pt_diagnostics.png into
the benchmark directory unless --output-dir is supplied.
"""

import argparse
import csv
import json
from pathlib import Path


def read_rows(path):
    with Path(path).open(newline="") as input_file:
        return list(csv.DictReader(input_file))


def running_mean(rows, value_column):
    """Return the cumulative sign-reweighted ratio for an observable trace."""
    signed_column = f"signed_{value_column}"
    if rows and signed_column not in rows[0]:
        raise ValueError(
            f"trace is missing {signed_column}; regenerate it with the corrected "
            "signed-numerator time-series format"
        )
    updates = []
    means = []
    elapsed = []
    numerator = 0.0
    denominator = 0.0
    for index, row in enumerate(rows, start=1):
        numerator += float(row[signed_column])
        denominator += float(row["sign"])
        updates.append(int(row["updates"]))
        means.append(numerator / denominator if denominator else float("nan"))
        elapsed.append(float(row["elapsed_seconds"]) if "elapsed_seconds" in row else None)
    return updates, means, elapsed


def load_traces(root, value_column, cold_temperature):
    fixed_rows = read_rows(root / "fixed_beta" / "trace.csv")
    pt_rows = [row for row in read_rows(root / "beta_tempered" / "trace.csv")
               if int(row["temperature"]) == cold_temperature]
    if not fixed_rows or not pt_rows:
        raise ValueError("fixed-beta or cold-slot trace is empty")
    fixed_updates, fixed_means, fixed_elapsed = running_mean(fixed_rows, value_column)
    pt_updates, pt_means, pt_elapsed = running_mean(pt_rows, value_column)
    return (fixed_updates, fixed_means, fixed_elapsed), (pt_updates, pt_means, pt_elapsed)


def wall_time_or_approximation(updates, elapsed, total_wall_seconds):
    if elapsed and all(value is not None for value in elapsed):
        return elapsed, False
    last_update = updates[-1]
    return [total_wall_seconds * update / last_update for update in updates], True


def make_plots(root, output_dir, value_column, show, reference_mode):
    # Import matplotlib only when plotting is requested, so --help remains
    # usable in environments where the optional plotting dependency is absent.
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("matplotlib is required: python3 -m pip install matplotlib") from error

    report_path = root / "comparison_report.json"
    report = json.loads(report_path.read_text())
    cold_temperature = len(report["betas"]) - 1
    exact_path = root / "exact_reference.json"
    use_exact = reference_mode == "exact" or (reference_mode == "auto" and exact_path.exists())
    if use_exact:
        if not exact_path.exists():
            raise SystemExit("exact_reference.json not found; run exact_diagonalization.py first")
        exact = json.loads(exact_path.read_text())
        reference = float(exact["cold_exact_mean"])
        reference_label = "exact reference"
    else:
        reference = float(report["reference"].get("value", report["reference"]["cold_pt_tail_mean"]))
        reference_label = "reference"
    tolerance = float(report["reference"]["tolerance"])
    (fixed_updates, fixed_means, fixed_elapsed), (pt_updates, pt_means, pt_elapsed) = load_traces(
        root, value_column, cold_temperature
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    convergence_path = output_dir / "convergence.png"
    figure, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(fixed_updates, fixed_means, label="fixed beta", linewidth=1.5)
    axes[0].plot(pt_updates, pt_means, label="beta PT, cold slot", linewidth=1.5)
    axes[0].axhline(reference, color="black", linestyle="--", label=reference_label)
    axes[0].axhspan(reference - tolerance, reference + tolerance,
                    color="gray", alpha=0.2, label="tolerance band")
    axes[0].set_ylabel(value_column)
    axes[0].set_title("Running observable estimate")
    axes[0].legend()

    fixed_error = [abs(value - reference) for value in fixed_means]
    pt_error = [abs(value - reference) for value in pt_means]
    axes[1].plot(fixed_updates, fixed_error, label="fixed beta", linewidth=1.5)
    axes[1].plot(pt_updates, pt_error, label="beta PT, cold slot", linewidth=1.5)
    axes[1].axhline(tolerance, color="black", linestyle="--", label="tolerance")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("QMC updates")
    axes[1].set_ylabel("absolute error")
    axes[1].set_title("Distance from reference")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(convergence_path, dpi=160)
    plt.close(figure)

    fixed_wall, fixed_approx = wall_time_or_approximation(
        fixed_updates, fixed_elapsed, float(report["fixed_beta"]["wall_seconds"])
    )
    pt_wall, pt_approx = wall_time_or_approximation(
        pt_updates, pt_elapsed, float(report["beta_tempered_cold_slot"]["wall_seconds"])
    )
    wall_time_path = output_dir / "wall_time.png"
    figure, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(fixed_wall, fixed_means, label="fixed beta", linewidth=1.5)
    axes[0].plot(pt_wall, pt_means, label="beta PT, cold slot", linewidth=1.5)
    axes[0].axhline(reference, color="black", linestyle="--", label=reference_label)
    axes[0].axhspan(reference - tolerance, reference + tolerance,
                    color="gray", alpha=0.2, label="tolerance band")
    axes[0].set_ylabel(value_column)
    axes[0].set_title("Running observable estimate versus wall time")
    axes[0].legend()

    axes[1].plot(fixed_wall, fixed_error, label="fixed beta", linewidth=1.5)
    axes[1].plot(pt_wall, pt_error, label="beta PT, cold slot", linewidth=1.5)
    axes[1].axhline(tolerance, color="black", linestyle="--", label="tolerance")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("wall time (seconds)")
    axes[1].set_ylabel("absolute error")
    axes[1].set_title("Distance from reference versus wall time")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(wall_time_path, dpi=160)
    plt.close(figure)

    diagnostics_path = output_dir / "pt_diagnostics.png"
    swaps_path = root / "beta_tempered" / "pt_swaps.csv"
    flow_path = root / "beta_tempered" / "pt_flow.csv"
    if swaps_path.exists() and flow_path.exists():
        swaps = read_rows(swaps_path)
        flow = read_rows(flow_path)
        edges = [int(row["edge"]) for row in swaps]
        acceptance = [float(row["acceptance"]) for row in swaps]
        trajectories = sorted({int(row["trajectory"]) for row in flow})
        temperatures = sorted({int(row["temperature"]) for row in flow})
        flow_lookup = {(int(row["trajectory"]), int(row["temperature"])): int(row["visits"])
                       for row in flow}
        flow_matrix = []
        for trajectory in trajectories:
            row = [flow_lookup.get((trajectory, temperature), 0) for temperature in temperatures]
            total = sum(row)
            flow_matrix.append([value / total if total else 0.0 for value in row])
        round_trips = int(float(flow[0]["round_trips"])) if flow else 0

        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(edges, acceptance, marker="o")
        axes[0].set_xlabel("temperature edge")
        axes[0].set_ylabel("swap acceptance")
        axes[0].set_ylim(0, 1.05)
        axes[0].set_title("PT swap acceptance")

        image = axes[1].imshow(flow_matrix, aspect="auto", origin="lower", vmin=0.0)
        axes[1].set_xlabel("temperature index")
        axes[1].set_ylabel("trajectory origin index")
        axes[1].set_xticks(range(len(temperatures)), temperatures)
        axes[1].set_yticks(range(len(trajectories)), trajectories)
        axes[1].set_title(f"Normalized trajectory flow ({round_trips} round trips)")
        figure.colorbar(image, ax=axes[1], label="fraction of visits")
        figure.tight_layout()
        figure.savefig(diagnostics_path, dpi=160)
        plt.close(figure)

    print(f"wrote {convergence_path}")
    print(f"wrote {wall_time_path}" + (" (timestamps approximated)" if fixed_approx or pt_approx else ""))
    if diagnostics_path.exists():
        print(f"wrote {diagnostics_path}")
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_directory")
    parser.add_argument("--output-dir", help="directory for generated PNG files")
    parser.add_argument("--observable-column", default="obs_0",
                        help="trace column to plot (default: obs_0)")
    parser.add_argument("--reference", choices=("auto", "pt-tail", "exact"), default="auto",
                        help="reference source; auto uses exact_reference.json when present")
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()
    root = Path(args.benchmark_directory).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root
    make_plots(root, output_dir, args.observable_column, args.show, args.reference)


if __name__ == "__main__":
    main()
