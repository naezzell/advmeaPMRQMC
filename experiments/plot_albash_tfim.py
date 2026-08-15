"""Plot the N=12 Albash-style QMC/exact verification report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def make_plot(report_path: Path, output_path: Path, tsteps: int, steps: int,
              steps_per_measurement: int, nbins: int, qmax: int) -> None:
    import matplotlib.pyplot as plt

    report = json.loads(report_path.read_text())
    rows = report["rows"]
    beta = [row["beta"] for row in rows]
    exact_energy = [row["exact"]["energy_per_site"] for row in rows]
    qmc_energy = [row["qmc"]["energy_per_site"] for row in rows]
    qmc_energy_error = [row["qmc"]["energy_error_per_site"] for row in rows]
    exact_heat = [row["exact"]["specific_heat_per_site"] for row in rows]
    qmc_heat = [row["qmc"]["specific_heat_per_site"] for row in rows]
    qmc_heat_error = [row["qmc"]["specific_heat_error_per_site"] for row in rows]
    runtimes = []
    for index in range(len(rows)):
        log = report_path.parent / f"beta_{index:02d}_{rows[index]['beta']:g}" / "qmc.log"
        match = re.search(r"wall-clock cpu time = ([^\s]+)", log.read_text())
        if match is None:
            raise RuntimeError(f"could not parse runtime from {log}")
        runtimes.append(float(match.group(1)))

    plt.style.use("ggplot")
    figure, axes = plt.subplots(1, 3, figsize=(14, 5))
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel(r"$\beta$")
    axes[0].plot(beta, exact_energy, "-", color="tab:blue", label="exact diagonalization")
    axes[0].errorbar(beta, qmc_energy, yerr=2 * np.array(qmc_energy_error), fmt="o",
                     color="tab:orange", capsize=3, label="PMR-QMC")
    axes[0].set_ylabel(r"$E/N$")
    axes[0].set_title("Energy per site")
    axes[0].legend(frameon=True, fontsize=8)

    axes[1].plot(beta, exact_heat, "-", color="tab:blue", label="exact diagonalization")
    axes[1].errorbar(beta, qmc_heat, yerr=2 * np.array(qmc_heat_error),
                     fmt="o--", color="tab:orange", capsize=3, label="PMR-QMC")
    axes[1].set_ylabel(r"$C/N$")
    axes[1].set_title("Specific heat per site")
    axes[1].legend(frameon=True, fontsize=8)
    axes[2].plot(beta, runtimes, "o-", color="tab:green")
    axes[2].set_ylabel("CPU time (s)")
    axes[2].set_title("QMC compute time")
    figure.suptitle(r"N=12 3-regular TFIM, $\Gamma=(10\beta)^{-1/2}$", y=0.98)
    caption = (f"Independent fixed-$\\beta$ PMR-QMC runs; no beta annealing. "
               f"$T_{{steps}}={tsteps}$, steps={steps}, "
               f"stepsPerMeasurement={steps_per_measurement}, Nbins={nbins}, qmax={qmax}. "
               "Markers show QMC with 2$\\sigma$ error bars; lines show exact diagonalization.")
    figure.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=9)
    figure.tight_layout(rect=[0, 0.16, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tsteps", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--steps-per-measurement", type=int, default=20)
    parser.add_argument("--nbins", type=int, default=100)
    parser.add_argument("--qmax", type=int, default=1000)
    args = parser.parse_args()
    make_plot(args.report, args.output, args.tsteps, args.steps,
              args.steps_per_measurement, args.nbins, args.qmax)


if __name__ == "__main__":
    main()
