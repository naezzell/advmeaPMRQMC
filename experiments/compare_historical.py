#!/usr/bin/env python3
"""Compare campaign estimates with archived paper CSV rows without retuning tolerances."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


THREE_BY_THREE_OBSERVABLES = {
    "energy": ("energy", "h", "h_std"),
    "specific_heat": ("specific_heat", "Cv", "Cv_std"),
    "offdiagonal_energy_susceptibility": (
        "energy_susceptibility", "offDiagES", "offDiagES_std"),
    "offdiagonal_fidelity_susceptibility": (
        "fidelity_susceptibility", "offDiagFS", "offDiagFS_std"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def select_archive_row(path: Path, length: int, coupling: float, beta: float,
                       profile: str = "three_by_three") -> dict:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if profile == "three_by_three":
        matches = [row for row in rows
                   if int(float(row["L"])) == length and close(float(row["lam"]), coupling)]
    elif profile == "paper1_four_by_four":
        matches = [row for row in rows if close(float(row["beta"]), beta)]
    else:
        raise ValueError(f"unknown archive profile: {profile}")
    if len(matches) != 1:
        raise ValueError(f"expected one archived row for L={length}, lambda={coupling}; got {len(matches)}")
    return matches[0]


def archived_observables(row: dict, profile: str) -> dict:
    if profile == "three_by_three":
        return {
            label: {"mean": float(row[mean]), "standard_error": float(row[error])}
            for label, (_, mean, error) in THREE_BY_THREE_OBSERVABLES.items()
        }
    if profile == "paper1_four_by_four":
        return {
            "energy": {
                "mean": float(row["Hdiag_value"]) + float(row["Hoffdiag_value"]),
                "standard_error": math.hypot(float(row["Hdiag_std"]), float(row["Hoffdiag_std"])),
            },
            "offdiagonal_energy_susceptibility": {
                "mean": float(row["offdiag_ES_value"]),
                "standard_error": float(row["offdiag_ES_std"]),
            },
            "offdiagonal_fidelity_susceptibility": {
                "mean": float(row["offdiag_FS_value"]),
                "standard_error": float(row["offdiag_FS_std"]),
            },
        }
    raise ValueError(f"unknown archive profile: {profile}")


def campaign_points(path: Path, length: int, coupling: float, beta: float) -> list:
    analyses = json.loads(path.read_text())
    points = []
    for analysis in analyses:
        for point in analysis.get("points", []):
            if (int(point["L"]) == length and close(float(point["lambda"]), coupling) and
                    close(float(point["beta"]), beta)):
                points.append(point)
    if not points:
        raise ValueError("campaign contains no matching points")
    return points


def inverse_variance(points: list, derived_name: str) -> dict:
    estimates = [point["derived"][derived_name] for point in points]
    weights = [1.0 / float(estimate["standard_error"]) ** 2 for estimate in estimates]
    total_weight = sum(weights)
    mean = sum(float(estimate["mean"]) * weight
               for estimate, weight in zip(estimates, weights)) / total_weight
    standard_error = math.sqrt(1.0 / total_weight)
    cochran_q = sum(weight * (float(estimate["mean"]) - mean) ** 2
                    for estimate, weight in zip(estimates, weights))
    return {"mean": mean, "standard_error": standard_error, "cochran_q": cochran_q,
            "runs": len(estimates)}


def compare(campaign: Path, archive: Path, length: int, coupling: float, beta: float,
            profile: str = "three_by_three") -> dict:
    points = campaign_points(campaign, length, coupling, beta)
    archived = select_archive_row(archive, length, coupling, beta, profile)
    archived_values = archived_observables(archived, profile)
    derived_names = {
        "energy": "energy",
        "specific_heat": "specific_heat",
        "offdiagonal_energy_susceptibility": "energy_susceptibility",
        "offdiagonal_fidelity_susceptibility": "fidelity_susceptibility",
    }
    comparisons = {}
    for label, old in archived_values.items():
        derived_name = derived_names[label]
        current = inverse_variance(points, derived_name)
        old_mean, old_error = old["mean"], old["standard_error"]
        combined_error = math.hypot(current["standard_error"], old_error)
        difference = current["mean"] - old_mean
        comparisons[label] = {
            "current": current,
            "archived": {"mean": old_mean, "standard_error": old_error},
            "difference": difference,
            "combined_standard_error": combined_error,
            "z_score": difference / combined_error,
            "within_three_combined_se": abs(difference) <= 3.0 * combined_error,
        }
    limitations = [
        "Current estimates are inverse-variance combinations of independently jackknifed runs.",
    ]
    if profile == "three_by_three":
        limitations.append(
            "The archived simulation also measured custom A/B observables, so raw compute costs are not matched.")
    else:
        limitations.append(
            "Diagonal/off-diagonal energy errors are combined without archived covariance.")
    return {
        "schema_version": 1,
        "archive_profile": profile,
        "coordinate": {"L": length, "lambda": coupling, "beta": beta},
        "campaign_summary": str(campaign),
        "campaign_summary_sha256": sha256(campaign),
        "archive_csv": str(archive),
        "archive_csv_sha256": sha256(archive),
        "run_ids": sorted({point["run_id"] for point in points}),
        "comparisons": comparisons,
        "all_within_three_combined_se": all(
            item["within_three_combined_se"] for item in comparisons.values()),
        "limitations": limitations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--L", required=True, type=int)
    parser.add_argument("--lambda", dest="coupling", required=True, type=float)
    parser.add_argument("--beta", required=True, type=float)
    parser.add_argument("--profile", choices=("three_by_three", "paper1_four_by_four"),
                        default="three_by_three")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.campaign, args.archive, args.L, args.coupling, args.beta, args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {args.output}; all comparisons pass: {result['all_within_three_combined_se']}")


if __name__ == "__main__":
    main()
