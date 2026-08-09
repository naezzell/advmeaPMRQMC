"""Bootstrap matched fast/reference estimator speedups over independent seeds."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import compare_estimator_runs


def quantile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    fraction = position - left
    return ordered[left] * (1 - fraction) + ordered[right] * fraction


def summarize(pairs, bootstrap_samples=10000, seed=20260809):
    comparisons = [compare_estimator_runs.compare(Path(fast), Path(reference))
                   for fast, reference in pairs]
    measurement = [item["measurement_speedup"] for item in comparisons]
    wall = [item["wall_speedup"] for item in comparisons]
    rng = random.Random(seed)
    boot_measurement, boot_wall = [], []
    for _ in range(bootstrap_samples):
        sample = [rng.randrange(len(comparisons)) for _ in comparisons]
        boot_measurement.append(statistics.median(measurement[index] for index in sample))
        boot_wall.append(statistics.median(wall[index] for index in sample))
    interval = lambda values: [quantile(values, 0.025), quantile(values, 0.975)]
    measurement_interval, wall_interval = interval(boot_measurement), interval(boot_wall)
    correctness = all(item["correctness_pass"] for item in comparisons)
    result = {
        "pairs": comparisons, "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed, "median_measurement_speedup": statistics.median(measurement),
        "measurement_speedup_95_interval": measurement_interval,
        "median_wall_speedup": statistics.median(wall),
        "wall_speedup_95_interval": wall_interval,
        "correctness_pass": correctness,
        "measurement_speedup_claim_pass": correctness and measurement_interval[0] > 1.0,
        "wall_speedup_claim_pass": correctness and wall_interval[0] > 1.0,
    }
    protocols = [item.get("protocol") for item in comparisons if item.get("protocol")]
    if protocols:
        result["protocol"] = {key: protocols[0][key] for key in
                              ("L", "lambda", "beta", "steps", "steps_per_measurement",
                               "source_commit", "hardware")}
        result["seeds"] = [protocol["seed"] for protocol in protocols]
        result["q_range"] = [min(item["mean_q"] for item in comparisons),
                             max(item["max_q"] for item in comparisons)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", nargs=2, metavar=("FAST", "REFERENCE"), required=True)
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    args = parser.parse_args()
    result = summarize(args.pair, args.bootstrap_samples, args.bootstrap_seed)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")
    raise SystemExit(0 if result["correctness_pass"] else 1)


if __name__ == "__main__":
    main()
