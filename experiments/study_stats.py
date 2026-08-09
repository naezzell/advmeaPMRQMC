"""Statistical diagnostics and derived observables for study.py campaigns."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import study


OBSERVABLE_INDEX = {
    # study.py passes driving_term.txt and transverse_magnetization.txt before
    # the fixed set of fourteen standard observables.
    "driving": 0,
    "transverse_magnetization": 1,
    "energy": 2,
    "energy2": 3,
    "hdiag": 4,
    "hoffdiag": 6,
    "hdiag_eint": 10,
    "hdiag_fint": 11,
    "hoffdiag_eint": 13,
    "hoffdiag_fint": 14,
}


def finite(values: Iterable[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    center = mean(values)
    return sum((value - center) ** 2 for value in values) / (len(values) - 1)


def standard_error(values: Sequence[float]) -> float:
    variance = sample_variance(values)
    return math.sqrt(variance / len(values)) if len(values) >= 2 else float("nan")


def ratio_trace(rows: Sequence[Mapping[str, float]], observable: str) -> List[float]:
    numerator = denominator = 0.0
    values = []
    for row in rows:
        numerator += float(row[f"signed_{observable}"])
        denominator += float(row["sign"])
        values.append(numerator / denominator if denominator else float("nan"))
    return values


def ratio_mean(rows: Sequence[Mapping[str, float]], observable: str) -> float:
    numerator = sum(float(row[f"signed_{observable}"]) for row in rows)
    denominator = sum(float(row["sign"]) for row in rows)
    return numerator / denominator if denominator else float("nan")


def linearized_ratio_values(rows: Sequence[Mapping[str, float]], observable: str) -> List[float]:
    estimate = ratio_mean(rows, observable)
    return [float(row[f"signed_{observable}"]) - estimate * float(row["sign"])
            for row in rows]


def integrated_autocorrelation(values: Sequence[float], max_lag: int = 1000) -> Dict[str, float]:
    """Geyer initial-positive-pair IAT with an explicit truncation warning."""
    values = finite(values)
    if len(values) < 4:
        return {"iat": float("nan"), "max_lag": 0, "included_lag": 0,
                "hit_cap": False, "valid": False}
    center = mean(values)
    centered = [value - center for value in values]
    variance = sum(value * value for value in centered) / len(centered)
    if variance <= 0:
        return {"iat": 1.0, "max_lag": 0, "included_lag": 0,
                "hit_cap": False, "valid": True}
    lag_limit = min(len(centered) // 2, int(max_lag))
    correlations = []
    for lag in range(1, lag_limit + 1):
        covariance = sum(centered[index] * centered[index + lag]
                         for index in range(len(centered) - lag)) / (len(centered) - lag)
        correlations.append(covariance / variance)
    included = []
    stopped = False
    for start in range(0, len(correlations) - 1, 2):
        if correlations[start] + correlations[start + 1] <= 0:
            stopped = True
            break
        included.extend(correlations[start:start + 2])
    iat = max(1.0, 1.0 + 2.0 * sum(included))
    hit_cap = not stopped and lag_limit == max_lag
    return {"iat": iat, "max_lag": lag_limit, "included_lag": len(included),
            "hit_cap": hit_cap, "valid": not hit_cap}


def blocking_plateau(values: Sequence[float], minimum_blocks: int = 50) -> Dict:
    """Compare adjacent dyadic blocking standard errors for a stable plateau."""
    values = finite(values)
    candidates = []
    block = 1
    while len(values) // block >= minimum_blocks:
        block_means = [mean(values[start:start + block])
                       for start in range(0, len(values) - block + 1, block)]
        candidates.append({"block_length": block, "blocks": len(block_means),
                           "standard_error": standard_error(block_means)})
        block *= 2
    plateau = False
    selected = candidates[-1] if candidates else None
    for left, right in zip(candidates, candidates[1:]):
        denominator = max(abs(left["standard_error"]), abs(right["standard_error"]), 1e-300)
        if abs(right["standard_error"] - left["standard_error"]) / denominator <= 0.1:
            plateau = True
            selected = right
    return {"valid": plateau and selected is not None,
            "selected": selected, "candidates": candidates}


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = average_rank
        position = end
    return ranks


def split_rank_normalized_rhat(chains: Sequence[Sequence[float]]) -> float:
    """Rank-normalized split R-hat; NaN when fewer than four split chains exist."""
    usable = [finite(chain) for chain in chains if len(finite(chain)) >= 4]
    if len(usable) < 2:
        return float("nan")
    half = min(len(chain) // 2 for chain in usable)
    if half < 2:
        return float("nan")
    split = [chain[:half] for chain in usable] + [chain[-half:] for chain in usable]
    flattened = [value for chain in split for value in chain]
    ranks = _ranks(flattened)
    count = len(flattened)
    normal = NormalDist()
    transformed = [normal.inv_cdf((rank - 0.375) / (count + 0.25)) for rank in ranks]
    transformed_chains = [transformed[index * half:(index + 1) * half]
                          for index in range(len(split))]
    chain_means = [mean(chain) for chain in transformed_chains]
    within = mean([sample_variance(chain) for chain in transformed_chains])
    if within <= 0:
        return 1.0
    between = half * sample_variance(chain_means)
    variance = ((half - 1) / half) * within + between / half
    return math.sqrt(max(variance / within, 0.0))


def drift_test(chains: Sequence[Sequence[float]], sigma: float = 2.0) -> Dict:
    early, late = [], []
    for chain in chains:
        clean = finite(chain)
        if len(clean) < 4:
            continue
        half = len(clean) // 2
        early.append(mean(clean[:half]))
        late.append(mean(clean[-half:]))
    if len(early) < 2:
        return {"valid": False, "difference": float("nan"), "threshold": float("nan")}
    difference = abs(mean(late) - mean(early))
    threshold = sigma * math.sqrt(standard_error(early) ** 2 + standard_error(late) ** 2)
    return {"valid": difference <= threshold, "difference": difference,
            "threshold": threshold}


def split_streams(rows: Sequence[Mapping[str, float]]) -> Dict[str, List[Mapping[str, float]]]:
    streams = {}
    for row in rows:
        key = str(row.get("stream", row.get("rank", row.get("ladder", 0))))
        streams.setdefault(key, []).append(row)
    return streams


def block_sums(rows: Sequence[Mapping[str, float]], observable_columns: Sequence[str],
               block_length: int) -> List[Dict[str, float]]:
    blocks = []
    for start in range(0, len(rows), block_length):
        block = rows[start:start + block_length]
        if len(block) < block_length:
            continue
        result = {"count": len(block), "sign": sum(float(row["sign"]) for row in block)}
        for column in observable_columns:
            result[column] = sum(float(row[f"signed_{column}"]) for row in block)
        blocks.append(result)
    return blocks


def derived_from_sums(sums: Mapping[str, float], beta: float, spins: int) -> Dict[str, float]:
    sign = sums["sign"]
    if not sign:
        return {name: float("nan") for name in
                ("energy", "energy_density", "specific_heat", "specific_heat_density",
                 "energy_susceptibility", "energy_susceptibility_density",
                 "fidelity_susceptibility", "fidelity_susceptibility_density", "effective_gap")}
    get = lambda column: sums.get(column, float("nan")) / sign
    energy, energy2 = get("obs_2"), get("obs_3")
    hoffdiag = get("obs_6")
    eint, fint = get("obs_13"), get("obs_14")
    specific_heat = beta * beta * (energy2 - energy * energy)
    energy_susceptibility = eint - beta * hoffdiag * hoffdiag
    fidelity_susceptibility = fint - (beta * hoffdiag) ** 2 / 8.0
    gap = (energy_susceptibility / (2.0 * fidelity_susceptibility)
           if math.isfinite(fidelity_susceptibility) and fidelity_susceptibility > 0 else float("nan"))
    return {
        "energy": energy, "energy_density": energy / spins,
        "specific_heat": specific_heat, "specific_heat_density": specific_heat / spins,
        "energy_susceptibility": energy_susceptibility,
        "energy_susceptibility_density": energy_susceptibility / spins,
        "fidelity_susceptibility": fidelity_susceptibility,
        "fidelity_susceptibility_density": fidelity_susceptibility / spins,
        "effective_gap": gap,
    }


def joint_block_jackknife(rows: Sequence[Mapping[str, float]], beta: float, spins: int,
                          block_length: int,
                          observable_columns: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]:
    required = ["obs_2", "obs_3", "obs_6", "obs_13", "obs_14"]
    if observable_columns is not None:
        required = [column for column in required if column in observable_columns]
    available = [column for column in required if rows and f"signed_{column}" in rows[0]]
    # Never let a jackknife block straddle independent chains/ladders.
    blocks = []
    for stream_rows in split_streams(rows).values():
        blocks.extend(block_sums(stream_rows, available, block_length))
    total = {"sign": sum(block["sign"] for block in blocks)}
    for column in available:
        total[column] = sum(block[column] for block in blocks)
    estimate = derived_from_sums(total, beta, spins)
    if len(blocks) < 2:
        return {name: {"mean": value, "standard_error": float("nan"), "blocks": len(blocks)}
                for name, value in estimate.items()}
    replicates = []
    for block in blocks:
        leave_one_out = {key: total[key] - block.get(key, 0.0) for key in total}
        replicates.append(derived_from_sums(leave_one_out, beta, spins))
    output = {}
    for name, value in estimate.items():
        values = finite(replicate[name] for replicate in replicates)
        if len(values) != len(replicates):
            error = float("nan")
        else:
            center = mean(values)
            error = math.sqrt((len(values) - 1) / len(values) *
                              sum((replicate - center) ** 2 for replicate in values))
        output[name] = {"mean": value, "standard_error": error, "blocks": len(blocks)}
    return output


def parse_trace(path: Path) -> List[Dict[str, float]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if not rows:
        return []
    if "sign" not in rows[0] or not any(key.startswith("signed_obs_") for key in rows[0]):
        raise ValueError(f"{path} does not contain signed trace columns")
    output = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            if key in ("stream", "rank", "ladder"):
                converted[key] = str(value)
            else:
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = value
        output.append(converted)
    return output


def split_parameter_points(rows: Sequence[Mapping[str, float]],
                           planned: Mapping[str, str]) -> List[Tuple[Dict, List[Mapping[str, float]]]]:
    """Preserve every stationary PT/QCPT slot as a production marginal."""
    method = planned["method"]
    coordinate = "slot" if method == "qcpt" else "temperature" if method == "beta_pt" else None
    if coordinate is None or not rows or coordinate not in rows[0]:
        return [({"point": 0, "beta": float(planned["beta"]),
                  "lambda": float(planned["lambda"])}, list(rows))]
    grouped = {}
    for row in rows:
        point = int(float(row[coordinate]))
        grouped.setdefault(point, []).append(row)
    output = []
    for point, point_rows in sorted(grouped.items()):
        first = point_rows[0]
        gamma = float(first["gamma"]) if method == "qcpt" else float(planned["lambda"])
        output.append(({"point": point, coordinate: point, "beta": float(first["beta"]),
                        "lambda": gamma}, point_rows))
    return output


def relative_error(result: Mapping[str, float]) -> float:
    value, error = result["mean"], result["standard_error"]
    if not math.isfinite(value) or not math.isfinite(error) or value == 0:
        return float("inf")
    return abs(error / value)


PRECISION_TARGETS = {
    "energy_density": 0.001,
    "specific_heat_density": 0.05,
    "energy_susceptibility_density": 0.05,
    "fidelity_susceptibility_density": 0.05,
    "effective_gap": 0.10,
}


def analyze_point(rows: Sequence[Mapping[str, float]], planned: Mapping[str, str],
                  summary: Mapping, coordinate: Mapping) -> Dict:
    streams = split_streams(rows)
    energy_column = "obs_2"
    diagnostics = []
    total_effective = 0.0
    for stream, stream_rows in streams.items():
        values = linearized_ratio_values(stream_rows, energy_column)
        autocorrelation = integrated_autocorrelation(values)
        plateau = blocking_plateau(values)
        effective = len(values) / autocorrelation["iat"] if autocorrelation["iat"] else 0.0
        total_effective += effective
        diagnostics.append({"stream": stream, "measurements": len(values),
                            "autocorrelation": autocorrelation,
                            "blocking": plateau, "effective_samples": effective})
    component_extractors = {
        "energy_numerator": lambda row: float(row["signed_obs_2"]),
        "sign": lambda row: float(row["sign"]),
    }
    if rows and "signed_obs_0" in rows[0]:
        component_extractors["driving_numerator"] = lambda row: float(row["signed_obs_0"])
    if rows and "expansion_order" in rows[0]:
        component_extractors["expansion_order"] = lambda row: float(row["expansion_order"])
    rhat_components, drift_components = {}, {}
    for name, extractor in component_extractors.items():
        component_chains = [[extractor(row) for row in stream_rows]
                            for stream_rows in streams.values()]
        rhat_components[name] = split_rank_normalized_rhat(component_chains)
        drift_components[name] = drift_test(component_chains)
    finite_rhats = finite(rhat_components.values())
    rhat = max(finite_rhats) if finite_rhats else float("nan")
    drift = {"valid": all(item["valid"] for item in drift_components.values()),
             "components": drift_components}
    valid_iats = finite(item["autocorrelation"]["iat"] for item in diagnostics)
    block_length = max(1, int(math.ceil(10 * max(valid_iats)))) if valid_iats else 1
    protocol = planned["protocol"]
    requested_columns = {
        "none": [], "cheap": ["obs_2", "obs_3"],
        "es_only": ["obs_2", "obs_3", "obs_6", "obs_13"],
        "fs_only": ["obs_2", "obs_3", "obs_6", "obs_14"],
    }.get(protocol, ["obs_2", "obs_3", "obs_6", "obs_13", "obs_14"])
    derived = joint_block_jackknife(rows, float(coordinate["beta"]), int(planned["L"]) ** 2,
                                    block_length, requested_columns)
    protocol_quantities = {
        "none": [],
        "cheap": ["energy_density", "specific_heat_density"],
        "es_only": ["energy_density", "specific_heat_density", "energy_susceptibility_density"],
        "fs_only": ["energy_density", "specific_heat_density", "fidelity_susceptibility_density"],
    }.get(protocol, ["energy_density", "specific_heat_density",
                     "energy_susceptibility_density", "fidelity_susceptibility_density",
                     "effective_gap"])
    precision = {name: relative_error(derived[name]) <= PRECISION_TARGETS[name]
                 for name in protocol_quantities}
    streams_sufficient = len(streams) >= 2
    thermalization_pass = (not streams_sufficient or
                           (math.isfinite(rhat) and rhat <= 1.01 and drift["valid"]))
    correlation_pass = (all(item["autocorrelation"]["valid"] for item in diagnostics) and
                        all(item["blocking"]["valid"] for item in diagnostics) and
                        derived["energy"]["blocks"] >= 50)
    convergence = thermalization_pass and correlation_pass
    analysis_pass = (protocol != "none" and summary.get("status") == "complete" and convergence and
                     total_effective >= 400 and all(precision.values()))
    wall = float(summary.get("timings", {}).get("simulation", float("nan")))
    ranks = 4
    measurement_core_seconds = sum(
        max((float(row.get("measurement_seconds", 0.0)) for row in stream_rows), default=0.0)
        for stream_rows in streams.values())
    return {
        **coordinate, "run_id": planned["run_id"], "status": "analyzed", "method": planned["method"],
        "protocol": planned["protocol"], "L": int(planned["L"]),
        "schedule_name": planned.get("schedule_name", ""), "move": planned.get("move", "none"),
        "beta_over_L": float(coordinate["beta"]) / int(planned["L"]), "seed": int(planned["seed"]),
        "tuning": planned["tuning"] == "True", "streams": len(streams),
        "rhat": rhat, "drift": drift, "diagnostics": diagnostics,
        "rhat_components": rhat_components,
        "effective_samples": total_effective, "block_length": block_length,
        "derived": derived, "precision": precision, "convergence_pass": convergence,
        "thermalization_pass": thermalization_pass, "correlation_pass": correlation_pass,
        "analysis_pass": analysis_pass, "wall_seconds": wall,
        "core_hours": wall * ranks / 3600.0 if math.isfinite(wall) else float("nan"),
        "ess_per_core_hour": total_effective / (wall * ranks / 3600.0)
        if math.isfinite(wall) and wall > 0 else float("nan"),
        "mean_sign": mean([float(row["sign"]) for row in rows]),
        "measurement_core_seconds": measurement_core_seconds,
        "measurement_core_fraction": measurement_core_seconds / (wall * ranks)
        if math.isfinite(wall) and wall > 0 else float("nan"),
    }


def tempering_metadata(run_directory: Path) -> Dict:
    swaps_path = run_directory / "tempered_swaps.csv"
    flow_path = run_directory / "tempered_flow.csv"
    result = {"worst_edge_acceptance": float("nan"), "round_trips": 0,
              "qmax_achieved": False, "mean_q": float("nan"),
              "crossed_weight_seconds": float("nan"),
              "measurement_core_seconds": float("nan")}
    if swaps_path.exists():
        with swaps_path.open(newline="") as stream:
            acceptances = [float(row["acceptance"]) for row in csv.DictReader(stream)]
        result["worst_edge_acceptance"] = min(acceptances) if acceptances else float("nan")
    if flow_path.exists():
        with flow_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            result.update({
                "round_trips": int(rows[0]["round_trips"]),
                "qmax_achieved": bool(int(rows[0]["qmax_achieved"])),
                "mean_q": float(rows[0]["mean_q"]),
                "crossed_weight_seconds": float(rows[0]["crossed_weight_seconds"]),
                "measurement_core_seconds": float(rows[0].get("measurement_seconds", "nan")),
            })
    return result


def attach_exact_reference(run_directory: Path, result: Dict, absolute_tolerance: float = 0.03) -> None:
    if int(result.get("L", 99)) > 3 or not result.get("points"):
        return
    from exact_diagonalization import exact_split_thermal_observables, exact_thermal_observables
    spins = int(result["L"]) ** 2
    for point in result["points"]:
        if result["method"] == "qcpt":
            exact = exact_split_thermal_observables(
                run_directory / "H_fixed.txt", run_directory / "H_gamma.txt",
                float(point["lambda"]), spins, float(point["beta"]))
        else:
            exact = exact_thermal_observables(run_directory / "H.txt", spins, float(point["beta"]))
        exact["energy_density"] = exact["energy"] / spins
        exact["specific_heat_density"] = exact["specific_heat"] / spins
        checks = {}
        for quantity in ("energy_density", "specific_heat_density"):
            estimate = point["derived"][quantity]
            tolerance = max(absolute_tolerance, 5 * float(estimate["standard_error"]))
            error = abs(float(estimate["mean"]) - exact[quantity])
            checks[quantity] = {"absolute_error": error, "tolerance": tolerance,
                                "passed": math.isfinite(error) and error <= tolerance}
        point["exact_reference"] = exact
        point["exact_checks"] = checks
        point["exact_pass"] = all(check["passed"] for check in checks.values())


def analyze_run(run_directory: Path) -> Dict:
    manifest = json.loads((run_directory / "manifest.json").read_text())
    summary = json.loads((run_directory / "summary.json").read_text())
    planned = manifest["planned"]
    # Independent stream files are canonical for convergence diagnostics.  The
    # rank-averaged compatibility trace is retained for older plotting tools,
    # but must not be concatenated with streams or it would double-count data.
    trace_paths = sorted(run_directory.glob("trace_stream.rank*.csv"))
    if not trace_paths:
        trace_paths = sorted(run_directory.glob("trace*.csv"))
    rows = []
    for path in trace_paths:
        parsed = parse_trace(path)
        if path.name != "trace.csv":
            stream_name = path.stem.replace("trace_", "")
            for row in parsed:
                row.setdefault("stream", stream_name)
        rows.extend(parsed)
    if not rows:
        result = {"run_id": planned["run_id"], "status": "no_trace", "analysis_pass": False,
                  "points": [], "trace_paths": [str(path) for path in trace_paths]}
        study.write_json(run_directory / "analysis.json", result)
        return result
    points = [analyze_point(point_rows, planned, summary, coordinate)
              for coordinate, point_rows in split_parameter_points(rows, planned)]
    result = {
        "run_id": planned["run_id"], "status": "analyzed", "method": planned["method"],
        "protocol": planned["protocol"], "L": int(planned["L"]), "seed": int(planned["seed"]),
        "analysis_pass": bool(points) and all(point["analysis_pass"] for point in points),
        "points": points, "trace_paths": [str(path) for path in trace_paths],
        "tempering": tempering_metadata(run_directory) if planned["method"] in ("beta_pt", "qcpt") else {},
        "schedule_name": planned.get("schedule_name", ""), "move": planned.get("move", "none"),
        "representation": planned.get("representation", "standard"),
        "periodic": planned.get("periodic", "True") == "True",
        "target_lambda": float(planned["lambda"]), "target_beta": float(planned["beta"]),
        "tuning": planned.get("tuning", "False") == "True",
    }
    attach_exact_reference(run_directory, result)
    study.write_json(run_directory / "analysis.json", result)
    return result


def flatten_analysis(result: Mapping) -> List[Dict]:
    output = []
    for point in result.get("points", []):
        row = {key: point.get(key) for key in
               ("run_id", "status", "method", "protocol", "L", "lambda", "beta",
                "beta_over_L", "point", "slot", "temperature", "seed", "tuning",
                "schedule_name", "move", "mean_sign", "measurement_core_seconds",
                "measurement_core_fraction",
                "streams", "rhat", "effective_samples", "convergence_pass",
                "analysis_pass", "wall_seconds", "core_hours", "ess_per_core_hour")}
        for name, estimate in point.get("derived", {}).items():
            row[name] = estimate.get("mean")
            row[name + "_se"] = estimate.get("standard_error")
        output.append(row)
    return output


def schedule_candidate_records(analyses: Sequence[Mapping]) -> List[Dict]:
    records = []
    for result in analyses:
        if result.get("method") != "qcpt" or not result.get("tuning") or not result.get("points"):
            continue
        points = result["points"]
        endpoint = max(points, key=lambda point: int(point["point"]))
        core_hours = endpoint.get("core_hours", float("nan"))
        sweep_ess = sum(float(point.get("effective_samples", 0.0)) for point in points)
        tempering = result.get("tempering", {})
        records.append({
            "run_id": result["run_id"], "L": result["L"],
            "target_lambda": result["target_lambda"], "target_beta": result["target_beta"],
            "protocol": result["protocol"], "representation": result["representation"],
            "periodic": result["periodic"], "move": result["move"],
            "schedule_name": result["schedule_name"], "seed": result["seed"],
            "target_ess_per_core_hour": endpoint.get("ess_per_core_hour", float("nan")),
            "sweep_ess_per_core_hour": sweep_ess / core_hours
            if math.isfinite(float(core_hours)) and float(core_hours) > 0 else float("nan"),
            "worst_edge_acceptance": tempering.get("worst_edge_acceptance", float("nan")),
            "round_trips": tempering.get("round_trips", 0),
            "qmax_achieved": tempering.get("qmax_achieved", False),
            "sign_gate": all(float(point.get("mean_sign", 0.0)) > 0 for point in points),
            "convergence_gate": all(bool(point.get("convergence_pass")) for point in points),
        })
    return records


def rank_schedule_records(records: Sequence[Mapping], minimum_seeds: int = 4,
                          minimum_acceptance: float = 0.15) -> List[Dict]:
    candidates = {}
    candidate_fields = ("L", "target_lambda", "target_beta", "protocol", "representation",
                        "periodic", "move", "schedule_name")
    for record in records:
        key = tuple(record[field] for field in candidate_fields)
        candidates.setdefault(key, []).append(record)
    ranked = []
    for key, runs in candidates.items():
        acceptances = finite(float(run["worst_edge_acceptance"]) for run in runs)
        target_scores = finite(float(run["target_ess_per_core_hour"]) for run in runs)
        sweep_scores = finite(float(run["sweep_ess_per_core_hour"]) for run in runs)
        eligible = (len({run["seed"] for run in runs}) >= minimum_seeds and
                    len(target_scores) == len(runs) and bool(acceptances) and
                    min(acceptances) >= minimum_acceptance and
                    sum(int(run["round_trips"]) for run in runs) > 0 and
                    not any(bool(run["qmax_achieved"]) for run in runs) and
                    all(bool(run["sign_gate"]) and bool(run["convergence_gate"]) for run in runs))
        row = dict(zip(candidate_fields, key))
        row.update({
            "runs": len(runs), "seeds": ";".join(str(seed) for seed in sorted({run["seed"] for run in runs})),
            "run_ids": ";".join(sorted(str(run["run_id"]) for run in runs)),
            "worst_edge_acceptance": min(acceptances) if acceptances else float("nan"),
            "round_trips": sum(int(run["round_trips"]) for run in runs),
            "median_target_ess_per_core_hour": statistics.median(target_scores) if target_scores else float("nan"),
            "median_sweep_ess_per_core_hour": statistics.median(sweep_scores) if sweep_scores else float("nan"),
            "eligible": eligible, "selected": False,
        })
        ranked.append(row)
    comparison_fields = candidate_fields[:-1]
    comparison_groups = {}
    for row in ranked:
        comparison_groups.setdefault(tuple(row[field] for field in comparison_fields), []).append(row)
    for group in comparison_groups.values():
        survivors = [row for row in group if row["eligible"]]
        if survivors:
            max(survivors, key=lambda row: (row["median_target_ess_per_core_hour"],
                                            row["median_sweep_ess_per_core_hour"]))["selected"] = True
    return sorted(ranked, key=lambda row: tuple(str(row[field]) for field in comparison_fields) +
                  (not row["selected"], -float(row["median_target_ess_per_core_hour"])
                   if math.isfinite(float(row["median_target_ess_per_core_hour"])) else float("inf")))


def selected_production_rows(root: Path, ranking: Sequence[Mapping]) -> List[Dict[str, str]]:
    selected = [row for row in ranking if row.get("selected")]
    if not selected or not (root / "plan.csv").exists():
        return []
    plan = study.read_csv(root / "plan.csv")
    campaign_manifest = json.loads((root / "campaign_manifest.json").read_text())
    production_seeds = campaign_manifest["config"].get(
        "production_seeds", [5100, 6100, 7100, 8100]
    )
    rows = []
    for winner in selected:
        template = next((row for row in plan
                         if row["method"] == "qcpt" and row["tuning"] == "True" and
                         row["schedule_name"] == str(winner["schedule_name"]) and
                         int(row["L"]) == int(winner["L"]) and
                         float(row["lambda"]) == float(winner["target_lambda"]) and
                         float(row["beta"]) == float(winner["target_beta"]) and
                         row["protocol"] == winner["protocol"] and
                         row["representation"] == winner["representation"] and
                         row.get("move", "none") == winner["move"]), None)
        if template is None:
            continue
        simulation = {name: int(template[name]) for name in
                      ("Tsteps", "steps", "steps_per_measurement", "qmax", "nbins",
                       "max_wall_seconds")}
        for seed in production_seeds:
            rows.append(study.make_plan_row(
                template["source_commit"], template["campaign"] + "_production", "qcpt",
                template["protocol"], int(template["L"]), float(template["lambda"]),
                float(template["beta"]), template["periodic"] == "True",
                template["representation"], int(template["parity"]), int(seed), False,
                simulation, schedule_name=template["schedule_name"], move=template.get("move", "none"),
            ))
    unique = {row["run_id"]: row for row in rows}
    return sorted(unique.values(), key=lambda row: row["run_id"])


def adaptive_extension_rows(root: Path, analyses: Sequence[Mapping]) -> Tuple[List[Dict[str, str]], List[Dict]]:
    """Create immutable successor runs; never weaken a failed statistical gate."""
    if not (root / "plan.csv").exists():
        return [], []
    plan = {row["run_id"]: row for row in study.read_csv(root / "plan.csv")}
    extensions, decisions = [], []
    for result in analyses:
        template = plan.get(result.get("run_id"))
        points = result.get("points", [])
        if template is None or not points or result.get("analysis_pass"):
            continue
        double_warmup = any(not point.get("thermalization_pass", False) for point in points)
        double_sampling = any(
            not point.get("correlation_pass", False) or
            float(point.get("effective_samples", 0.0)) < 400 or
            not all(point.get("precision", {}).values())
            for point in points
        )
        simulation = {name: int(template[name]) for name in
                      ("Tsteps", "steps", "steps_per_measurement", "qmax", "nbins",
                       "max_wall_seconds")}
        if double_warmup:
            simulation["Tsteps"] *= 2
        if double_sampling:
            simulation["steps"] *= 2
        wall = max((float(point.get("wall_seconds", float("nan"))) for point in points), default=float("nan"))
        old_updates = int(template["Tsteps"]) + int(template["steps"])
        new_updates = simulation["Tsteps"] + simulation["steps"]
        predicted_wall = wall * new_updates / old_updates if math.isfinite(wall) else float("nan")
        reasons = ((["thermalization"] if double_warmup else []) +
                   (["correlation_ess_or_precision"] if double_sampling else []))
        decision = {"parent_run_id": template["run_id"], "reasons": reasons,
                    "predicted_wall_seconds": predicted_wall, "job_cap_seconds": simulation["max_wall_seconds"]}
        if math.isfinite(predicted_wall) and predicted_wall > simulation["max_wall_seconds"]:
            decision.update({"status": "job_cap_rejected", "successor_run_id": None})
            decisions.append(decision)
            continue
        successor = study.make_plan_row(
            study.source_commit(), template["campaign"], template["method"],
            template["protocol"], int(template["L"]), float(template["lambda"]),
            float(template["beta"]), template["periodic"] == "True",
            template["representation"], int(template["parity"]), int(template["seed"]),
            template["tuning"] == "True", simulation, schedule_name=template["schedule_name"],
            move=template.get("move", "none"))
        if successor["run_id"] in plan:
            decision.update({"status": "already_planned", "successor_run_id": successor["run_id"]})
        else:
            extensions.append(successor)
            decision.update({"status": "planned", "successor_run_id": successor["run_id"]})
        decisions.append(decision)
    unique = {row["run_id"]: row for row in extensions}
    return sorted(unique.values(), key=lambda row: row["run_id"]), decisions


def beta_convergence(rows: Sequence[Mapping], quantity: str) -> List[Dict]:
    groups = {}
    for row in rows:
        key = (row.get("method"), row.get("protocol"), row.get("L"), row.get("lambda"), row.get("seed"))
        if math.isfinite(float(row.get(quantity, float("nan")))):
            groups.setdefault(key, []).append(row)
    output = []
    for key, samples in groups.items():
        samples.sort(key=lambda row: float(row["beta"]))
        comparisons = []
        for left, right in zip(samples, samples[1:]):
            difference = abs(float(right[quantity]) - float(left[quantity]))
            pooled = math.sqrt(float(right[quantity + "_se"]) ** 2 + float(left[quantity + "_se"]) ** 2)
            scale = max(abs(float(right[quantity])), 1e-300)
            comparisons.append({
                "left_beta": float(left["beta"]), "right_beta": float(right["beta"]),
                "difference": difference, "pooled_standard_error": pooled,
                "statistically_stable": difference <= 2 * pooled,
                "precision_stable": difference / scale <= PRECISION_TARGETS[quantity],
            })
        converged = (len(comparisons) >= 2 and all(
            comparison["statistically_stable"] and comparison["precision_stable"]
            for comparison in comparisons[-2:]
        ))
        output.append({"method": key[0], "protocol": key[1], "L": key[2],
                       "lambda": key[3], "seed": key[4], "quantity": quantity,
                       "converged": converged, "comparisons": comparisons,
                       "next_beta_over_L": None if converged or not samples else
                       min(8.0, 2 * float(samples[-1]["beta_over_L"]))})
    return output


def analyze_campaign(root: Path) -> None:
    analyses = []
    runs = root / "runs"
    if runs.exists():
        for run_directory in sorted(path for path in runs.iterdir() if path.is_dir()):
            if (run_directory / "manifest.json").exists() and (run_directory / "summary.json").exists():
                analyses.append(analyze_run(run_directory))
    flat = [row for result in analyses for row in flatten_analysis(result)]
    study.write_csv(root / "summary.csv", flat)
    study.write_json(root / "summary.json", analyses)
    beta = []
    for quantity in PRECISION_TARGETS:
        beta.extend(beta_convergence(flat, quantity))
    study.write_json(root / "beta_convergence.json", beta)
    schedule_ranking = rank_schedule_records(schedule_candidate_records(analyses))
    study.write_csv(root / "schedule_ranking.csv", schedule_ranking)
    study.write_json(root / "schedule_ranking.json", schedule_ranking)
    study.write_csv(root / "production_plan.csv", selected_production_rows(root, schedule_ranking))
    extension_rows, extension_decisions = adaptive_extension_rows(root, analyses)
    study.write_csv(root / "extension_plan.csv", extension_rows)
    study.write_json(root / "extension_decisions.json", extension_decisions)
    print(f"analyzed {len(analyses)} runs; {sum(bool(result.get('analysis_pass')) for result in analyses)} passed")
