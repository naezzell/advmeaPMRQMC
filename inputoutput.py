from qiskit.quantum_info import SparsePauliOp, Operator
import pandas as pd
import numpy as np

def write_as_paulis(filename, op):
    qop = Operator(op)
    pauli_op = SparsePauliOp.from_operator(qop)
    with open(filename, "w") as f:
        for coeff, gates in zip(pauli_op.coeffs, pauli_op.paulis):
            f.write(f"{np.real(coeff)} ")
            for index, gate in enumerate(gates):
                if str(gate) != "I":
                    f.write(f"{index + 1} {gate} ")
            f.write("\n")


from collections import defaultdict
import re

def parse_simulation_output(file_path_or_text, is_file=True):
    if is_file:
        with open(file_path_or_text, "r") as f:
            text = f.read()
    else:
        text = file_path_or_text

    # 1. Regex pattern to capture observable blocks and extract name, mean, and std.dev.
    block_pattern = re.compile(
        r"Total of (?:derived )?observable(?: #\d+)?: (?P<obs>\S+)\s*\n"
        r"Total mean\(O\) = (?P<mean>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\n"
        r"Total std\.dev\.\(O\) = (?P<std>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

    # 2. Regex pattern to filter target observables and extract index 'k'
    target_pattern = re.compile(
        r"^(measure_Hdiag_kint_|susceptibility_k|ratio_k)(\d+)$"
    )

    # Nested dictionary to store data by k: {k: {metric_mean: val, metric_std: val}}
    data = defaultdict(dict)

    for match in block_pattern.finditer(text):
        obs_name = match.group("obs")
        mean_val = float(match.group("mean"))
        std_val = float(match.group("std"))

        # Check if the observable matches one of our target patterns
        k_match = target_pattern.match(obs_name)
        if k_match:
            prefix, k_str = k_match.groups()
            k = int(k_str)

            # Map prefixes to clean base metric names
            if "measure_Hdiag" in prefix:
                metric = "measure_Hdiag_kint"
            elif "susceptibility" in prefix:
                metric = "susceptibility"
            else:
                metric = "ratio"

            # Assign values to the nested dictionary
            data[k][f"{metric}_mean"] = mean_val
            data[k][f"{metric}_std"] = std_val

    # 3. Convert nested dict into a pandas DataFrame
    df = (
        pd.DataFrame.from_dict(data, orient="index")
        .rename_axis("k")
        .reset_index()
        .sort_values("k")
    )

    return df