import logging
from pathlib import Path
import pandas as pd


def compare_groups(directory_path, groups, output_path=None):
    """
    Compare metrics between control and patient groups.

    Args:
        directory_path (Path): Path to the base directory containing group metrics.
        groups (list): List of groups to compare (e.g., ["control", "pd"]).
        output_path (Path): Optional path to save results.

    Returns:
        dict: Differences in graph and node metrics.
    """
    directory_path = Path(directory_path)
    statistics = {}

    for group in groups:
        try:
            statistics[group] = load_statistics(directory_path, group)
        except (FileNotFoundError, ValueError) as exc:
            logging.error(f"Skipping group '{group}': {exc}")
            continue

    if "control" not in statistics:
        raise ValueError("Control group metrics are required for comparison")

    control_statistics = statistics["control"]
    results = {}

    for group in groups:
        if group == "control" or group not in statistics:
            continue

        patient_statistics = statistics[group]
        graph_differences = calculate_differences(control_statistics["graph"], patient_statistics["graph"])
        node_differences = calculate_differences(control_statistics["node"], patient_statistics["node"])

        results[group] = {"graph_differences": graph_differences, "node_differences": node_differences}

        # Save results to CSV if output_path is specified
        if output_path:
            save_path = Path(output_path) / group
            save_path.mkdir(parents=True, exist_ok=True)
            graph_differences.to_csv(save_path / "graph_differences.csv", index=False)
            node_differences.to_csv(save_path / "node_differences.csv", index=False)

    return results


def load_statistics(directory_path, group_name):
    """
    Load graph and node metrics for a specific group.

    Args:
        directory_path (Path): Path to the directory containing metrics files.
        group_name (str): Name of the group (e.g., "control", "pd").

    Returns:
        dict: Graph and node metrics as DataFrames.

    Raises:
        FileNotFoundError: If the required metrics files are not found.
        ValueError: If the metrics files are empty.
    """
    graph_metrics_file = directory_path / group_name / "stats" / "graph_statistics.csv"
    node_metrics_file = directory_path / group_name / "stats" / "node_statistics.csv"

    # Ensure files exist
    if not graph_metrics_file.exists() or not node_metrics_file.exists():
        raise FileNotFoundError(f"Metrics files not found for group: {group_name}")

    # Load CSV files
    graph_metrics = pd.read_csv(graph_metrics_file)
    node_metrics = pd.read_csv(node_metrics_file)

    if graph_metrics.empty or node_metrics.empty:
        raise ValueError(f"Metrics files are empty for group: {group_name}")

    return {"graph": graph_metrics, "node": node_metrics}


def calculate_differences(control_metrics, patient_metrics):
    """
    Calculate differences for a specific metric type (graph or node) between groups.

    Args:
        control_metrics (pd.DataFrame): Metrics for the control group.
        patient_metrics (pd.DataFrame): Metrics for the patient group.

    Returns:
        pd.DataFrame: Differences in metrics between groups.
    """
    # Ensure columns match between the two datasets
    if set(control_metrics.columns) != set(patient_metrics.columns):
        raise ValueError("Control and patient metrics must have the same columns")

    differences = patient_metrics.copy()

    for column in control_metrics.columns:
        if column not in ["Node", "Metric"]:
            differences[column] = abs(patient_metrics[column] - control_metrics[column])

    return differences
