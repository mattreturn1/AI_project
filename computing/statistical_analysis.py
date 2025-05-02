from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
import pandas as pd
from pathlib import Path
import logging


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
    metrics = {}

    # Load metrics for each specified group
    for group in groups:
        try:
            metrics[group] = load_metrics(directory_path, group)
        except (FileNotFoundError, ValueError) as exc:
            logging.error(f"Skipping group '{group}': {exc}")
            continue

    # Ensure control group metrics are available
    if "control" not in metrics:
        raise ValueError("Control group metrics are required for comparison")

    control_metrics = metrics["control"]
    results = {}

    # Compare data between control and other groups
    for group in groups:
        if group == "control" or group not in metrics:
            continue

        patient_metrics = metrics[group]
        graph_differences = calculate_significant_differences(
            control_metrics["graph"], patient_metrics["graph"], "graph"
        )
        node_differences = calculate_significant_differences(
            control_metrics["node"], patient_metrics["node"], "node"
        )

        results[group] = {"graph_differences": graph_differences, "node_differences": node_differences}

        # Save results to CSV if output_path is specified
        if output_path:
            save_path = Path(output_path) / group
            save_path.mkdir(parents=True, exist_ok=True)
            graph_differences.to_csv(save_path / "graph_analysis.csv", index=False)
            node_differences.to_csv(save_path / "node_analysis.csv", index=False)

    return results


def load_metrics(directory_path, group_name):
    """
    Load graph and node metrics for a specific group.

    Args:
        directory_path (Path): Path to the directory containing metrics files.
        group_name (str): Name of the group (e.g., "control", "pd").

    Returns:
        dict: Graph and node metrics as DataFrames.
    """
    graph_metrics_file = directory_path / group_name / "metrics" / "graph_metrics.csv"
    node_metrics_file = directory_path / group_name / "metrics" / "node_metrics.csv"

    if not graph_metrics_file.exists() or not node_metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found for group: {group_name}")

    graph_metrics = pd.read_csv(graph_metrics_file)
    node_metrics = pd.read_csv(node_metrics_file)

    if graph_metrics.empty or node_metrics.empty:
        raise ValueError(f"Metrics file is empty for group: {group_name}")

    return {"graph": graph_metrics, "node": node_metrics}


def calculate_significant_differences(control_metrics, patient_metrics, metric_type):
    """
    Calculate differences for a specific metric type (graph or node) between groups.

    Args:
        control_metrics (pd.DataFrame): Metrics for the control group.
        patient_metrics (pd.DataFrame): Metrics for the patient group.
        metric_type (str): Type of metric ("graph" or "node").

    Returns:
        pd.DataFrame: Differences in metrics between groups.
    """
    if metric_type == "graph":
        differences = patient_metrics.copy()
        for column in control_metrics.columns:
            if column != "Graph":
                differences[column] = verify_significant_differences(control_metrics[column], patient_metrics[column])
    else:
        differences = pd.DataFrame(columns=["Node", "Closeness", "Clustering", "Degree"])
        for node in range(1,117):
            differences.loc[node - 1, "Node"] = node
            node_control_metrics = control_metrics[control_metrics["Node"] == node]
            node_patient_metrics = patient_metrics[patient_metrics["Node"] == node]

            for column in control_metrics.columns:
                if column not in ["Node", "Graph"]:
                    differences.loc[node - 1, column] = verify_significant_differences(node_control_metrics[column],
                                                                                       node_patient_metrics[column])

    return differences


def verify_significant_differences(first_group, second_group):
    """
    Perform statistical tests to verify significant differences between two groups.

    Args:
        first_group (pd.Series): Data for the first group.
        second_group (pd.Series): Data for the second group.

    Returns:
        bool: True if there is a significant difference, False otherwise.
    """
    if is_normal_distribution(first_group) and is_normal_distribution(second_group):
        if are_variances_similar(first_group, second_group):
            t_stat, p_value = ttest_ind(first_group, second_group, equal_var=True)
        else:
            t_stat, p_value = ttest_ind(first_group, second_group, equal_var=False)
    else:
        mwu_stat, p_value = mannwhitneyu(first_group, second_group)

    return p_value < 0.05


def is_normal_distribution(group):
    """
    Check if a group follows a normal distribution using the Shapiro-Wilk test.

    Args:
        group (pd.Series): Data to test.

    Returns:
        bool: True if the distribution is normal, False otherwise.
    """
    _, p_value = shapiro(group)

    return p_value >= 0.05


def are_variances_similar(first_group, second_group, test_type=levene):
    """
    Check if the variances of two groups are similar using Levene's test.

    Args:
        first_group (pd.Series): Data for the first group.
        second_group (pd.Series): Data for the second group.
        test_type (function): Statistical test to use for variance comparison.

    Returns:
        bool: True if variances are similar, False otherwise.
    """
    _, p_value = test_type(first_group, second_group)

    return p_value >= 0.05
