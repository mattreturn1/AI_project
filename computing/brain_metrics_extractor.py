from computing import metrics_computator
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_metrics(input_directory_path, output_directory_path):
    """
    Extract global and node-level metrics from brain network files in a directory.
    """
    input_directory = Path(input_directory_path)
    output_directory = Path(output_directory_path)

    if not input_directory.exists() or not input_directory.is_dir():
        logging.error(f"Directory {input_directory_path} does not exist or is not a directory.")

    mat_files = list(input_directory.glob("*.mat"))
    if not mat_files:
        logging.warning(f"No .mat files found in directory {input_directory_path}.")

    graph_metrics, node_metrics = initialize_metrics()

    for file in mat_files:
        process_file(file, graph_metrics, node_metrics)

    save_results(graph_metrics, node_metrics, output_directory)


def initialize_metrics():
    """
    Initialize containers for graph and node metrics.
    """
    graph_metrics = {
        "closeness": [],
        "clustering": [],
        "degree": []
    }

    node_metrics = {
        "closeness": [[] for _ in range(116)],
        "clustering": [[] for _ in range(116)],
        "degree": [[] for _ in range(116)]
    }

    return graph_metrics, node_metrics


def process_file(file, graph_metrics, node_metrics):
    """
    Process a single file to compute metrics and update the metrics containers.
    """
    try:
        logging.info(f"Processing file: {file.parent}/{file.name}")
        brain_network = from_matrix_to_network(file)

        closeness_centrality = metrics_computator.compute_closeness_centrality(brain_network)
        clustering_coefficient = metrics_computator.compute_clustering_coefficients(brain_network)
        degree_centrality = metrics_computator.compute_degree_centrality(brain_network)

        # Aggregate global metrics
        graph_metrics["closeness"].append(np.mean(list(closeness_centrality.values())))
        graph_metrics["clustering"].append(np.mean(list(clustering_coefficient.values())))
        graph_metrics["degree"].append(np.mean(list(degree_centrality.values())))

        # Aggregate node-level metrics
        for i, node in enumerate(brain_network.nodes):
            node_metrics["closeness"][i].append(closeness_centrality[node])
            node_metrics["clustering"][i].append(clustering_coefficient[node])
            node_metrics["degree"][i].append(degree_centrality[node])

    except Exception as exc:
        logging.error(f"Error processing file {file.parent}/{file.name}: {exc}")


def from_matrix_to_network(file_path):
    """
    Convert a correlation matrix into a weighted network graph.
    """
    try:
        matrix = metrics_computator.load_and_process_matrix(file_path)
        return metrics_computator.create_weighted_graph(matrix)
    except Exception as exc:
        logging.error(f"Error converting matrix to network for file {file_path}: {exc}")
        raise


def save_results(graph_metrics, node_metrics, directory):
    """
    Save metrics and statistics to CSV files.
    """
    create_directory(directory / "metrics")
    create_directory(directory / "stats")

    save_graph_metrics(graph_metrics, directory / "metrics" / "graph_metrics.csv")
    save_node_metrics(node_metrics, directory / "metrics" / "node_metrics.csv")

    graph_statistics = compute_graph_statistics(graph_metrics)
    node_statistics = compute_node_statistics(node_metrics)

    save_graph_statistics(graph_statistics, directory / "stats" / "graph_statistics.csv")
    save_node_statistics(node_statistics, directory / "stats" / "node_statistics.csv")


def save_graph_metrics(graph_metrics, output_file):
    """
    Save graph metrics to a CSV file.
    """
    #create_directory(output_file)

    data = []
    for graph in range(len(graph_metrics["closeness"])):
        data.append({
            "Graph": graph + 1,
            "Closeness": graph_metrics["closeness"][graph],
            "Clustering": graph_metrics["clustering"][graph],
            "Degree": graph_metrics["degree"][graph]
        })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)


def save_node_metrics(node_metrics, output_file):
    """
    Save node-level metrics to a CSV file.
    """
    #create_directory(output_file)

    data = []
    for node in range(116):
        for graph in range(len(node_metrics["closeness"][node])):
            data.append({
                "Node": node + 1,
                "Graph": graph + 1,
                "Closeness": node_metrics["closeness"][node][graph],
                "Clustering": node_metrics["clustering"][node][graph],
                "Degree": node_metrics["degree"][node][graph]
            })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)


def compute_graph_statistics(graph_metrics):
    """
    Compute statistics (mean, median, std) for global metrics.
    """
    return {
        metric: {
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values)
        }
        for metric, values in graph_metrics.items()
    }


def compute_node_statistics(node_metrics):
    """
    Compute statistics (mean, median, std) for node-level metrics.
    """
    return {
        metric: {
            "mean": [np.mean(values) for values in node_metrics[metric]],
            "median": [np.median(values) for values in node_metrics[metric]],
            "std": [np.std(values) for values in node_metrics[metric]]
        }
        for metric in node_metrics
    }


def save_graph_statistics(graph_statistics, output_file):
    """
    Save graph statistics to a CSV file.
    """
    #create_directory(output_file)

    data = []
    for metric in graph_statistics.keys():
        data.append({
            "Metric": metric,
            "Mean": graph_statistics[metric]["mean"],
            "Median": graph_statistics[metric]["median"],
            "Standard Deviation": graph_statistics[metric]["std"]
        })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)


def save_node_statistics(node_statistics, output_file):
    """
    Save node-level statistics to a CSV file.
    """
    #create_directory(output_file)

    data = []
    for node in range(116):
        for metric in node_statistics.keys():
            data.append({
                "Node": node + 1,
                "Metric": metric,
                "Mean": node_statistics[metric]["mean"][node],
                "Median": node_statistics[metric]["median"][node],
                "Standard Deviation": node_statistics[metric]["std"][node]
            })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)


def create_directory(directory_path):
    """
    Ensure the directory for the given file path exists.
    If it doesn't exist, create it (including all necessary parent directories).
    """
    directory = Path(directory_path)
    directory.mkdir(parents=True, exist_ok=True)
    logging.info(f"Directory {directory_path} created.")
