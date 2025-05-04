from computing import metrics_computator
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def create_directory(directory_path):
    """
    Ensure the directory for the given file path exists.
    If it doesn't exist, create it (including all necessary parent directories).
    """
    directory = Path(directory_path)
    directory.mkdir(parents=True, exist_ok=True)
    logging.info(f"Directory {directory_path} created.")


class BrainMetricsExtractor:
    """
    Extracts region-level brain network metrics from .mat files and saves them to disk.
    """
    def __init__(self, input_dir, output_dir, roi_file):
        self.input_directory = Path(input_dir)
        self.output_directory = Path(output_dir)
        self.roi = self.load_roi(roi_file)
        self.abide_roi_metrics = self.initialize_abide_roi_metrics()

    def load_roi(self, roi_file):
        """
        Load the ROI definition from a JSON file.

        Parameters:
            roi_file (str or Path): Path to the JSON file containing ROI definitions.

        Returns:
            dict: A dictionary mapping region names to lists of subregion dictionaries.
        """
        roi_path = Path(roi_file)
        if not roi_path.exists():
            logging.error(f"ROI file {roi_file} not found.")
            raise FileNotFoundError(f"ROI file '{roi_file}' does not exist.")

        try:
            with open(roi_path, 'r') as file:
                roi_data = json.load(file)

            if not isinstance(roi_data, dict):
                logging.error("Invalid ROI file format. Expected a JSON object at the top level.")
                raise ValueError("Invalid ROI file format.")

            return roi_data

        except json.JSONDecodeError as exc:
            logging.error(f"Error decoding JSON file {roi_file}: {exc}")
            raise

    def initialize_abide_roi_metrics(self):
        """
        Initialize an empty list for each metric under each region name.
        """
        abide_roi_metrics = {}
        for region in self.roi:
            abide_roi_metrics[region] = {
                "closeness": [],
                "clustering": [],
                "degree": []
            }

        return abide_roi_metrics

    def extract_metrics(self):
        """
        Compute graph metrics for each subject and aggregate them by ROI.
        """
        logging.info("Computing graph metrics for each subject...")

        if not self.input_directory.exists() or not self.input_directory.is_dir():
            logging.error(f"Directory {self.input_directory} does not exist or is not a directory.")

        mat_files = list(self.input_directory.glob("*.mat"))
        if not mat_files:
            logging.warning(f"No .mat files found in directory {self.input_directory}.")

        for file in mat_files:
            self.process_file(file)

        self.save_results()

    def process_file(self, file):
        """
        Process a single file to compute metrics and update the metrics containers.
        """
        try:
            logging.info(f"Processing file: {file.parent}/{file.name}")
            brain_network = from_matrix_to_network(file)

            # Compute node-level metrics
            closeness_centrality = metrics_computator.compute_closeness_centrality(brain_network)
            clustering_coefficient = metrics_computator.compute_clustering_coefficients(brain_network)
            degree_centrality = metrics_computator.compute_degree_centrality(brain_network)

            # Region-level metrics
            for region, subregions in self.roi.items():
                ids = [sub["id"] for sub in subregions]

                # Compute mean only for nodes present in the graph
                closeness_vals = [closeness_centrality[i] for i in ids if i in closeness_centrality]
                clustering_vals = [clustering_coefficient[i] for i in ids if i in clustering_coefficient]
                degree_vals = [degree_centrality[i] for i in ids if i in degree_centrality]

                # Append mean values to abide_roi_metrics
                self.abide_roi_metrics[region]["closeness"].append(np.mean(closeness_vals))
                self.abide_roi_metrics[region]["clustering"].append(np.mean(clustering_vals))
                self.abide_roi_metrics[region]["degree"].append(np.mean(degree_vals))

        except Exception as exc:
            logging.error(f"Error processing file {file.parent}/{file.name}: {exc}")

    def save_results(self):
        """
        Save metrics to CSV files.
        """
        metrics_dir = self.output_directory / "metrics"
        create_directory(metrics_dir)
        output_file = metrics_dir / "abide_roi_metrics.csv"
        self.save_abide_roi_metrics(output_file)

    def save_abide_roi_metrics(self, output_file):
        """
        Save region-level metrics to a CSV file.
        """
        data = []
        for region, metrics in self.abide_roi_metrics.items():
            num_subjects = len(metrics["closeness"])
            for i in range(num_subjects):
                data.append({
                    "Subject": i + 1,
                    "Region": region,
                    "Closeness": metrics["closeness"][i],
                    "Clustering": metrics["clustering"][i],
                    "Degree": metrics["degree"][i]
                })

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        logging.info(f"Saved ROI metrics to {output_file}")
