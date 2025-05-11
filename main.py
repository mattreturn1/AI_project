from computing.brain_metrics_extractor import BrainMetricsExtractor
from dataset import folders_organizer
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define base paths
path = Path()
metadata = path / "dataset" / "metadata"
abide_dir = path / "dataset" / "abide"
analysis_dir = path / "analysis"
roi_file = metadata / "roi.json"
metadata_file = metadata / "ABIDE_metadata.csv"

def organize_folders():
    """
    Organize dataset folders for ABIDE if not already organized.
    """
    if not abide_dir.exists():
        logging.info("Organizing ABIDE dataset folders...")
        folders_organizer.process_csv(metadata / "ABIDE_metadata.csv", source="abide")
    else:
        logging.info("ABIDE dataset folders are already organized.")

def extract_all_metrics(input_dir, output_base_dir, roi):
    """
    Extract metrics for each group and age category using BrainMetricsExtractor.
    """
    for age_group in input_dir.iterdir():
        if not age_group.is_dir():
            continue

        for group in age_group.iterdir():
            output_group = group.relative_to("dataset")
            output_path = output_base_dir / output_group

            if not output_path.exists():
                logging.info(f"Extracting metrics for {group}...")

                extractor = BrainMetricsExtractor(
                    input_dir=group,
                    output_dir=output_path,
                    roi_file=roi,
                    metadata_file=metadata
                )
                extractor.extract_metrics()
            else:
                logging.info(f"Metrics for {group} already extracted.")

def main():
    logging.info("Pipeline started.")
    organize_folders()
    extract_all_metrics(abide_dir, analysis_dir, roi_file)
    logging.info("Pipeline completed.")

if __name__ == "__main__":
    main()


