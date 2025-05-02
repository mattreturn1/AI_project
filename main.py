import logging
from dataset import folders_organizer
from computing import brain_metrics_extractor, networks_comparator, statistical_analysis
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

path = Path()
metadata = path / "dataset" / "metadata"
abide_dir = path / "dataset" / "abide"
ppmi_dir = path / "dataset" / "ppmi"
analysis_dir = path / "analysis"

def organize_folders():
    """If necessary, organizes data in specific folders."""
    if not abide_dir.exists() and not ppmi_dir.exists():
        logging.info("Organizing dataset folders...")
        folders_organizer.process_csv(metadata / "ABIDE_metadata.csv", "abide")
        folders_organizer.process_csv(metadata / "PPMI_metadata.csv", "ppmi")
    else:
        logging.info("Dataset folders are already organized.")


def extract_all_metrics(input_dir, output_base_dir):
    """Extracts the metrics for each group."""
    for age_group in input_dir.iterdir():
        if not age_group.is_dir():
            continue

        for group in age_group.iterdir():
            output_group = group.relative_to("dataset")
            output_path = output_base_dir / output_group
            if not output_path.exists():
                logging.info(f"Extracting metrics for {group}...")
                brain_metrics_extractor.extract_metrics(group, output_path)
            else:
                logging.info(f"Metrics for {group} already extracted.")


def analyze_groups(input_dir, group_names, comparison_folder):
    """Compares and analyzes groups."""
    for age_group in input_dir.iterdir():
        if not age_group.is_dir():
            continue

        comparison_path = age_group / comparison_folder
        if not comparison_path.exists():
            logging.info(f"Comparing and analyzing groups in {age_group}...")
            networks_comparator.compare_groups(age_group, group_names, comparison_path)
            statistical_analysis.compare_groups(age_group, group_names, comparison_path)
        else:
            logging.info(f"Comparison for {age_group} already completed.")


def main():
    logging.info("Pipeline started.")
    organize_folders()

    # Compute metrics
    extract_all_metrics(abide_dir, analysis_dir)
    extract_all_metrics(ppmi_dir, analysis_dir)

    # Analyze metrics
    analyze_groups(analysis_dir / "abide", ["control", "patient"], "comparison")
    analyze_groups(analysis_dir / "ppmi", ["control", "pd", "prodromal", "swedd"], "comparison")

    logging.info("Pipeline completed.")


if __name__ == "__main__":
    main()

