from pathlib import Path
import logging
import pandas as pd
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_csv(file_path, source='abide'):
    """
    Processes a CSV file to organize fMRI files by age group and diagnostic group.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded CSV: {file_path}")

        # Filter only fMRI entries
        df = df[df['Modality'] == 'fMRI']
        df_control = df[df['Group'] == 'Control']
        df_patient = df[df['Group'] != 'Control']

        for _, row in df_control.iterrows():
            folder = find_folder_by_substring(str(row['Subject']), source)
            if not folder:
                continue
            file = search_files_in_folder(folder)

            age_group = get_age_group_abide(row['Age'])
            if file:
                move_file_from_to(str(folder), f"dataset/{source}/{age_group}/control", file.name)

        for _, row in df_patient.iterrows():
            folder = find_folder_by_substring(str(row['Subject']), source)
            if not folder:
                continue
            file = search_files_in_folder(folder)

            age_group = get_age_group_abide(row['Age'])
            if file:
                move_file_from_to(str(folder), f"dataset/{source}/{age_group}/patient", file.name)

    except Exception as exc:
        logging.error(f"Error processing CSV file '{file_path}': {exc}")

def move_file_from_to(source_folder, destination_folder, filename):
    """
    Moves a file from the source folder to the destination folder.
    Creates the destination folder if it doesn't exist.
    """
    if not isinstance(filename, str) or not isinstance(source_folder, str):
        logging.warning("The parameters are not strings.")
        return

    source_file = Path(source_folder) / filename
    if not source_file.exists():
        logging.warning(f"File '{filename}' not found in directory '{source_folder}'.")
        return

    destination_folder = Path(destination_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)
    destination_file = destination_folder / filename

    try:
        shutil.move(str(source_file), str(destination_file))
        logging.info(f"Moved '{filename}' to '{destination_folder}'.")
    except Exception as exc:
        logging.error(f"Error moving file '{filename}': {exc}")

def find_folder_by_substring(substring, source):
    """
    Searches for a folder under the source directory that contains the given substring.
    """
    try:
        source_path = Path(source)
        for item in source_path.iterdir():
            if item.is_dir() and substring in item.name:
                return item

        logging.warning(f"Folder with substring '{substring}' not found in '{source}'.")
        return None
    except Exception as exc:
        logging.error(f"Error searching for folder: {exc}")
        return None


def search_files_in_folder(folder_path):
    """
    Recursively searches for a file containing 'AAL116_correlation_matrix' in the folder.
    """
    folder_path = Path(folder_path)
    for file in folder_path.rglob('*'):
        if 'AAL116_correlation_matrix' in file.name:
            return file

    logging.warning(f"File with substring 'AAL116_correlation_matrix' not found in '{folder_path}'.")
    return None


def get_age_group_abide(age):
    """
    Determines age group for ABIDE dataset.
    """
    if age < 12:
        return '11-'
    elif age < 18:
        return '12_17'
    elif age < 26:
        return '18_25'
    else:
        return '25+'
