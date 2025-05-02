import shutil
from pathlib import Path
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Main function that sort the needed files
def process_csv(file_path, source):
    try:
        # Load the data in a DataFrame
        df = pd.read_csv(file_path)
        logging.info(f"Caricato CSV: {file_path}")

        # Filter the string containing fMRI and divide control and patient
        df = df[df['Modality'] == 'fMRI']
        df_control = df[df['Group'] == 'Control']
        df_patient = df[df['Group'] != 'Control']

        for _, row in df_control.iterrows():
            folder = find_folder_by_substring(str(row['Subject']), source)
            if not folder:
                continue
            file = search_files_in_folder(folder)

            if source == 'abide':
                age_group = get_age_group_abide(row['Age'])
            else:
                age_group = get_age_group_ppmi(row['Age'])

            if file:
                move_file_from_to(str(folder), f"dataset/{source}/" + age_group + "/control", file.name)

        for _, row in df_patient.iterrows():
            folder = find_folder_by_substring(str(row['Subject']), source)
            if not folder:
                continue
            file = search_files_in_folder(folder)

            if source == 'abide':
                age_group = get_age_group_abide(row['Age'])
            else:
                age_group = get_age_group_ppmi(row['Age'])

            if file and source == "abide":
                move_file_from_to(str(folder), f"dataset/{source}/" + age_group + "/patient", file.name)
            elif file and source == "ppmi":
                if row["Group"] == "PD":
                    move_file_from_to(str(folder), f"dataset/{source}/" + age_group + "/pd", file.name)
                elif row["Group"] == "Prodromal":
                    move_file_from_to(str(folder), f"dataset/{source}/" + age_group + "/prodromal", file.name)
                elif row["Group"] == "SWEDD":
                    move_file_from_to(str(folder), f"dataset/{source}/" + age_group + "/swedd", file.name)

    except Exception as exc:
        logging.error(f"Error during the process of CSV file '{file_path}': {exc}")

# Move a specific file from a folder to another one
def move_file_from_to(source_folder, destination_folder, filename):
    if not isinstance(filename, str) or not isinstance(source_folder, str):
        logging.warning("The parameters are not strings.")
        return

    source_file = Path(source_folder) / filename
    if not source_file.exists():
        logging.warning(f"File '{filename}' not found in the directory '{source_folder}'.")
        return

    destination_folder = Path(destination_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)
    destination_file = destination_folder / filename

    try:
        shutil.move(str(source_file), str(destination_file))
        logging.info(f"File '{filename}' move to '{destination_folder}'.")
    except Exception as exc:
        logging.error(f"Error during the repositioning of file '{filename}': {exc}")

# Search a specific folder using a subtring
def find_folder_by_substring(substring, source):
    try:
        source_path = Path(source)
        for item in source_path.iterdir():
            if item.is_dir() and substring in item.name:
                return item
        logging.warning(f"Directory with substring '{substring}' not found in '{source}'.")
    except Exception as exc:
        logging.error(f"Error during the searching of the directory: {exc}")

# Search a specific file inside a folder using a subtring
def search_files_in_folder(folder_path):
    folder_path = Path(folder_path)
    for file in folder_path.rglob('*'):
        if 'AAL116_correlation_matrix' in file.name:
            return file
    logging.warning(f"File with substring 'AAL116_correlation_matrix' not found in '{folder_path}'.")

# Help function to filter the groups by age for ABIDE
def get_age_group_abide(age):
    if age < 12:
        return '11-'
    elif age < 18:
        return '12_17'
    elif age < 26:
        return '18_25'
    else:
        return '25+'

# Help function to filter the groups by age for PPMI
def get_age_group_ppmi(age):
    if age < 61:
        return '60-'
    elif age < 71:
        return '60_70'
    else:
        return '70+'
