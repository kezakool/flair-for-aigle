import re
from datetime import datetime
import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)

def concat_df_parquet_files(folder_path, pattern=r"df_set_results_tmp_raster_"):
    """
    Concatenates all Parquet files in a folder that match a given pattern into a single DataFrame.
    
    :param folder_path: Path to the folder containing Parquet files.
    :param pattern: Regex pattern to match filenames.
    :return: Concatenated pandas DataFrame.
    """
    parquet_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if pattern in f]
        
    if not parquet_files:
        raise FileNotFoundError("No matching Parquet files found.")

    df_list = [pd.read_parquet(file) for file in parquet_files]
    
    return pd.concat(df_list, ignore_index=True)


def find_first_four_digits(text):
    """
    Finds the first sequence of exactly four consecutive digits in a string.

    Args:
        text (str): The input string to search.

    Returns:
        str: The first sequence of four consecutive digits found, or None if no such sequence exists.
    """
    # Regular expression pattern to find four consecutive digits
    pattern = r'\d{4}'
    # Search for the pattern in the text
    match = re.search(pattern, text)
    if match:
        return match.group()  # Returns the matched string
    else:
        return None

def correct_string(s):
    """_summary_
    Utilise une regex pour trouver les quatre 
    séquences de chiffres séparées par des tirets suivies d'un underscore et les remplace par 
    les mêmes séquences de chiffres mais avec les tirets remplacés par des points.
    Args:
        s (_type_): _description_

    Returns:
        _type_: _description_
    """
    return re.sub(r"(\d+)-(\d+)-(\d+)-(\d+)_", r"\1.\2-\3.\4.", s)

def remove_files_except_with_substrings(folder_path, substring1, substring2):
    # List all files in the folder
    try:
        files = os.listdir(folder_path)
    except FileNotFoundError:
        logger.info(f"The folder {folder_path} does not exist.")
        return

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        # Check if it's a file and not a directory
        if os.path.isfile(file_path):
            # Check if the file name contains both substrings
            if substring1 in file_name and substring2 in file_name:
                logger.debug(f"Keeping file: {file_name}")
            else:
                logger.info(f"Removing old ckpt file : {file_name}")
                os.remove(file_path)

def generate_timestamp():
    """
    Generates a timestamp string in the format YYMMDDhhmm.
    
    Returns:
        str: The formatted timestamp.
    """

    # Get the current time
    now = datetime.now()
    
    # Format the datetime as a string 'YYMMDDhhmm'
    timestamp = now.strftime("%y%m%d%H%M")
    
    return timestamp
