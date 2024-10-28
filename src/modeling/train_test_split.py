"""
This module processes an Amazon review dataset by reading, shuffling, splitting, and saving the data
into training and testing sets. It reads a raw text file containing labeled reviews, converts the
data into a pandas DataFrame, splits the dataset, and saves the processed data to text files.

The module supports the following functionalities:
- Reading a labeled Amazon review dataset in text format.
- Converting the dataset into a DataFrame with 'Label' and 'Review' columns.
- Shuffling the dataset to ensure randomness.
- Splitting the data into training and testing sets.
- Saving the processed train and test sets to text files.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger
import typer

# setting path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.config import DATA_DIR, RAW_DATA_DIR
from src import utilities

app = typer.Typer()

def read_data(path: Path) -> pd.DataFrame:
    """
    Reads a dataset from a given file path in .txt format and returns a DataFrame.

    Args:
        path (Path): The file path to the dataset in .txt format.

    Returns:
        pd.DataFrame: A DataFrame containing two columns: 'Label' and 'Review',
                      where 'Label' represents the labels, and 'Review' contains
                      the corresponding text reviews.
    """
    logger.info(f"Reading data from {path}...")

    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    labels, reviews = [], []

    for line in lines:
        split_line = line.strip().split(' ', 1)
        labels.append(split_line[0])
        reviews.append(split_line[1])

    df = pd.DataFrame({'Label': labels, 'Review': reviews})

    logger.info(f"Data reading complete. Total records: {len(df)}")
    return df

def write_data(data: pd.DataFrame, output_path: Path) -> None:
    """
    Writes a dataset to a .txt file at the specified output path.

    Args:
        data (pd.DataFrame): The dataset containing 'Label' and 'Review' columns
                             to be written to a file.
        output_path (Path): The file path where the dataset will be saved in .txt format.

    Returns:
        None
    """
    logger.info(f"Writing data to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in data.iterrows():
            try:
                clean_text = row['Review'].strip('"""')
            except (AttributeError, KeyError):
                clean_text = "N/A"  # Default for missing 'Review' column

            f.write(f"{row['Label']} {clean_text}\n")

    logger.info(f"Data written to {output_path}")

@app.command()
def main() -> None:
    """
    Main function to process and split the Amazon review dataset.

    Returns:
        None
    """
    logger.info("Retrieving Params file.")
    params = utilities.get_params(root_dir)

    # Dataset paths
    dataset_path = DATA_DIR / params["full_dataset"]
    train_path = RAW_DATA_DIR / params["train_dataset"]
    test_path = RAW_DATA_DIR / params["test_dataset"]

    test_size: float = 0.1
    random_state: int = 42

    logger.info("Starting dataset processing...")

    dataset = read_data(dataset_path)

    logger.info("Shuffling dataset...")
    dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

    logger.info(f"Splitting dataset into train and test sets (test size = {test_size})...")
    train, test = train_test_split(dataset, test_size=test_size, random_state=random_state)

    logger.info("Saving train and test sets...")
    write_data(train, train_path)
    write_data(test, test_path)

    logger.info("Dataset processing complete.")

if __name__ == "__main__":
    app()
