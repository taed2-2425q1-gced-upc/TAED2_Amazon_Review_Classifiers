"""
This module processes an Amazon review dataset by reading, splitting, and saving the data into train 
and test sets. It reads a raw text file containing labeled reviews, converts the data into a pandas 
DataFrame, shuffles the dataset, and splits it into training and testing sets. The processed data is 
saved as text files.

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

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.config import DATA_DIR

# Dataset paths.
dataset_path = DATA_DIR / "raw/amazon_reviews_sample.txt"
train_path = DATA_DIR / "split/train.txt"
test_path = DATA_DIR / "split/test.txt"

# Load the data.
def read_data(path):
    """
    Read a dataset from a given file path in .txt format and return a DataFrame.

    Args:
        path (str): The file path to the dataset in .txt format.

    Returns:
        pd.DataFrame: A DataFrame containing two columns: 'Label' and 'Review',
                      where 'Label' represents the labels, and 'Review' contains
                      the corresponding text reviews.
    """
    # Reads the dataset.
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    labels = []
    reviews = []

    # Splits the lines of the dataset into labels and reviews.
    for line in lines:
        split_line = line.strip().split(' ', 1)
        labels.append(split_line[0])
        reviews.append(split_line[1])

    # Defines a dataframe with the labels and reviews.
    df = pd.DataFrame(
        {
            'Label': labels,
            'Review': reviews
        }
    )

    return df

def write_data(data, output_path):
    """
    Write a dataset to a .txt file at the specified output path.

    Args:
        data (pd.DataFrame): The dataset containing 'Label' and 'Review' columns
        to be written to a file.
        output_path (str): The file path where the dataset will be saved in .txt format.

    Returns:
        None
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in data.iterrows():
            # Write label, space, and the corresponding text
            try:
                clean_text = row['Review'].strip('"""')
            except AttributeError:  # Handles cases where 'Review' isn't a string
                clean_text = row['Review']
            except KeyError:  # Handles cases where 'Review' key is missing
                clean_text = "N/A"  # Assign a default value if the key is missing

            f.write(f"{row['Label']} {clean_text}\n")

dataset = read_data(dataset_path) # Labels are balanced.

# Suffles the dataset.
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Splits the dataset into train and test datasets.
train, test = train_test_split(dataset, test_size=0.1, random_state=42)

# Saves the train and test datasets.
write_data(train, train_path)
write_data(test, test_path)
