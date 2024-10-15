# Imports and settings.
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.config import DATA_DIR

# Dataset paths.
dataset_path = DATA_DIR / "raw/amazon_reviews_sample.txt"
train_path = DATA_DIR / "split/train.txt"
test_path = DATA_DIR / "split/test.txt"

# Load the data.
def read_data(dataset_path):
    """
    Given a path to a dataset, in .txt format, reads the dataset and returns two dataframes,
    one for the text reviews and one for the associated labels.
    """
    # Reads the dataset.
    with open(dataset_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    labels = []
    reviews = []

    # Splits the lines of the dataset into labels and reviews.
    for line in lines:
        split_line = line.strip().split(' ', 1)
        labels.append(split_line[0])
        reviews.append(split_line[1])
    
    # Defines a dataframe with the labels and reviews.
    dataset = pd.DataFrame(
        {
            'Label': labels,
            'Review': reviews
        }
    )

    return dataset

# Writes the dataset.
def write_data(dataset, output_path):
    """
    Given a dataset and the output path, writes the dataset to a .txt file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for index, row in dataset.iterrows():
            # Write label, space, and the corresponding text
            try:
                clean_text = row['Review'].strip('"""')
            except:
                clean_text = row['Review']

            f.write(f"{row['Label']} {clean_text}\n")

dataset = read_data(dataset_path) # Labels are balanced.

# Suffles the dataset.
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Splits the dataset into train and test datasets.
train, test = train_test_split(dataset, test_size=0.1, random_state=42)

# Saves the train and test datasets.
write_data(train, train_path)
write_data(test, test_path)