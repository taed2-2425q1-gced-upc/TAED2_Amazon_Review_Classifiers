# Imports and settings.
import pandas as pd
from sklearn.model_selection import train_test_split

# Data paths.
train_dataset_path = 'original/train.txt'
test_dataset_path = 'original/test.txt'

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
        label = 1 if split_line[0] == '__label__2' else 0
        review = split_line[1]
        labels.append(label)
        reviews.append(review)
    
    # Defines a dataframe with the labels and reviews.
    dataset = pd.DataFrame(
        {
            'Label': labels,
            'Review': reviews
        }
    )

    return dataset

train_dataset = read_data(train_dataset_path) # Contains 3600000 entries. Labels are balanced.
test_dataset = read_data(test_dataset_path) # Contains 400000 entries. Labels are balanced.

# Merges the train and test datasets.
dataset = pd.concat([train_dataset, test_dataset], ignore_index=True)

# Suffles the dataset.
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Splits the dataset into train and test datasets.
train, test = train_test_split(dataset, test_size=0.1, random_state=42)

# Saves the train and test datasets.
train.reset_index(drop=True, inplace=True)
train.to_csv('resampling/train.txt', sep=' ', index=False)

test.reset_index(drop=True, inplace=True)
test.to_csv('resampling/test.txt', sep=' ', index=False)
