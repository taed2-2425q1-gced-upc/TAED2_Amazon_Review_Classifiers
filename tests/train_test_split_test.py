"""
This module contains unit tests for the train_test_split functionality,
which is responsible for reading data from files, writing data to files,
shuffling the dataset, and splitting it into training and testing sets.

Tests include:

- Mocking file reading operations to provide sample data for testing
  the reading functionality.
- Validating the correctness of the `read_data` function by checking
  that the DataFrame returned matches the expected structure and content.
- Ensuring that the `write_data` function correctly writes the DataFrame
  content to a file and that the output matches the expected format.
- Testing the shuffling functionality to confirm that the dataset can
  be shuffled without losing any data or changing its structure.
- Verifying that the `train_test_split` function accurately splits the
  dataset into specified training and testing sets, ensuring all
  original data is accounted for in the output.

Fixtures:
- mock_data_file: Mocks the file reading operation to simulate reading
  sample data for testing the read_data function.

Test Framework:
- Pytest: Utilized for structuring and executing the tests.
- Pandas: Used for data manipulation and validation in the tests.

Sample Data:
- The sample data used in testing consists of labeled product reviews
  to simulate real input data for the reading and splitting processes.

"""


import sys
from pathlib import Path
from unittest.mock import mock_open, patch
from sklearn.model_selection import train_test_split
import pytest
import pandas as pd

# Add src folder manually to sys.path if not already there
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the modules after the path adjustments
from src.modeling.train_test_split import read_data, write_data

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

# Sample data to mock the file read.
SAMPLE_DATA = """1 This product is great!
0 I did not like this product at all.
1 Amazing quality and fast shipping.
0 Poor quality, broke after one use.
"""

# Expected dataframe after read_data
expected_df = pd.DataFrame({
    'Label': ['1', '0', '1', '0'],
    'Review': [
        "This product is great!",
        "I did not like this product at all.",
        "Amazing quality and fast shipping.",
        "Poor quality, broke after one use."
    ]
})


@pytest.fixture
def mock_data_file():
    """Fixture to mock open() for reading data."""
    with patch("builtins.open", mock_open(read_data=SAMPLE_DATA)) as mock_file:
        yield mock_file


def test_read_data(mock_data_file):
    """Test the read_data function."""
    # Test reading the sample data
    df = read_data("dummy_path.txt")  # We pass any path since it's mocked

    # Check that the dataframe matches the expected dataframe
    pd.testing.assert_frame_equal(df, expected_df)


def test_write_data(tmpdir):
    """Test the write_data function."""
    # Using pytest's tmpdir to create a temporary directory and file for testing
    test_output_file = tmpdir.join("output.txt")

    # Write the expected dataframe to the temporary file
    write_data(expected_df, str(test_output_file))

    # Read the file back to check the content
    with open(test_output_file, "r", encoding="utf-8") as f:
        written_data = f.read()

    # Compare the written data with the expected output
    expected_output = "1 This product is great!\n" \
                      "0 I did not like this product at all.\n" \
                      "1 Amazing quality and fast shipping.\n" \
                      "0 Poor quality, broke after one use.\n"

    assert written_data == expected_output


def test_shuffle_data():
    """Test that the dataset is shuffled correctly."""
    # Make sure shuffling the dataset doesn't lose any data or change structure
    shuffled_df = expected_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Check that the shuffled dataframe has the same elements, but in a different order
    assert len(shuffled_df) == len(expected_df)
    assert sorted(shuffled_df['Label']) == sorted(expected_df['Label'])
    assert sorted(shuffled_df['Review']) == sorted(expected_df['Review'])
    assert not shuffled_df.equals(expected_df)  # The order should be different


def test_train_test_split():
    """Test that the data is split into train and test sets correctly."""
    global expected_df  # Declare expected_df as global to modify it

    # Splitting the DataFrame
    train_df, test_df = train_test_split(expected_df, test_size=0.1, random_state=42)

    # Check the sizes of the train and test sets
    assert len(train_df) == 3  # 90% of 4 samples
    assert len(test_df) == 1  # 10% of 4 samples

    # Combine the DataFrames to ensure content validity
    combined_df = pd.concat([train_df, test_df]).reset_index(drop=True)

    # Print DataFrames for debugging
    print("Combined DataFrame:")
    print(combined_df)
    print("\nExpected DataFrame:")
    print(expected_df)

    # Check if all rows in combined_df are in expected_df
    for index, row in combined_df.iterrows():
        # Check if the row exists in expected_df
        text = "in combined_df does not exist in expected_df."
        assert any((expected_df == row).all(axis=1)), f"Row {index}"+text

        # Remove the matched row from expected_df
        expected_df = expected_df[~((expected_df == row).all(axis=1))]

    # Check that there are no remaining rows in expected_df
    assert expected_df.empty, "There are remaining rows in expected_df that are not in combined_df."

    print("All rows in combined_df are accounted for in expected_df, and no extra rows remain.")
