import sys
from pathlib import Path

# Print the current sys.path
print("sys.path before modification:", sys.path)

# Add src folder manually to sys.path if not already there
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the modules after the path adjustments
from src.modeling.train_test_split import read_data, write_data

import pytest
import pandas as pd
from unittest.mock import mock_open, patch
from sklearn.model_selection import train_test_split

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

# Sample data to mock the file read.
sample_data = """1 This product is great!
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
    with patch("builtins.open", mock_open(read_data=sample_data)) as mock_file:
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
        assert any((expected_df == row).all(axis=1)), f"Row {index} in combined_df does not exist in expected_df."

        # Remove the matched row from expected_df
        expected_df = expected_df[~((expected_df == row).all(axis=1))]

    # Check that there are no remaining rows in expected_df
    assert expected_df.empty, "There are remaining rows in expected_df that are not in combined_df."

    print("All rows in combined_df are accounted for in expected_df, and no extra rows remain.")

