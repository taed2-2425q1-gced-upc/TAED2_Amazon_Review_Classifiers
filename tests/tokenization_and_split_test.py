"""
This module contains unit tests for the functionality of the
tokenization_and_split module, which is responsible for tokenizing
text data and splitting it into training and validation sets.

Tests include:

- Mocking the reading of a training dataset from a text file.
- Verifying that the main function performs the expected operations:
  - Loads and tokenizes the training data.
  - Saves the tokenizer and the processed sequences and labels to
    appropriate files.
- Ensuring that all expected log messages are captured during
  the execution of the main function.

Fixtures:
- mock_data_file: Mocks the file reading operation for the
  training dataset, providing sample data for testing.

Test Framework:
- Pytest: Used for structuring and running the tests.
- Loguru: Utilized for logging outputs and capturing log messages
  during tests.

Sample Data:
- The sample data used in testing contains labeled product reviews,
  which are used to simulate real input data for the tokenization
  process.

"""



import sys
from pathlib import Path
from unittest.mock import mock_open, patch
import pytest
from loguru import logger

# Add the parent directory (where src is located) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the modules after adjusting the sys.path
from src.modeling.tokenization_and_split import main

# Sample data to mock the file read for the 'train_dataset.txt'
# Ensure each line has a label and a review, separated by a space.
SAMPLE_DATA = """1 This product is great!
0 I did not like this product at all.
1 Amazing quality and fast shipping.
0 Poor quality, broke after one use.
"""

@pytest.fixture
def mock_data_file():
    """Fixture to mock open() for reading data."""
    with patch("builtins.open", mock_open(read_data=SAMPLE_DATA)) as mock_file:
        yield mock_file

def test_main(mock_data_file, capsys):
    """Test the main functionality of the tokenization_and_split module."""

    lines = SAMPLE_DATA.splitlines()

    mock_data_file.return_value.__enter__.return_value.readlines.return_value = lines

    # Patching the open() call for both the params.yaml and dataset file
    with patch("pickle.dump") as mock_pickle_dump, \
         patch("src.utilities.get_params", return_value={
            "train_sequences": "train_sequences.pkl",
            "train_labels": "train_labels.pkl",
            "val_sequences": "val_sequences.pkl",
            "val_labels": "val_labels.pkl",
            "train_dataset": "train_dataset.txt",  # Using the mock data file content
            "tokenizer": "tokenizer.pkl",
            "hyperparameters": {"num_words": 10000}
         }):  # Mocking the YAML params loading

        main()

        # Check that the number of calls to pickle.dump is as expected
        assert mock_pickle_dump.call_count == 5  # Should save tokenizer and 4 pickled datasets

                # Set log level and configure loguru to output to stdout
        logger.remove()  # Remove the default logger
        logger.add(sys.stdout, level="INFO")  # Add a new logger that outputs to stdout

        # Run the main function and capture stdout/stderr
        with capsys.disabled():
            main()  # Execute the main function

        # Capture the output after calling main
        captured = capsys.readouterr()  # Capture stdout and stderr output
        print(captured.out)  # Optional: Print the captured output for debugging

        # List of all expected log messages
        expected_messages = [
            "Retrieving Params file.",
            "Loading training data and extracting labels and reviews...",
            "Tokenizing training data...",
            "Saving tokenizer...",
            "Tokenizer saved at:",
            "Splitting data into training and validation sets...",
            "Saving training and validation data...",
            "Train sequences saved at:",
            "Validation sequences saved at:",
            "Train labels saved at:",
            "Validation labels saved at:"
        ]

        # Verify all expected log messages are in the captured output
        for message in expected_messages:
            assert message in captured.out, \
                f"Expected log message not found in log output: '{message}'"
