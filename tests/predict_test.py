"""
This module contains tests for the functions responsible for predicting 
sentiment from text inputs. The tests validate the core functionality 
of sentiment prediction and the integration of model loading and 
parameter handling within the main execution flow.

Tested Functions:
- `predict_sentiment`: Tests the sentiment prediction functionality 
  of the model. It simulates model predictions for both positive and 
  negative sentiments and ensures the correct output based on the 
  predicted probabilities.

- `main`: Tests the main function of the predict module, which 
  handles loading the model and tokenizer, reading input data, 
  performing predictions, and logging the process. This function is 
  tested in conjunction with mocked external dependencies to ensure 
  that it behaves as expected when invoked.

Fixtures:
- `mock_data_file`: Provides a mocked file reading environment that 
  simulates the presence of a text file containing sample reviews. 
  This fixture is used to test the `main` function's ability to read 
  and process input data.

Example Data:
The tests utilize sample review data that represents typical input 
for the sentiment analysis model, enabling the simulation of predictions.

Usage:
To execute these tests, run the test file using pytest. Ensure that 
pytest and the necessary dependencies are installed in your environment.
"""


import sys
from pathlib import Path
from unittest.mock import mock_open, patch, MagicMock
import pytest
import numpy as np
from loguru import logger

# Add the parent directory (where src is located) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the modules after adjusting the sys.path
from src.modeling.predict import main, predict_sentiment

# Sample data to mock the file read
SAMPLE_DATA = """This product is great!
I did not like this product at all.
Amazing quality and fast shipping.
Poor quality, broke after one use.
"""

@pytest.fixture
def mock_data_file():
    """Fixture to mock open() for reading data."""
    with patch("builtins.open", mock_open(read_data=SAMPLE_DATA)) as mock_file:
        yield mock_file

def test_predict_sentiment():
    """Test sentiment prediction functionality."""
    # Mock the tokenizer and model
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    # Set up tokenizer mock
    mock_tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]

    # Mock model prediction for positive sentiment
    mock_model.predict.return_value = np.array([[0.8]])  # Simulating a positive sentiment
    sentiment, _ = predict_sentiment("This is a great product!", mock_model, mock_tokenizer)
    assert sentiment == 'Positive'

    # Mock model prediction for negative sentiment
    mock_model.predict.return_value = np.array([[0.2]])  # Simulating a negative sentiment
    sentiment, _ = predict_sentiment("This is a terrible product!", mock_model, mock_tokenizer)
    assert sentiment == 'Negative'

def test_main(mock_data_file, capsys):
    """Test the main functionality of the predict module."""
    with patch("src.modeling.predict.tf.keras.models.load_model") as mock_load_model, \
         patch("src.modeling.predict.utilities.pickle.load") as mock_pickle_load, \
         patch("src.modeling.predict.mlflow.log_param") as mock_log_param, \
         patch("src.modeling.predict.mlflow.start_run") as mock_start_run, \
         patch("src.modeling.predict.utilities.get_params") as mock_get_params:

        # Mock parameters returned by get_params
        mock_get_params.return_value = {
            'model': 'dummy_model.h5',
            'tokenizer': 'dummy_tokenizer.pkl',
            'predict_dataset': 'dummy_predict_path.txt'
        }

        mock_model = MagicMock()
        # Set the model's predict method to return a mock prediction
        mock_model.predict.return_value = [0.7]  # Mock a positive prediction (float > 0.5)

        mock_load_model.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_pickle_load.return_value = mock_tokenizer

        # Set log level and configure loguru to output to stdout
        logger.remove()  # Remove the default logger
        logger.add(sys.stdout, level="INFO")  # Add a new logger that outputs to stdout

        # Run the main function without parameters
        with capsys.disabled():
            main()  # Execute the main function

        # Capture the output after calling main
        captured = capsys.readouterr()  # Capture stdout and stderr output
        print(captured.out)  # Optional: Print the captured output for debugging

        # List of all expected log messages
        expected_messages = [
            "Retrieving Params file.",
            "Predicting sentiments for",
            "Using model"
        ]

        # Verify all expected log messages are in the captured output
        for message in expected_messages:
            assert message in captured.out, \
                f"Expected log message not found in log output: '{message}'"


if __name__ == "__main__":
    pytest.main()
