"""
This module contains tests for the functions responsible for evaluating 
the performance of a machine learning model on sentiment analysis tasks. 
The tests cover the main functionalities, including the splitting of 
reviews and labels and the evaluation of a model based on those reviews.

Tested functions:
- `split_reviews_labels`: Validates the splitting of input text reviews 
  and their corresponding labels into separate lists. The function is 
  expected to handle formatted labels correctly and extract the reviews.

- `main`: Tests the overall functionality of the evaluation script, 
  including loading model parameters, processing input data, and 
  evaluating the model. The main function integrates various components 
  such as model loading, tokenizer handling, and logging.

Fixtures:
- `mock_tokenizer`: Provides a mock tokenizer to simulate the behavior 
  of a Keras tokenizer, specifically its `texts_to_sequences` method.

- `mock_model`: Mocks a TensorFlow model, allowing the evaluation method 
  to return predefined loss and accuracy values.

- `mock_data`: Contains sample review data used to test the 
  `split_reviews_labels` function.

Example Data:
The tests utilize mock data representing formatted review strings 
with associated labels to simulate the input expected by the functions.

Usage:
To run these tests, execute the test file with pytest. Ensure that 
pytest and necessary dependencies are installed in the environment.
"""

import sys
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import pytest
from loguru import logger

# Adjusting sys.path to include the src directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.modeling.evaluate import split_reviews_labels, main

@pytest.fixture
def mock_tokenizer():
    """Fixture for a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]
    return tokenizer

@pytest.fixture
def mock_model():
    """Fixture for a mock TensorFlow model."""
    model = MagicMock()
    model.evaluate.return_value = (0.1, 0.95)  # Example values for loss and accuracy
    return model

@pytest.fixture
def mock_data():
    """Fixture for mock data."""
    return ["__label__2 This is a positive review.", "__label__1 This is a negative review."]

def test_split_reviews_labels(mock_data):
    """Test the split_reviews_labels function."""
    reviews, labels = split_reviews_labels(mock_data)

    assert len(reviews) == 2
    assert len(labels) == 2
    assert reviews[0] == "This is a positive review."
    assert reviews[1] == "This is a negative review."
    assert labels[0] == 1  # Positive
    assert labels[1] == 0  # Negative

class MockTokenizer:
    """A mock tokenizer class that simulates Keras Tokenizer behavior."""
    def texts_to_sequences(self, texts):
        """Mock the texts_to_sequences method."""
        # Just return a list of dummy sequences for each input text.
        return [[1, 2, 3] for _ in texts]

@patch('src.modeling.evaluate.mlflow')
@patch('src.modeling.evaluate.tf.keras.models.load_model')
@patch('builtins.open', new_callable=mock_open)
@patch('src.modeling.evaluate.utilities.get_params')
def test_main(mock_get_params, mock_open_file, mock_load_model, mock_mlflow, capsys):
    """Test the main function."""

    mock_mlflow.start_run.return_value = MagicMock()  # Mock the MLflow run

    # Mock parameters
    mock_get_params.return_value = {
        'evaluation_file_name': 'dummy_test.txt',
        'model': 'dummy_model.h5',
        'tokenizer': 'dummy_tokenizer.pkl',
        'max_review_length': 10
    }

    # Create a mock model object with an `evaluate` method
    mock_model = MagicMock()
    mock_model.evaluate.return_value = (0.1, 0.95)  # Simulate evaluation output (loss, accuracy)

    # Set this mock model as the return value of `load_model`
    mock_load_model.return_value = mock_model

    # Create a mock tokenizer instance
    mock_tokenizer = MockTokenizer()

    # Serialize the mock tokenizer using pickle
    pickle_data = pickle.dumps(mock_tokenizer)

    # The `open` call should return bytes for the tokenizer file and text for the evaluation file.
    mock_open_file.side_effect = [
        mock_open(
            read_data="__label__2 This is a positive review."+
            "\n__label__1 This is a negative review.").return_value,
        mock_open(read_data=pickle_data).return_value  # Tokenizer file (binary data)
    ]

    # Set log level and configure loguru to output to stdout
    logger.remove()  # Remove the default logger
    logger.add(sys.stdout, level="INFO")  # Add a new logger that outputs to stdout

    # Capture the output after calling main
    main()

    # Assert that the evaluation was called
    mock_model.evaluate.assert_called_once()

    # Capture log output
    captured = capsys.readouterr()

    # Verify log messages
    expected_messages = [
        "Retrieving Params file.",
        "Using model",
        "Predicting sentiments for",
    ]
    for message in expected_messages:
        assert message in captured.out, \
            f"Expected log message not found in log output: '{message}'"
