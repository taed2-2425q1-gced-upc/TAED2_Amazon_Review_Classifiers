import pytest
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import pickle
from unittest.mock import MagicMock, patch
from loguru import logger

# Adjusting sys.path to include the src directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.modeling.evaluate import check_tensorflow_version, predict_sentiment, split_reviews_labels, main

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
    model.predict.return_value = np.array([[0.8]])
    return model

@pytest.fixture
def mock_data():
    """Fixture for mock data."""
    return ["__label__2 This is a positive review.", "__label__1 This is a negative review."]

# Mocked TensorFlow version for testing
tf_version_mock = '2.10.0'

def test_check_tensorflow_version():
    """Test TensorFlow version check and installation."""
    with patch("src.modeling.evaluate.tf") as mock_tf:
        mock_tf.__version__ = tf_version_mock
        check_tensorflow_version()  # Should not raise an exception

        # Simulating a different version
        mock_tf.__version__ = '2.5.0'  
        with pytest.raises(SystemExit):
            check_tensorflow_version()

def test_predict_sentiment(mock_model, mock_tokenizer):
    """Test the predict_sentiment function."""
    text = "This is a test review."
    
    with patch('tensorflow.keras.models.load_model', return_value=mock_model):
        result = predict_sentiment(text, mock_model, mock_tokenizer)
        
    assert result == 'Positive'
    mock_tokenizer.texts_to_sequences.assert_called_once_with([text])
    mock_model.predict.assert_called_once()

def test_split_reviews_labels(mock_data):
    """Test the split_reviews_labels function."""
    reviews, labels = split_reviews_labels(mock_data)
    
    assert len(reviews) == 2
    assert len(labels) == 2
    assert reviews[0] == "This is a positive review."
    assert reviews[1] == "This is a negative review."
    assert labels[0] == 1  # Positive
    assert labels[1] == 0  # Negative

@patch('src.modeling.evaluate.mlflow')
@patch('src.modeling.evaluate.tf.keras.models.load_model')
@patch('builtins.open', new_callable=MagicMock)
@patch('src.modeling.evaluate.pickle.load')  # Fixed patch
def test_main(mock_pickle_load, mock_open, mock_load_model, mock_mlflow, capsys):
    """Test the main function."""

    # Mocking tokenizer
    mock_tokenizer = MagicMock()
    mock_pickle_load.return_value = mock_tokenizer

    # Mocking model
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    # Mocking the evaluate method to return loss and accuracy
    mock_model.evaluate.return_value = (0.1, 0.95)  # Example values for loss and accuracy

    # Mocking file read to return mock data
    mock_open.return_value.__enter__.return_value.readlines.return_value = [
        '__label__2 This is a positive review.', 
        '__label__1 This is a negative review.'
    ]

    evaluate_data_path = Path("dummy_test.txt")
    model_path = Path("dummy_model.h5")
    tokenizer_path = Path("dummy_tokenizer.pkl")

    # Run the main function
    main(evaluate_data_path=evaluate_data_path, model_path=model_path, tokenizer_path=tokenizer_path)

    # Assertions
    # Check the calls for open
    mock_open.assert_any_call(tokenizer_path, 'rb')
    mock_open.assert_any_call(evaluate_data_path, 'r', encoding='utf-8')

    # Ensure that open was called the correct number of times
    assert mock_open.call_count == 2
    
    # Set log level and configure loguru to output to stdout
    logger.remove()  # Remove the default logger
    logger.add(sys.stdout, level="INFO")  # Add a new logger that outputs to stdout

    # Run the main function and capture stdout/stderr
    with capsys.disabled():  # This disables the capturing so you can see print statements while debugging
        main()  # Execute the main function

    # Capture the output after calling main
    captured = capsys.readouterr()  # Capture stdout and stderr output
    print(captured.out)  # Optional: Print the captured output for debugging

    # List of all expected log messages
    expected_messages = [
        "Using model",
        "Loading model and tokenizer...",
        "Predicting sentiments for"
    ]

    # Verify all expected log messages are in the captured output
    for message in expected_messages:
        assert message in captured.out, \
            f"Expected log message not found in log output: '{message}'"
