import sys
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import mock_open, patch, MagicMock
import pickle
from loguru import logger
import tensorflow as tf

# Add the parent directory (where src is located) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the modules after adjusting the sys.path
from src.modeling.predict import main, check_tensorflow_version, predict_sentiment

# Mocked TensorFlow version for testing
tf_version_mock = '2.10.0'

# Sample data to mock the file read
sample_data = """This product is great!
I did not like this product at all.
Amazing quality and fast shipping.
Poor quality, broke after one use.
"""

@pytest.fixture
def mock_data_file():
    """Fixture to mock open() for reading data."""
    with patch("builtins.open", mock_open(read_data=sample_data)) as mock_file:
        yield mock_file

def test_check_tensorflow_version():
    """Test TensorFlow version check and installation."""
    with patch("src.modeling.predict.tf") as mock_tf:
        mock_tf.__version__ = tf_version_mock
        check_tensorflow_version()  # Should not raise an exception

        # Simulating a different version
        mock_tf.__version__ = '2.5.0'  
        with pytest.raises(SystemExit):
            check_tensorflow_version()

def test_predict_sentiment():
    """Test sentiment prediction functionality."""
    # Mock the tokenizer and model
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    # Set up tokenizer mock
    mock_tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]

    # Mock model prediction for positive sentiment
    mock_model.predict.return_value = np.array([[0.8]])  # Simulating a positive sentiment
    sentiment = predict_sentiment("This is a great product!", mock_model, mock_tokenizer)
    assert sentiment == 'Positive'

    # Mock model prediction for negative sentiment
    mock_model.predict.return_value = np.array([[0.2]])  # Simulating a negative sentiment
    sentiment = predict_sentiment("This is a terrible product!", mock_model, mock_tokenizer)
    assert sentiment == 'Negative'

def test_main(mock_data_file, capsys):
    """Test the main functionality of the predict module."""
    with patch("src.modeling.predict.tf.keras.models.load_model") as mock_load_model, \
         patch("src.modeling.predict.pickle.load") as mock_pickle_load, \
         patch("src.modeling.predict.mlflow.log_param") as mock_log_param, \
         patch("src.modeling.predict.mlflow.start_run") as mock_start_run:

        mock_model = MagicMock()
        # Set the model's predict method to return a mock prediction
        mock_model.predict.return_value = [0.7]  # Mock a positive prediction (float > 0.5)

        mock_load_model.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_pickle_load.return_value = mock_tokenizer

        # Run the main function
        main(
            predict_data_path=Path("dummy_predict_path.txt"),  # Path is mocked
            model_path=Path("dummy_model_path.h5"),            # Path is mocked
            tokenizer_path=Path("dummy_tokenizer_path.pkl")    # Path is mocked
        )
        
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
            "Predicting sentiments for",
        ]

        # Verify all expected log messages are in the captured output
        for message in expected_messages:
            assert message in captured.out, \
                f"Expected log message not found in log output: '{message}'"


if __name__ == "__main__":
    pytest.main()
