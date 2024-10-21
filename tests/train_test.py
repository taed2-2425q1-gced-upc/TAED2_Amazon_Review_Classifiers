import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from loguru import logger  # Import loguru for logging

# Adjusting sys.path to include the src directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.modeling.train import map_and_reshape_labels, data_generator, check_tensorflow_version, main

# Mocked TensorFlow version for testing
tf_version_mock = '2.10.0'

def test_check_tensorflow_version():
    """Test TensorFlow version check and installation."""
    with patch("src.modeling.train.tf") as mock_tf:
        mock_tf.__version__ = tf_version_mock
        check_tensorflow_version()  # Should not raise an exception

        # Simulating a different version
        mock_tf.__version__ = '2.5.0'  
        with pytest.raises(SystemExit):
            check_tensorflow_version()


def test_map_and_reshape_labels():
    """Test label mapping and reshaping."""
    labels = ['__label__1', '__label__2', '__label__1']
    expected_output = np.array([[0], [1], [0]])
    output = map_and_reshape_labels(labels)
    np.testing.assert_array_equal(output, expected_output)


def test_data_generator():
    """Test data generator for batching and padding."""
    reviews = [[1, 2, 3], [4, 5], [6]]
    labels = [0, 1, 0]
    batch_size = 2
    maxlen = 5

    gen = data_generator(reviews, labels, batch_size, maxlen)
    padded_sequences, batch_labels = next(gen)

    expected_padded = np.array([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    expected_labels = np.array([0, 1])

    np.testing.assert_array_equal(padded_sequences, expected_padded)
    np.testing.assert_array_equal(batch_labels, expected_labels)

# Patch the necessary modules
@patch('src.modeling.train.pickle')
@patch('src.modeling.train.mlflow')
@patch('src.modeling.train.EmissionsTracker')
@patch('src.modeling.train.tf.keras.models.Sequential')
def test_main(mock_sequential, mock_emissions_tracker, mock_mlflow, mock_pickle, capsys):
    """Test the main training function."""
    
    # Set up mock objects
    mock_mlflow.start_run.return_value = MagicMock()
    mock_mlflow.log_params.return_value = None
    mock_mlflow.log_metrics.return_value = None

    # Mock embedding matrix and sequences/labels
    mock_embedding_matrix = np.random.rand(10000, 100)
    mock_pickle.load.side_effect = [
        mock_embedding_matrix,  # Embedding matrix
        [np.random.randint(1, 10000, size=(250,)) for _ in range(10)],  # Train sequences
        ['__label__1'] * 10,  # Train labels
        [np.random.randint(1, 10000, size=(250,)) for _ in range(5)],  # Val sequences
        ['__label__2'] * 5  # Val labels
    ]

    # Mock the Keras model to prevent actual training
    mock_model = mock_sequential.return_value
    mock_model.fit.return_value = None
    mock_model.evaluate.return_value = (0.1, 0.9)
    mock_model.compile.return_value = None
    mock_model.save.return_value = None

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
        "Loading the embedding matrix...",
        "Building the model...",
        "Compiling the model...",
        "Loading training data...",
        "Training the model...",
        "Loading validation data...",
        "Padding validation data...",
        "Validation loss: 0.100000, Validation accuracy: 0.900000",
        "Saving the model..."
    ]

    # Verify all expected log messages are in the captured output
    for message in expected_messages:
        assert message in captured.out, \
            f"Expected log message not found in log output: '{message}'"