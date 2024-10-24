"""
This module contains unit tests for the functionality of the 
train_test_split module, which handles reading data from files, 
writing data to files, shuffling datasets, and splitting data into 
training and testing sets.

### Tests Included:

- **Mocking File Reading**: Provides sample data for testing the 
  read_data function by simulating file reading operations.
  
- **Validating read_data**: Checks that the DataFrame returned 
  by read_data matches the expected structure and content.
  
- **Testing write_data**: Ensures that the write_data function 
  accurately writes DataFrame content to a file and that the output 
  matches the expected format.
  
- **Shuffling Functionality**: Confirms that the dataset can be 
  shuffled without losing any data or altering its structure.
  
- **Train-Test Splitting**: Verifies that the train_test_split 
  function accurately divides the dataset into specified training 
  and testing sets, ensuring all original data is accounted for in 
  the output.

### Fixtures:

- **mock_data_file**: Mocks the file reading operation to simulate 
  reading sample data for the read_data function.

### Test Framework:

- **Pytest**: Utilized for structuring and executing the tests.

- **Pandas**: Used for data manipulation and validation within the 
  tests.

### Sample Data:

- The sample data used for testing consists of labeled product 
  reviews, designed to simulate real input data for the reading 
  and splitting processes.
"""


import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from loguru import logger  # Import loguru for logging

# Adjusting sys.path to include the src directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.modeling.train import map_and_reshape_labels, data_generator, main


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
    with capsys.disabled():
        main()  # Execute the main function

    # Capture the output after calling main
    captured = capsys.readouterr()  # Capture stdout and stderr output

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
