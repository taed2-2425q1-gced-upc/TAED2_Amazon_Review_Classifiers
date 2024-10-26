"""
This module contains tests for the functions responsible for generating
embedding matrices from GloVe pre-trained embeddings. The tests cover 
the main functionality of the module, including loading GloVe embeddings,
creating an embedding matrix based on a provided tokenizer, and ensuring 
the integration of all components within the `main` function.

Tested functions:
- `load_glove_embeddings`: Validates loading GloVe embeddings from a 
  text file and constructing an appropriate embedding matrix that aligns 
  with the vocabulary of the tokenizer.

- `main`: Tests the overall functionality of the embedding matrix generation 
  script, including parameter retrieval, tokenizer loading, GloVe embedding 
  loading, and the saving of the resulting embedding matrix.

Fixtures:
- `mock_tokenizer`: Provides a mock tokenizer to simulate the behavior 
  of a Keras tokenizer with a predefined word index.
- `mock_glove_file`: Mocks the reading of GloVe embedding data from 
  a text file.
- `mock_embedding_matrix_path`: Creates a temporary path for saving 
  the generated embedding matrix.
- `mock_tokenizer_path`: Prepares a temporary path for storing a 
  mock tokenizer object.

Example GloVe Data:
The tests utilize sample GloVe data formatted as strings to simulate 
the content of a GloVe file for testing purposes.

Usage:
To run these tests, execute the test file with pytest. Ensure that 
pytest and necessary dependencies are installed in the environment.
"""


import sys
from pathlib import Path
import pickle
from unittest.mock import mock_open, patch
import pytest
import numpy as np
from loguru import logger
import tensorflow as tf

# Adjusting sys.path to include the src directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.modeling.embedding_matrix_generation import main, load_glove_embeddings

# Sample GloVe data to mock the file read for the 'glove.txt'
SAMPLE_GLOVE_DATA = """example 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
test 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1
glove 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2
"""

@pytest.fixture
def mock_tokenizer():
    """Fixture to mock the tokenizer."""
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.word_index = {'example': 1, 'test': 2, 'glove': 3}
    return tokenizer

@pytest.fixture
def mock_glove_file():
    """Fixture to mock open() for reading GloVe data."""
    with patch("builtins.open", mock_open(read_data=SAMPLE_GLOVE_DATA)) as mock_file:
        yield mock_file

@pytest.fixture
def mock_embedding_matrix_path(tmpdir):
    """Fixture for mock embedding matrix path."""
    return tmpdir.join("embedding_matrix.pkl")

@pytest.fixture
def mock_tokenizer_path(tmpdir, mock_tokenizer):
    """Fixture for mock tokenizer path."""
    path = tmpdir.join("tokenizer.pkl")
    with open(path, 'wb') as f:
        pickle.dump(mock_tokenizer, f)
    return path


def test_load_glove_embeddings(mock_tokenizer):
    """Test loading GloVe embeddings and creating the embedding matrix."""
    with patch("builtins.open", mock_open(read_data=SAMPLE_GLOVE_DATA)):
        embedding_matrix = load_glove_embeddings(
                           "dummy_path.txt",mock_tokenizer.word_index, embedding_dim=10)

    expected_shape = (4, 10)  # 3 words + 1 for padding
    assert embedding_matrix.shape == expected_shape
    np.testing.assert_array_almost_equal(
        embedding_matrix[1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], decimal=6)
    np.testing.assert_array_almost_equal(
        embedding_matrix[2], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], decimal=6)
    np.testing.assert_array_almost_equal(
        embedding_matrix[3], [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], decimal=6)

def test_main(mock_embedding_matrix_path, mock_tokenizer_path, capsys):
    """Test the main function of embedding_matrix_generation."""

    # Mock the params that main() retrieves so we use the mock paths instead
    mock_params = {
        "embeddings": "mock_glove.txt",
        "embedding_matrix": str(mock_embedding_matrix_path),
        "tokenizer": str(mock_tokenizer_path),
    }

    # Patch the utilities.get_params to return the mocked params
    with patch("src.utilities.get_params", return_value=mock_params):
        with patch("src.modeling.embedding_matrix_generation.load_glove_embeddings",
                   return_value=np.zeros((4, 100))):
            # Use the mock_tokenizer_path and patch it for reading
            with open(mock_tokenizer_path, 'rb') as f:
                mock_tokenizer = pickle.load(f)

            mock_tokenizer.word_index = {'example': 1, 'test': 2, 'glove': 3}

            # Set log level and configure loguru to output to stdout
            logger.remove()
            logger.add(sys.stdout, level="INFO")

            # Run the main function with mocked arguments
            with capsys.disabled():  # Disable capturing for debugging
                main()

            # Ensure the embedding matrix was saved
            with open(mock_embedding_matrix_path, 'rb') as f:
                embedding_matrix = pickle.load(f)
            assert embedding_matrix.shape == (4, 100)

            # Capture the output after calling main
            captured = capsys.readouterr()

            # List of all expected log messages
            expected_messages = [
                "Retrieving Params file.",
                "Loading tokenizer...",
                "Tokenizer loaded successfully.",
                "Loading GloVe pre-trained embeddings and creating embedding matrix...",
                "Embedding matrix created successfully.",
                "Saving embedding matrix...",
                f"Embedding matrix saved at: {mock_embedding_matrix_path}"
            ]

            # Verify all expected log messages are in the captured output
            for message in expected_messages:
                assert message in captured.out, \
                    f"Expected log message not found in log output: '{message}'"


if __name__ == "__main__":
    pytest.main()
