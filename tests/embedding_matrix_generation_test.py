import sys
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import mock_open, patch
import pickle
import tensorflow as tf

# Adjusting sys.path to include the src directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.modeling.embedding_matrix_generation import main, check_tensorflow_version, load_glove_embeddings

# Mocked TensorFlow version for testing
tf_version_mock = '2.10.0'

@pytest.fixture
def mock_tokenizer():
    """Fixture to mock the tokenizer."""
    tokenizer = tf.keras.preprocessing.text.Tokenizer() # Create an instance of Keras Tokenizer
    tokenizer.word_index = {'example': 1, 'test': 2, 'glove': 3}  # Set the word index
    return tokenizer

@pytest.fixture
def mock_glove_data():
    """Fixture to mock GloVe data."""
    return "example 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0\n" \
           "test 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1\n" \
           "glove 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2\n"

@pytest.fixture
def mock_embedding_matrix_path(tmpdir):
    """Fixture for mock embedding matrix path."""
    return tmpdir.join("embedding_matrix.pkl")

@pytest.fixture
def mock_tokenizer_path(tmpdir, mock_tokenizer):
    """Fixture for mock tokenizer path."""
    path = tmpdir.join("tokenizer.pkl")
    with open(path, 'wb') as f:  # Ensure 'wb' mode is used
        pickle.dump(mock_tokenizer, f)  # Use the Keras Tokenizer
    return path


def test_check_tensorflow_version():
    """Test TensorFlow version check."""
    # Mock the tf module
    with patch("src.modeling.embedding_matrix_generation.tf") as mock_tf:
        mock_tf.__version__ = tf_version_mock
        
        check_tensorflow_version()  # Should not raise an exception

        # Simulating a different version
        mock_tf.__version__ = '2.5.0'
        with pytest.raises(SystemExit):
            check_tensorflow_version()

def test_load_glove_embeddings(mock_glove_data, mock_tokenizer):
    """Test loading GloVe embeddings and creating the embedding matrix."""
    # Mock the open function to read GloVe data
    with patch("builtins.open", mock_open(read_data=mock_glove_data)):
        embedding_matrix = load_glove_embeddings("dummy_path.txt", mock_tokenizer.word_index, embedding_dim=10)

    # Expected shape of the embedding matrix
    expected_shape = (4, 10)  # 3 words + 1 for padding
    assert embedding_matrix.shape == expected_shape
    np.testing.assert_array_almost_equal(embedding_matrix[1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], decimal=6)
    np.testing.assert_array_almost_equal(embedding_matrix[2], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], decimal=6)
    np.testing.assert_array_almost_equal(embedding_matrix[3], [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], decimal=6)

def test_main(mock_embedding_matrix_path, mock_tokenizer_path):
    """Test the main function of embedding_matrix_generation."""
    with patch("src.modeling.embedding_matrix_generation.check_tensorflow_version"):
        with patch("src.modeling.embedding_matrix_generation.load_glove_embeddings", return_value=np.zeros((4, 100))):
            # Use the actual mock_tokenizer defined in the fixture
            with open(mock_tokenizer_path, 'rb') as f:
                mock_tokenizer = pickle.load(f)

            # Run the main function with mocked arguments
            main(embeddings_path=Path("dummy_glove_path.txt"),
                 embedding_matrix_path=mock_embedding_matrix_path,
                 tokenizer_path=mock_tokenizer_path)

            # Ensure that the embedding matrix was saved
            with open(mock_embedding_matrix_path, 'rb') as f:
                embedding_matrix = pickle.load(f)
            assert embedding_matrix.shape == (4, 100)  # Check shape of the matrix


if __name__ == "__main__":
    pytest.main()
