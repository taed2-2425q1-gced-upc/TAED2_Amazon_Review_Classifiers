"""" Unit tests for the utilities module. """

import sys
from pathlib import Path
import pickle
from unittest.mock import mock_open, patch
import yaml
import pytest

# Add the parent directory (where src is located) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utilities import get_params, get_tokenizer, get_evaluation_file_lines, \
    set_tokenizer, check_tensorflow_version

# Mocked TensorFlow version for testing
TF_VERSION_MOCK = '2.10.0'

def test_get_params():
    """Test loading parameters from a YAML file."""
    mock_params = {
        'model': 'dummy_model.h5',
        'tokenizer': 'dummy_tokenizer.pkl',
        'predict_dataset': 'dummy_predict_path.txt'
    }

    # Mock the open function to simulate reading from a YAML file
    with patch("builtins.open", mock_open(read_data=yaml.dump(mock_params))):
        result = get_params(Path("/mock/root"))
        assert result == mock_params

def test_get_tokenizer():
    """Test loading a tokenizer from a pickle file."""
    # Use a simple dictionary to mock a tokenizer for pickling
    mock_tokenizer = {'tokenizer_key': 'tokenizer_value'}

    # Mock the open function to simulate reading a tokenizer
    with patch("builtins.open", mock_open(read_data=pickle.dumps(mock_tokenizer))) as mock_file:
        result = get_tokenizer(Path("/mock/tokenizer.pkl"))
        assert result == mock_tokenizer
        mock_file.assert_called_once_with(Path("/mock/tokenizer.pkl"), 'rb')

def test_get_evaluation_file_lines():
    """Test reading lines from an evaluation file."""
    mock_lines = ["This product is great!\n", "I did not like this product at all.\n"]

    # Mock the open function to simulate reading from a review file
    with patch("builtins.open", mock_open(read_data=''.join(mock_lines))):
        result = get_evaluation_file_lines(Path("/mock/evaluation.txt"))
        assert result == mock_lines

def test_set_tokenizer():
    """Test saving a tokenizer to a pickle file."""
    # Use a simple dictionary to mock a tokenizer for pickling
    mock_tokenizer = {'tokenizer_key': 'tokenizer_value'}
    tokenizer_file_path = Path("/mock/tokenizer.pkl")

    # Use a mock to replace the file write operation
    with patch("builtins.open", mock_open()) as mock_file:
        result = set_tokenizer(tokenizer_file_path, mock_tokenizer)
        assert result == tokenizer_file_path
        mock_file.assert_called_once_with(tokenizer_file_path, 'wb')
        mock_file().write.assert_called_once_with(pickle.dumps(mock_tokenizer))

def test_check_tensorflow_version():
    """Test TensorFlow version check."""
    with patch("src.utilities.tf") as mock_tf:
        mock_tf.__version__ = TF_VERSION_MOCK
        check_tensorflow_version()  # Should not raise an exception

        # Simulate an incorrect TensorFlow version
        mock_tf.__version__ = '2.5.0'
        with pytest.raises(SystemExit):
            check_tensorflow_version()  # Should raise SystemExit

if __name__ == "__main__":
    pytest.main()
