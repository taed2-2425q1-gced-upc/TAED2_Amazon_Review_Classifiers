import sys
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import mock_open, patch
import pickle

# Print the current sys.path
print("sys.path before modification:", sys.path)

# Add the parent directory (where src is located) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the modules after adjusting the sys.path
from src.modeling.tokenization_and_split import main, check_tensorflow_version

# Mocked TensorFlow version for testing
tf_version_mock = '2.10.0'

# Sample data to mock the file read
sample_data = """1 This product is great!
0 I did not like this product at all.
1 Amazing quality and fast shipping.
0 Poor quality, broke after one use.
"""

@pytest.fixture
def mock_data_file():
    """Fixture to mock open() for reading data."""
    with patch("builtins.open", mock_open(read_data=sample_data)) as mock_file:
        yield mock_file

def test_check_tensorflow_version():
    """Test TensorFlow version check and installation."""
    with patch("src.modeling.tokenization_and_split.tf") as mock_tf:
        mock_tf.__version__ = tf_version_mock
        check_tensorflow_version()  # Should not raise an exception

        # Simulating a different version
        mock_tf.__version__ = '2.5.0'  
        with pytest.raises(SystemExit):
            check_tensorflow_version()

def test_main(mock_data_file):
    """Test the main functionality of the tokenization_and_split module."""
    # Define paths for the mock outputs
    train_sequences_path = Path("mock_train_sequences.pkl")
    val_sequences_path = Path("mock_val_sequences.pkl")
    train_labels_path = Path("mock_train_labels.pkl")
    val_labels_path = Path("mock_val_labels.pkl")
    
    with patch("pickle.dump") as mock_pickle_dump, patch("gc.collect") as mock_gc_collect:
        main(
            train_data_path=Path("dummy_path.txt"),  # Path is mocked
            train_sequences_path=train_sequences_path,
            val_sequences_path=val_sequences_path,
            train_labels_path=train_labels_path,
            val_labels_path=val_labels_path
        )

        # Check that the number of calls to pickle.dump is as expected
        assert mock_pickle_dump.call_count == 5  # Should save tokenizer and 4 pickled datasets

if __name__ == "__main__":
    pytest.main()
