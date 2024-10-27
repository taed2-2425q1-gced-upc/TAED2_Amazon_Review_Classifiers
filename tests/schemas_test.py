"""
This module contains tests for the Pydantic models used for validating 
incoming requests related to sentiment prediction. The primary focus 
is on the `PredictRequest` model, which ensures that the input data 
conforms to the expected structure and constraints.

Tested Models:
- `PredictRequest`: Validates the structure and constraints of the 
  review input. It checks that the review is a non-empty string 
  within specified length limits.

Key Tests:
- `test_valid_review`: Confirms that a valid review string is accepted 
  and stored correctly in the model.
  
- `test_review_length_validation_pass`: Ensures that a review string 
  that is exactly at the maximum length is valid and does not raise 
  errors.
  
- `test_review_length_validation_fail`: Verifies that a review string 
  exceeding the maximum allowed length raises a `ValidationError` with 
  the correct error message and location.

Additional Tests (commented out):
- `test_empty_review_validation_fail`: (Currently commented out) 
  Confirms that an empty review string raises a `ValidationError` 
  indicating that the field is required.

Fixtures:
- `mock_get_params`: Mocks the `get_params` function from the 
  utilities module, allowing tests to run without relying on 
  external parameter sources. This fixture is defined with module 
  scope for efficiency.

Usage:
To run these tests, execute the test file using pytest. Ensure that 
pytest and the necessary dependencies, including Pydantic, are 
installed in your environment.
"""


import sys
from pathlib import Path
from unittest.mock import patch
import pytest
from pydantic import ValidationError

# Add the parent directory (where src is located) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.app.schemas import PredictRequest

@pytest.fixture(scope="module")
def mock_get_params():
    """Fixture to mock the get_params function from utilities."""
    with patch("src.utilities.get_params") as mock:
        mock.return_value = {"max_review_length": 250}  # Mocking the maximum review length
        yield mock

def test_valid_review():
    """Test that a valid review passes validation."""
    review_text = "This is a valid review."
    request = PredictRequest(review=review_text)
    assert request.review == review_text  # The review should be stored as is

def test_review_length_validation_pass():
    """Test that a review of valid length passes validation."""
    review_text = "A " * 250  # Maximum length
    request = PredictRequest(review=review_text)
    assert request.review == review_text  # Should not raise an error

def test_review_length_validation_fail():
    """Test that a review exceeding the maximum length raises a validation error."""
    review_text = "A " * 251  # Exceeds maximum length
    with pytest.raises(ValidationError) as exc_info:
        PredictRequest(review=review_text)

    # Check for the specific error type and structure
    assert exc_info.value.errors()[0]['loc'] == ('review',)
    assert "Value error," in exc_info.value.errors()[0]['msg']  # Check for Value error
    text = "Value error, The input review exceeds with 251 the maximum length of 250 words."
    assert text in exc_info.value.errors()[0]['msg']

if __name__ == "__main__":
    pytest.main()
