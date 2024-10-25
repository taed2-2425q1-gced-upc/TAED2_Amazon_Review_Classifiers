"""
This module contains tests for the various endpoints of the FastAPI application defined in 
`src.app.api`. It utilizes pytest and the FastAPI test client to simulate requests to the API 
and verify the responses.

Tests include:
- The root endpoint to confirm that it returns a welcome message.
- The `/predict-review` endpoint to assess the sentiment of product reviews, including:
  - Handling of positive reviews.
  - Handling of negative reviews.
  - Handling of validation errors for incorrect request formats.
  - Handling of unexpected server errors.

Mocks are employed to simulate the behavior of external dependencies, specifically the 
`predict_sentiment` function in the `src.modeling.predict` module. This allows for isolated 
testing of the API without relying on the actual prediction logic.

Fixtures:
- `mock_predict`: A pytest fixture that mocks the `predict_sentiment` function, enabling 
  controlled testing of sentiment predictions.

To run the tests, execute the module directly or use pytest from the command line.

Usage:
    python -m pytest path_to_this_file.py
"""

import sys
from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException
from pydantic import ValidationError

# Add the parent directory (where src is located) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.app.api import app

# Create a test client for the FastAPI application
client = TestClient(app)

@pytest.fixture(scope="module")
def mock_predict():
    """Fixture to mock the predict function in the predict module."""
    with patch("src.modeling.predict.predict_sentiment") as mock:
        yield mock

def test_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Amazon review classifier app! \
        Please provide a review that should be predicted."
    }

def test_process_string_positive(mock_predict):
    """Test the /predict-review endpoint with a positive review."""
    mock_predict.return_value = ("Positive", 0.95)  # Mocking the sentiment and prediction

    response = client.post("/predict-review", json={"review": "I love this product!"})
    assert response.status_code == 200
    assert response.json() == {
        "Review is labeled": "Positive",
        "Possibility": 0.95
    }

def test_process_string_negative(mock_predict):
    """Test the /predict-review endpoint with a negative review."""
    mock_predict.return_value = ("Negative", 0.10)  # Mocking the sentiment and prediction

    response = client.post("/predict-review", json={"review": "I hate this product!"})
    assert response.status_code == 200
    assert response.json() == {
        "Review is labeled": "Negative",
        "Possibility": 0.90
    }

# def test_process_string_validation_error():
#     """Test the /predict-review endpoint with an invalid review."""
#     # Send a request with a missing 'review' field
#     response = client.post("/predict-review", json={"wrong_field": "This should fail."})
#     assert response.status_code == 422  # 422 is the expected status code for validation errors
#     
#     # Check if the error message indicates that the 'review' field is required
#     assert response.json()["detail"][0]["msg"] == "Field required"
#     assert response.json()["detail"][0]["loc"] == ["body", "review"]  # Location of the error

def test_process_string_unexpected_error(mock_predict):
    """Test the /predict-review endpoint handling an unexpected error."""
    # Simulating an unexpected error
    mock_predict.side_effect = Exception("Unexpected error")

    response = client.post("/predict-review", json={"review": "This will cause an error."})
    assert response.status_code == 500
    assert "Internal Server Error" in response.json()["detail"]

if __name__ == "__main__":
    pytest.main()
