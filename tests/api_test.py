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
- The `/predict-reviews` endpoint to handle multiple review predictions.

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
from pydantic import BaseModel

# Add the parent directory (where src is located) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.app.api import app

# Create a test client for the FastAPI application
client = TestClient(app)

# Define a model to use for validation
class PredictRequest(BaseModel):
    review: str

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
    """Test the /predictReview endpoint with a positive review."""
    mock_predict.return_value = ("Positive", 0.95)  # Mocking the sentiment and prediction

    response = client.post("/predictReview", json={"review": "I love this product!"})
    assert response.status_code == 200
    assert response.json() == {
        "Review is labeled": "Positive",
        "Confidence": '95.0%'
    }

def test_process_string_negative(mock_predict):
    """Test the /predictReview endpoint with a negative review."""
    mock_predict.return_value = ("Negative", 0.10)  # Mocking the sentiment and prediction

    response = client.post("/predictReview", json={"review": "I hate this product!"})
    assert response.status_code == 200
    assert response.json() == {
        "Review is labeled": "Negative",
        "Confidence": '90.0%'
    }

def test_process_string_unexpected_error(mock_predict):
    """Test the /predictReview endpoint handling an unexpected error."""
    # Simulating an unexpected error
    mock_predict.side_effect = Exception("Unexpected error")

    response = client.post("/predictReview", json={"review": "This will cause an error."})
    assert response.status_code == 500
    assert "Internal Server Error" in response.json()["detail"]

def test_process_batch_reviews(mock_predict):
    """Test the /predictReviews endpoint with valid reviews."""
    mock_predict.side_effect = [
        ("Positive", 0.95),  # First review
        ("Negative", 0.10),  # Second review
    ]

    response = client.post("/predictReviews", json=[
        {"review": "I love this product!"},
        {"review": "I hate this product!"}
    ])
    assert response.status_code == 200
    assert response.json() == [
        {"Review is labeled": "Positive", "Confidence": '95.0%'},
        {"Review is labeled": "Negative", "Confidence": '90.0%'}
    ]

def test_process_batch_reviews_unexpected_error(mock_predict):
    """Test the /predictReviews endpoint handling an unexpected error."""
    # Simulating an unexpected error
    mock_predict.side_effect = Exception("Unexpected error")

    response = client.post("/predictReviews", json=[
        {"review": "This will cause an error."}
    ])
    assert response.status_code == 500
    assert "Internal Server Error" in response.json()["detail"]

if __name__ == "__main__":
    pytest.main()
