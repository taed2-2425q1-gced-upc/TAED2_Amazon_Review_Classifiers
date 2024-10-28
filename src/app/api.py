"""
Amazon Review Classifier API using FastAPI.

This module implements a FastAPI-based web application that provides endpoints for classifying
the sentiment of Amazon reviews using a pre-trained machine learning model. It loads a
TensorFlow model and a tokenizer from a specified YAML configuration file, which are used
to process and predict the sentiment of user-submitted reviews.

The following routes are available:
    - GET / : Returns a welcome message with a brief introduction to the API.
    - POST /predictReview : Accepts a JSON payload containing a review text and returns the
      predicted sentiment ('Positive' or 'Negative') along with a confidence score.
    - POST /predictReviews : Accepts a JSON payload containing a list of review texts and returns the
      predicted sentiments ('Positive' or 'Negative') along with the confidence scores.

Modules used:
    - `tensorflow`: For loading and using the machine learning model.
    - `pydantic`: For validating incoming request data.
    - `fastapi`: For building the web API.
    - `sys` and `pathlib`: For managing file paths and directory navigation.

Note:
    - The model and tokenizer are loaded once when the application starts.
    - Ensure that the `params.yaml` file, model, and tokenizer paths are correctly specified.
"""


import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
import tensorflow as tf

# Setting path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.app.schemas import PredictRequest
from src.modeling import predict
from src.config import MODELS_DIR, RESOURCES_DIR
from src import utilities

# Create a FastAPI instance
app = FastAPI()

# Get parameter file
params = utilities.get_params(root_dir)

# Load the trained model and tokenizer
model = tf.keras.models.load_model(Path(MODELS_DIR / params["model"]))
tokenizer =  utilities.get_tokenizer(Path(RESOURCES_DIR / params["tokenizer"]))

# Root route to return basic information
@app.get("/")
def root():
    """
    Root endpoint that returns a welcome message.

    Returns:
        dict: A welcome message with information on the app's purpose.
    """
    return {
        "message": "Welcome to the Amazon review classifier app! \
        Please provide a review that should be predicted."}

# Endpoint to accept a review that should be predicated in the request body,
# it first gets validated, then will be used for infehrence
@app.post("/predictReview")
def process_review(request: PredictRequest):
    """
    POST endpoint to process a review for sentiment classification.

    Args:
        request (PredictRequest): A Pydantic model containing the review text to be classified.

    Returns:
        dict: A dictionary containing the sentiment label ('Positive' or 'Negative')
              and the possibility (confidence score of the prediction).

    Raises:
        HTTPException: If validation fails or an unexpected error occurs during processing.
    """
    try:

        # Predict sentiment for given review
        sentiment, prediction = predict.predict_sentiment(request.review, model, tokenizer)

        possibility = 0.0
        if sentiment != "Negative":
            possibility = float(prediction)
        else:
            possibility = 1.0 - float(prediction)

        # Return the processed result
        return {"Review is labeled": sentiment,
                "Confidence": f"{str(round(possibility * 100, 2))}%"}

    except ValidationError as exception:
        raise HTTPException(status_code=400, detail=f"Validation Error: {str(exception)}")
    except Exception as exception:
        # Log the exception and return a 500 error
        print(f"Unexpected Error: {str(exception)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(exception)}")


@app.post("/predictReviews")
def process_reviews(requests: list[PredictRequest]):
    """
    POST endpoint to process a batch of reviews for sentiment classification.

    Args:
        requests  [PredictRequest]: A Pydantic model containing a list of review texts to be classified.

    Returns:
        list[dict]: A list of dictionaries containing the sentiment label ('Positive' or 'Negative')
              and the possibility (confidence score of the prediction) for each review.

    Raises:
        HTTPException: If validation fails or an unexpected error occurs during processing.
    """
    labeled_reviews = []
    try:
        for request in requests:
            labeled_reviews.append(process_review(request))

        return labeled_reviews

    except ValidationError as exception:
        raise HTTPException(status_code=400, detail=f"Validation Error: {str(exception)}")
    except Exception as exception:
        # Log the exception and return a 500 error
        print(f"Unexpected Error: {str(exception)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(exception)}")
