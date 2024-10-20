"""
Amazon Review Classifier API using FastAPI.

This module implements a FastAPI-based web application that provides endpoints for classifying
the sentiment of Amazon reviews using a pre-trained machine learning model. It loads a
TensorFlow model and a tokenizer from a specified YAML configuration file, which are used
to process and predict the sentiment of user-submitted reviews.

The following routes are available:
    - GET / : Returns a welcome message with a brief introduction to the API.
    - POST /predict-review : Accepts a JSON payload containing a review text and returns the
      predicted sentiment ('Positive' or 'Negative') along with a confidence score.

Modules used:
    - `tensorflow`: For loading and using the machine learning model.
    - `pydantic`: For validating incoming request data.
    - `pickle`: For loading the tokenizer from disk.
    - `yaml`: For loading parameters from the configuration file.
    - `fastapi`: For building the web API.
    - `sys` and `pathlib`: For managing file paths and directory navigation.

Note:
    - The model and tokenizer are loaded once when the application starts.
    - Ensure that the `params.yaml` file, model, and tokenizer paths are correctly specified.
"""


import sys
from pathlib import Path
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
import tensorflow as tf
import yaml

# Setting path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.app.schemas import PredictRequest
from src.modeling import predict
from src.config import MODELS_DIR, RESOURCES_DIR

# Create a FastAPI instance
app = FastAPI()

# Get parameter file
params_path = root_dir/ "params.yaml"

# Load the YAML file
with open(params_path, "r") as file:
    params = yaml.safe_load(file)

# Load the trained model and tokenizer
model = tf.keras.models.load_model(Path(MODELS_DIR / params["predict"]["model"]))
with open(Path(RESOURCES_DIR / params["predict"]["tokenizer"]), 'rb') as handle:
    tokenizer = pickle.load(handle)

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
@app.post("/predict-review")
def process_string(request: PredictRequest):
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
                "Possibility": possibility}

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Validation Error: {str(e)}")
    except Exception as e:
        # Log the exception and return a 500 error
        print(f"Unexpected Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
