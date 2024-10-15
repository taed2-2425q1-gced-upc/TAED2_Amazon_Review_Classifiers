from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import typer
from loguru import logger
from tqdm import tqdm
import sys
import mlflow
from codecarbon import EmissionsTracker

# Setting path 
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.config import MODELS_DIR, RAW_DATA_DIR, RESOURCES_DIR

app = typer.Typer()
tracker = EmissionsTracker()

import dagshub
dagshub.init(repo_owner='Benji33', repo_name='TAED2_Amazon_Review_Classifiers', mlflow=True)

mlflow.set_experiment("amazon-reviews-predict")


def predict_sentiment(text, model, tokenizer):
    """Predict sentiment for a given text."""
    # Preprocess the text before predicting (tokenizing and padding)
    sequence = tokenizer.texts_to_sequences([text])  # Convert text to sequence
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=250)  # Pad sequence
    prediction = model.predict(padded_sequence)[0]
    sentiment_label = 'Negative' if prediction < 0.5 else 'Positive'
    return sentiment_label

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    predict_data_path: Path = RAW_DATA_DIR / "predict.txt",
    model_path: Path = MODELS_DIR / "sentiment_model.h5",
    # Assuming you have the tokenizer saved at the same path as before
    tokenizer_path: Path = RESOURCES_DIR / "tokenizer.pkl",
):
    tracker.start()
    logger.info(f"Using model {model_path} to predict data from {predict_data_path}")
    
    # Start MLflow run
    with mlflow.start_run():

        # Load the trained model and tokenizer
        model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)

        # Read reviews from predict.txt
        with open(predict_data_path, 'r', encoding='utf-8') as file:
            reviews = file.readlines()  # Read all lines from the file

        # Log predict file as artifact
        mlflow.log_artifact(predict_data_path)
        mlflow.log_param(f"Amount of reviews", len(reviews))
        # Predict sentiments for each review
        logger.info(f"Predicting sentiments for {len(reviews)} reviews...")
        review_counter = 1
        for review in tqdm(reviews, desc="Predicting"):
            review = review.strip()  # Remove leading/trailing whitespace
            if review:  # Check if the review is not empty
                sentiment = predict_sentiment(review, model, tokenizer)
                logger.success(f"Review: {review}\nSentiment: {sentiment}")

                # Log the input review and prediction to MLflow
                #mlflow.log_param(f"Review {review_counter} - {review}", sentiment)
                mlflow.log_param(f"review_{review_counter}", review)
                mlflow.log_param(f"sentiment_{review_counter}", sentiment)
                review_counter += 1
        tracker.stop()
        
if __name__ == "__main__":
    app()
