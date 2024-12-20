"""
This module predicts sentiment for Amazon reviews using a pre-trained TensorFlow model. It loads the 
trained model and tokenizer, reads reviews from a text file, processes the reviews, and predicts the 
sentiment (either 'Positive' or 'Negative'). The module also logs relevant data, including the 
predictions, into MLflow, and tracks the environmental impact using CodeCarbon.

The module supports the following functionalities:
- TensorFlow version check and installation if necessary.
- Loading a pre-trained sentiment model and tokenizer.
- Reading reviews from a text file and preprocessing them for sentiment prediction.
- Predicting the sentiment for each review using the model.
- Logging prediction results and input reviews into MLflow.
- Tracking emissions during the prediction process.
"""


from pathlib import Path
import sys
import typing
import tensorflow as tf
import typer
from loguru import logger
from tqdm import tqdm
import mlflow
import dagshub
from codecarbon import EmissionsTracker

# Setting path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.config import MODELS_DIR, RAW_DATA_DIR, RESOURCES_DIR
from src import utilities

# Initialize Typer app and EmissionsTracker
app = typer.Typer()
tracker = EmissionsTracker()

# Initialize DagsHub integration with MLflow
dagshub.init(repo_owner='Benji33', repo_name='TAED2_Amazon_Review_Classifiers', mlflow=True)

# Set the experiment for MLflow
mlflow.set_experiment("amazon-reviews-predict")

def predict_sentiment(text: str, model: tf.keras.Model, tokenizer) -> typing.Tuple[str, float]:
    """
    Predict sentiment for a given text using a pre-trained model and tokenizer.

    Args:
        text (str): The input review text to predict the sentiment for.
        model (tf.keras.Model): The pre-trained TensorFlow model used for sentiment prediction.
        tokenizer: The tokenizer to convert text into sequences for the model.

    Returns:
        Tuple[str, float]: Tuple containing the predicted sentiment label ('Positive' or 'Negative')
        and the model's prediction probability (float).
    """
    # Preprocess the text before predicting (tokenizing and padding)
    sequence = tokenizer.texts_to_sequences([text])  # Convert text to sequence
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
    sequence, padding='post', maxlen=250)
    prediction = model.predict(padded_sequence)[0]
    sentiment_label = 'Negative' if prediction < 0.5 else 'Positive'
    return sentiment_label, prediction


mlflow.set_experiment("amazon-reviews-predict")

@app.command()
def main():
    """
    Main function to predict sentiments for a dataset of Amazon reviews.

    This function loads a pre-trained model and tokenizer, reads review data from a text file,
    and predicts the sentiment for each review. The predictions and input data are logged to MLflow,
    and carbon emissions during the inference process are tracked using EmissionsTracker.
    """

    # Start tracking carbon emissions
    tracker.start()

    # Check if TensorFlow is already version 2.10.0
    utilities.check_tensorflow_version()

    logger.info("Retrieving Params file.")
    params = utilities.get_params(root_dir)

    # Construct paths to the model, tokenizer, and prediction dataset
    model_path: Path = MODELS_DIR / params['model']
    tokenizer_path: Path = RESOURCES_DIR / params["tokenizer"]
    predict_data_path: Path = RAW_DATA_DIR / params["predict_dataset"]

    logger.info(f"Using model {model_path} to predict data from {predict_data_path}")


    # Start MLflow run for tracking the prediction process
    with mlflow.start_run():

        # Load the trained model and tokenizer
        model = tf.keras.models.load_model(model_path)
        tokenizer = utilities.get_tokenizer(tokenizer_path)

        # Read reviews from predict.txt
        with open(predict_data_path, 'r', encoding='utf-8') as file:
            reviews = file.readlines()  # Read all lines from the file

        # Log prediction dataset as artifact in MLflow
        mlflow.log_artifact(predict_data_path)
        mlflow.log_param("Amount of reviews", len(reviews))
        # Predict sentiments for each review
        logger.info(f"Predicting sentiments for {len(reviews)} reviews...")
        review_counter = 1
        for review in tqdm(reviews, desc="Predicting"):
            review = review.strip()  # Remove leading/trailing whitespace
            if review:  # Check if the review is not empty
                sentiment, _ = predict_sentiment(review, model, tokenizer)
                logger.success(f"Review: {review}\nSentiment: {sentiment}")

                # Log the input review and its prediction to MLflow
                mlflow.log_param(f"review_{review_counter}", review)
                mlflow.log_param(f"sentiment_{review_counter}", sentiment)
                review_counter += 1

        # Stop tracking emissions
        tracker.stop()

if __name__ == "__main__":
    app()
