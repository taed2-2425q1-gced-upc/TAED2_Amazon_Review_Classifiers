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

def predict_sentiments(reviews: typing.List[str], model: tf.keras.Model, tokenizer) \
    -> typing.List[typing.Tuple[str, float]]:
    """
    Predict sentiments for a batch of texts using a pre-trained model and tokenizer.

    Args:
        reviews (List[str]): The input review texts to predict the sentiments for.
        model (tf.keras.Model): The pre-trained TensorFlow model used for sentiment prediction.
        tokenizer: The tokenizer to convert texts into sequences for the model.

    Returns:
        List[Tuple[str, float]]: List containing tuples of predicted
        sentiment labels ('Positive' or 'Negative') and the model's 
        prediction probabilities (floats).
    """
    # Preprocess the texts before predicting (tokenizing and padding)
    sequences = tokenizer.texts_to_sequences(reviews)  # Convert texts to sequences
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,
    padding='post', maxlen=250)

    # Suppress TensorFlow's progress output
    predictions = model.predict(padded_sequences, verbose=0)

    # Convert predictions to sentiment labels
    sentiments = [('Negative' if pred < 0.5 else 'Positive', pred) for pred in predictions]

    return sentiments

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
    predict_output_path: Path = RAW_DATA_DIR / params["predict_output"]

    logger.info(f"Using model {model_path} to predict data from {predict_data_path}")

    # Start MLflow run for tracking the prediction process
    with mlflow.start_run():
        # Load the trained model and tokenizer
        model = tf.keras.models.load_model(model_path)
        tokenizer = utilities.get_tokenizer(tokenizer_path)

        # Read reviews from the prediction dataset
        with open(predict_data_path, 'r', encoding='utf-8') as file:
            reviews = file.readlines()  # Read all lines from the file

        # Log prediction dataset as artifact in MLflow
        mlflow.log_artifact(predict_data_path)
        mlflow.log_param("Amount of reviews", len(reviews))

        # Define batch size
        batch_size = 1024  # Adjust the batch size as needed

        # Use tqdm to show progress for the prediction process
        logger.info(f"Predicting sentiments for {len(reviews)} reviews...")

        # Create a single tqdm instance for the total number of reviews
        with tqdm(total=len(reviews), desc="Predicting Sentiments") as pbar:
            for i in range(0, len(reviews), batch_size):
                batch_reviews = [review.strip() for review in \
                    reviews[i:i + batch_size] if review.strip()]  # Create a batch
                if batch_reviews:  # Check if the batch is not empty
                    sentiments = predict_sentiments(batch_reviews, model, tokenizer)
                    # Print the reviews and their predicted sentiments to a file
                    with open(predict_output_path, 'a', encoding='utf-8') as f:
                        for review, (sentiment, _) in zip(batch_reviews, sentiments):
                            f.write(f"Review: {review}\nPredicted Sentiment: {sentiment}\n\n")
                pbar.update(len(batch_reviews))  # Update the progress bar
        # Log the "predictions.txt" file as an artifact in MLflow
        mlflow.log_artifact(predict_output_path)

        # Stop tracking emissions
        tracker.stop()

if __name__ == "__main__":
    app()
