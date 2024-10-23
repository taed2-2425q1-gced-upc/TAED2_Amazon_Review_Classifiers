"""
Module for predicting sentiment from Amazon reviews using a pre-trained
TensorFlow model and tokenizer. This script reads test data, evaluates the
model's performance on the dataset, and logs relevant metrics using MLflow.

The module also integrates with DagsHub for experiment tracking and supports
the following functionalities:
- Preprocessing text data (tokenizing and padding)
- Evaluating model accuracy and loss on labeled test data
- Logging model performance and input data artifacts to MLflow
"""

import sys
import typing
from pathlib import Path
import numpy as np
import tensorflow as tf
import typer
from loguru import logger
import mlflow
import dagshub

# Setting path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.config import MODELS_DIR, RAW_DATA_DIR, RESOURCES_DIR
from src import utilities

# Initialize Typer app for command-line interface
app = typer.Typer()

# Initialize DagsHub integration
dagshub.init(repo_owner='Benji33', repo_name='TAED2_Amazon_Review_Classifiers', mlflow=True)

def split_reviews_labels(input_lines: list[str]) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Split raw input lines into reviews and their corresponding labels.

    Args:
        input_lines (list[str]): List of lines containing review data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays,
        one for reviews and one for labels (1 for positive, 0 for negative).
    """
    labels = []
    reviews = []

    for line in input_lines:
        split_line = line.strip().split(' ', 1)
        label = 1 if split_line[0] == '__label__2' else 0
        review = split_line[1]
        labels.append(label)
        reviews.append(review)

    # Convert reviews and labels to numpy arrays
    reviews = np.array(reviews)
    labels = np.array(labels)

    return reviews, labels

# Set the experiment for MLflow
mlflow.set_experiment("amazon-reviews-predict")


@app.command()
def main():
    """
    Main function to evaluate the sentiment prediction model on a test dataset.

    This function loads the pre-trained model and tokenizer, reads test data
    from a file, and evaluates the model's accuracy and loss. It logs the
    evaluation results and input artifacts to MLflow.
    """

    # Check TensorFlow version and install if not 2.10.0
    utilities.check_tensorflow_version()

    logger.info("Retrieving Params file.")
    params = utilities.get_params(root_dir)

    # Construct constants
    evaluate_data_path: Path = RAW_DATA_DIR / params['evaluation_file_name']
    model_path: Path = MODELS_DIR / params['model']
    tokenizer_path: Path = RESOURCES_DIR / params["tokenizer"]
    max_review_length: int = params["max_review_length"]

    logger.info(f"Using model {str(model_path)} to evaluate performance on \
                data from {str(evaluate_data_path)}")

    # Start MLflow run
    with mlflow.start_run():

        # Log input data as artifact
        mlflow.log_artifact(str(evaluate_data_path))

        # Get lines from evaluation file
        evaluate_file_lines = utilities.get_evaluation_file_lines(evaluate_data_path)

        # Split reviews and labels from the input data
        reviews, labels = split_reviews_labels(evaluate_file_lines)

        mlflow.log_param("Amount of reviews", len(reviews))
        mlflow.log_param("Amount of labels", len(labels))

        # Data sanity check
        if len(reviews) != len(labels):
            raise ValueError(f"Mismatch between reviews and labels: \
                             {len(reviews)} reviews but {len(labels)} \
                              labels. Ensure that each review has a \
                              corresponding label in the input file.")

        # Get the tokenizer
        tokenizer = utilities.get_tokenizer(tokenizer_path)

        # Get the model
        model = tf.keras.models.load_model(model_path)

        # Tokenize and pad the reviews
        sequences = tokenizer.texts_to_sequences(reviews)
        padded_review_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, padding='post', maxlen=max_review_length)

        # Predict sentiments for the reviews
        logger.info(f"Predicting sentiments for {len(reviews)} reviews...")

        # Evaluate the model's performance
        loss, accuracy = model.evaluate(padded_review_sequences, labels, batch_size=256)
        print("Test loss:", loss)
        print("Test accuracy:", accuracy)

        # Log performance metrics to MLflow
        mlflow.log_metric("Test loss", loss)
        mlflow.log_metric("Test accuracy", accuracy)


if __name__ == "__main__":
    app()
