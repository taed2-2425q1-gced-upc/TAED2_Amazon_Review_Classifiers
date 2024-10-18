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

import pickle
import sys
import typing
import subprocess
from pathlib import Path
import numpy as np
import tensorflow as tf
import typer
from loguru import logger
import mlflow
import dagshub
from src.config import MODELS_DIR, RAW_DATA_DIR, RESOURCES_DIR

# Setting path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

# Initialize Typer app for command-line interface
app = typer.Typer()

# Initialize DagsHub integration
dagshub.init(repo_owner='Benji33', repo_name='TAED2_Amazon_Review_Classifiers', mlflow=True)

def check_tensorflow_version():
    """ Check TensorFlow version and install if not 2.10.0. """

    if tf.__version__ == '2.10.0':
        logger.info("TensorFlow version 2.10.0 already installed.")
    else:
        logger.info(f"Current TensorFlow ver: {tf.__version__}. Installing TensorFlow 2.10.0...")
        subprocess.check_call(['pip', 'uninstall', '-y', 'tensorflow'])
        subprocess.check_call(['pip', 'install', 'tensorflow==2.10.0'])
        logger.info("Exiting execution after installing TensorFlow version 2.10.0.")
        sys.exit("Please restart the runtime to apply changes.")

def predict_sentiment(text: str, model: tf.keras.Model, tokenizer):
    """
    Predict sentiment for a given text using a pre-trained TensorFlow model.

    Args:
        text (str): Input text for sentiment prediction.
        model (tf.keras.Model): Loaded TensorFlow model used for prediction.
        tokenizer: Tokenizer used to preprocess the text data.

    Returns:
        str: Predicted sentiment label ('Positive' or 'Negative').
    """
    # Preprocess the text by tokenizing and padding
    sequence = tokenizer.texts_to_sequences([text])  # Convert text to sequence
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
    sequence, padding='post', maxlen=250)
    prediction = model.predict(padded_sequence)[0]

    # Determine sentiment label based on prediction
    sentiment_label = 'Negative' if prediction < 0.5 else 'Positive'
    return sentiment_label

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
def main(
    evaluate_data_path: Path = RAW_DATA_DIR / "test.txt",
    model_path: Path = MODELS_DIR / "sentiment_model_1_ep.h5",
    tokenizer_path: Path = RESOURCES_DIR / "tokenizer.pkl",
    max_review_length: int = 250,
):
    """
    Main function to evaluate the sentiment prediction model on a test dataset.

    This function loads the pre-trained model and tokenizer, reads test data
    from a file, and evaluates the model's accuracy and loss. It logs the
    evaluation results and input artifacts to MLflow.

    Args:
        evaluate_data_path (Path): Path to the test data file.
        model_path (Path): Path to the trained model file (.h5).
        tokenizer_path (Path): Path to the saved tokenizer file (.pkl).
        max_review_length (int): Maximum length for padding reviews (default is 250).
    """
    logger.info(f"Using model {model_path} to evaluate performance on \
                data from {evaluate_data_path}")

    # Start MLflow run
    with mlflow.start_run():

        # Load the trained model and tokenizer
        logger.info("Loading model and tokenizer...")
        model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)

        # Read reviews from evaluation file
        with open(evaluate_data_path, 'r', encoding='utf-8') as file:
            evaluate_file_lines = file.readlines()  # Read all lines from the file

        # Log input data as artifact
        mlflow.log_artifact(evaluate_data_path)

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


        # Tokenize and pad the reviews
        sequences = tokenizer.texts_to_sequences(reviews)
        padded_review_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, padding='post', maxlen=max_review_length)

        # Predict sentiments for the reviews
        logger.info(f"Predicting sentiments for {len(reviews)} reviews...")

        # Evaluate the model's performance
        loss, accuracy = model.evaluate(padded_review_sequences, labels, batch_size=256)
        print("Validation loss:", loss)
        print("Validation accuracy:", accuracy)

        # Log performance metrics to MLflow
        mlflow.log_metric("Validation loss", loss)
        mlflow.log_metric("Validation accuracy", accuracy)


if __name__ == "__main__":
    app()
