"""
Main script for training an Amazon review classifier using TensorFlow and MLflow.
This script handles the following tasks:
- Checking the TensorFlow version and installing the correct version if needed.
- Loading the previously generated embedding matrix.
- Building a sentiment analysis model using a Bidirectional LSTM.
- Training the model on the training data.
- Evaluating the model on the validation data.
- Saving the trained model.
"""

import pickle
import subprocess
from pathlib import Path
import sys
import gc
import numpy as np
import tensorflow as tf
import dagshub
import typer
from loguru import logger
import mlflow
from codecarbon import EmissionsTracker
from src.config import MODELS_DIR, RAW_DATA_DIR, EXTERNAL_DATA_DIR

# Setting path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

app = typer.Typer()
tracker = EmissionsTracker()

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

def map_and_reshape_labels(labels):
    """
    Maps the labels to integers and reshapes the array.
    
    Args:
        labels (list): List of string labels to be mapped.

    Returns:
        np.ndarray: Mapped and reshaped label array.
    """
    label_mapping = {'__label__1': 0, '__label__2': 1}  # Internal mapping
    mapped_labels = [label_mapping[label] for label in labels]
    return np.array(mapped_labels).reshape(-1, 1)

def data_generator(reviews, labels, batch_size, maxlen):
    """
    Generate batches of padded sequences and corresponding labels for training.

    Args:
        reviews (list): List of tokenized reviews (sequences of integers) to be 
        padded and used as input data.
        labels (list or array-like): List or array of labels corresponding to the reviews.
        batch_size (int): Number of samples per batch.
        maxlen (int): Maximum length for padding the sequences.

    Yields:
        tuple: A tuple (padded_sequences, batch_labels), where padded_sequences
        are the padded input sequences, and batch_labels are the corresponding 
        labels for the batch.
    """
    total_samples = len(reviews)

    while True:
        for i in range(0, total_samples, batch_size):
            batch_reviews = reviews[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            batch_reviews, padding='post', maxlen=maxlen)

            yield padded_sequences, batch_labels

mlflow.set_experiment("amazon-reviews-test")

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    train_sequences_path: Path = RAW_DATA_DIR / "train_sequences.pkl",
    train_labels_path: Path = RAW_DATA_DIR / "train_labels.pkl",
    val_sequences_path: Path = RAW_DATA_DIR / "val_sequences.pkl",
    val_labels_path: Path = RAW_DATA_DIR / "val_labels.pkl",
    embedding_matrix_path: Path = EXTERNAL_DATA_DIR / "embedding_matrix.pkl",
    # -----------------------------------------
):
    """
    Main function to run the Amazon review sentiment classification training.

    Args:
        train_sequences_path: Path to the training sequences.
        train_labels_path: Path to the training labels.
        val_sequences_path: Path to the validation sequences.
        val_labels_path: Path to the validation labels.
        embedding_matrix_path: Path to the pre-trained embedding matrix.
    """
    tracker.start()

    check_tensorflow_version()

    with mlflow.start_run():

        # ---- SETTING HYPERPARAMETERS ----
        hyperparams = {
            "num_words": 10000,
            "maxlen": 250,
            "embedding_dim": 100,
            "lstm_units": 128,
            "dropout": 0.5,
            "batch_size": 256,
            "num_epochs": 1,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "metrics": "accuracy"
        }

        # ---- LOADING EMBEDDING MATRIX ----
        logger.info("Loading the embedding matrix...")
        with open(embedding_matrix_path, 'rb') as f:
            embedding_matrix = pickle.load(f)

        # ---- BUILDING THE MODEL ----
        logger.info("Building the model...")
        model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(hyperparams["num_words"], hyperparams["embedding_dim"],
                    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                    input_length=hyperparams["maxlen"], trainable=False),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hyperparams["lstm_units"],
            return_sequences=True)),
            tf.keras.layers.Dropout(hyperparams["dropout"]),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hyperparams["lstm_units"])),
            tf.keras.layers.Dropout(hyperparams["dropout"]),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        logger.info("Compiling the model...")
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        del embedding_matrix
        gc.collect()

        # ---- DATA LOADING----
        logger.info("Loading training data...")
        with open(train_sequences_path, 'rb') as f:
            train_sequences = pickle.load(f)
        with open(train_labels_path, 'rb') as f:
            train_labels = pickle.load(f)

        # ---- LABEL MAPPING ----
        train_labels = map_and_reshape_labels(train_labels)

        # ---- TRAINING ----
        logger.info("Training the model...")

        train_gen = data_generator(train_sequences, train_labels,
        hyperparams["batch_size"], hyperparams["maxlen"])

        model.fit(train_gen, steps_per_epoch=len(train_labels) // hyperparams["batch_size"],
                epochs=hyperparams["num_epochs"])

        del train_sequences, train_labels
        gc.collect()

        # ---- LOADING VALIDATION DATA ----
        logger.info("Loading validation data...")
        with open(val_sequences_path, 'rb') as f:
            val_sequences = pickle.load(f)
        with open(val_labels_path, 'rb') as f:
            val_labels = pickle.load(f)

        hyperparams.update({
            "num_train_samples": len(train_labels),
            "num_val_samples": len(val_labels)
        })

        # Log all the hyperparameters in one call
        mlflow.log_params(hyperparams)

        val_labels = map_and_reshape_labels(val_labels)

        # ---- VALIDATION ----
        logger.info("Padding validation data...")
        padded_val_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        val_sequences, padding='post', maxlen=hyperparams["maxlen"])

        del val_sequences
        gc.collect()

        loss, accuracy = model.evaluate(padded_val_sequences, val_labels)
        logger.info(f"Validation loss: {loss:.6f}, Validation accuracy: {accuracy:.6f}")

        metrics = {
            "validation_loss": loss,
            "validation_accuracy": accuracy
        }
        mlflow.log_metrics(metrics)

        # ---- SAVE THE MODEL ----
        logger.info("Saving the model...")
        model.save(MODELS_DIR / "sentiment_model_1_ep_256_bs.h5")

        tracker.stop()

if __name__ == "__main__":
    app()
