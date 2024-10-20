"""
This module processes and tokenizes Amazon review data for sentiment classification. It reads the 
raw text data, tokenizes the reviews, splits the data into training and validation sets, and saves 
the processed sequences and labels for model training. The module also checks TensorFlow version 
compatibility and handles memory management with garbage collection.

The module supports the following functionalities:
- Checking and installing the required TensorFlow version.
- Reading and processing labeled Amazon review data.
- Tokenizing reviews and saving the tokenizer for future use.
- Splitting data into training and validation sets.
- Saving tokenized sequences and labels for training.
"""


from pathlib import Path
import sys
import pickle
import subprocess
import gc
import dagshub
import numpy as np
import tensorflow as tf
import typer
from loguru import logger
from sklearn.model_selection import train_test_split
from src.config import RAW_DATA_DIR, RESOURCES_DIR

# setting path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

app = typer.Typer()

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


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    train_data_path: Path = RAW_DATA_DIR / "train.txt",
    train_sequences_path: Path = RAW_DATA_DIR/ "train_sequences.pkl",
    val_sequences_path: Path = RAW_DATA_DIR / "val_sequences.pkl",
    train_labels_path: Path = RAW_DATA_DIR / "train_labels.pkl",
    val_labels_path: Path = RAW_DATA_DIR / "val_labels.pkl"
    # -----------------------------------------
):
    """
    Main function to run the Amazon review sentiment classification training.

    Args:
        train_data_path: Path to the training data file.
        train_sequences_path: Path to save the tokenized training sequences.
        val_sequences_path: Path to save the tokenized validation sequences.
        train_labels_path: Path to save the training labels.
        val_labels_path: Path to save the validation labels.

    Returns:
        None
    """


    check_tensorflow_version()

    # ---- SETTING HYPERPARAMETERS ----
    num_words=10000

    # ---- DATA LOADING ----
    logger.info("Loading training data and extracting labels and reviews...")

    # Initialize lists to store the labels and the reviews
    labels = []
    reviews = []

    # Read the file line by line
    with open(train_data_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Split the label from the review based on the first space
            label, review = line.split(' ', 1)
            # Append to the respective lists
            labels.append(label)
            reviews.append(review.strip())

    labels = np.array(labels) # Convert to numpy array

    # ---- DATA TOKENIZATION ----
    logger.info("Tokenizing training data...")

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(reviews)
    sequences = tokenizer.texts_to_sequences(reviews)

    # ---- SAVING TOKENIZER ----
    tokenizer_path: Path = RESOURCES_DIR / "tokenizer.pkl"
    logger.info("Saving tokenizer...")
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    logger.info(f"Tokenizer saved at: {tokenizer_path}")

    del reviews
    del tokenizer
    gc.collect()

    # ---- SPLITTING DATA ----
    logger.info("Splitting data into training and validation sets...")
    x_train, x_val, y_train, y_val = train_test_split(sequences, labels,
    test_size=0.2, random_state=42, shuffle=False)

    del sequences
    gc.collect()

    del labels
    gc.collect()

    # ---- SAVING DATA ----
    logger.info("Saving training and validation data...")
    with open(train_sequences_path, 'wb') as f:
        pickle.dump(x_train, f)
    logger.info(f"Train sequences saved at: {train_sequences_path}")

    with open(val_sequences_path, 'wb') as f:
        pickle.dump(x_val, f)
    logger.info(f"Validation sequences saved at: {val_sequences_path}")

    with open(train_labels_path, 'wb') as f:
        pickle.dump(y_train, f)
    logger.info(f"Train labels saved at: {train_labels_path}")

    with open(val_labels_path, 'wb') as f:
        pickle.dump(y_val, f)
    logger.info(f"Validation labels saved at: {val_labels_path}")

if __name__ == "__main__":
    app()
