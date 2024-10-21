"""
Main script for preprocessing the Amazon review data.
"""

from pathlib import Path
import sys
import gc
import typer
from loguru import logger
import numpy as np
import pickle
import subprocess
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import dagshub

# setting path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.config import RAW_DATA_DIR, RESOURCES_DIR
from src import utilities

app = typer.Typer()

dagshub.init(repo_owner='Benji33', repo_name='TAED2_Amazon_Review_Classifiers', mlflow=True)

@app.command()
def main():
    """
    Main function to run the Amazon review sentiment classification training.
    """

    logger.info("Retrieving Params file.")
    params = utilities.get_params(root_dir)

    # Construct constants
    train_sequences_path: Path = RAW_DATA_DIR / params["train_sequences"]
    train_labels_path: Path = RAW_DATA_DIR / params["train_labels"]
    val_sequences_path: Path = RAW_DATA_DIR / params["val_sequences"]
    val_labels_path: Path = RAW_DATA_DIR / params["val_labels"]
    train_data_path: Path = RAW_DATA_DIR / params["train_dataset"]
    tokenizer_path: Path = RESOURCES_DIR / params["tokenizer"]

    # Step 1: Check if TensorFlow is already version 2.10.0
    if tf.__version__ == '2.10.0':
        print("Resuming execution, TensorFlow is already version 2.10.0")
        logger.info("Resuming execution, TensorFlow is already version 2.10.0")
    else:
        print(f"Current TensorFlow version: {tf.__version__}. Installing TensorFlow 2.10.0...")

        # Step 2: Uninstall current TensorFlow version
        subprocess.check_call(['pip', 'uninstall', '-y', 'tensorflow'])

        # Step 3: Install TensorFlow 2.10.0
        subprocess.check_call(['pip', 'install', 'tensorflow==2.10.0'])

        # Step 4: After installation, inform the user to restart the environment
        print("Please restart the runtime for the changes to take effect.")
        logger.info("Exiting exectution after installing TensorFlow version 2.10.0")

        # End the program here
        sys.exit("Exiting program. Please restart the runtime to apply changes.")

    # ---- SETTING HYPERPARAMETERS ----
    num_words=params["hyperparameters"]["num_words"]

    # ---- DATA LOADING ----
    logger.info("Loading training data and extracting labels and reviews...")

    # Initialize lists to store the labels and the reviews
    labels = []
    reviews = []

    # Read the file line by line
    with open( train_data_path, 'r') as file:
        for line in file:
            # Split the label from the review based on the first space
            label, review = line.split(' ', 1)
            # Append to the respective lists
            labels.append(label)
            reviews.append(review.strip())

    labels = np.array(labels) # Convert to numpy array

    # ---- DATA TOKENIZATION ----
    logger.info("Tokenizing training data...")

    tokenizer = Tokenizer(num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(reviews)
    sequences = tokenizer.texts_to_sequences(reviews)

    # ---- SAVING TOKENIZER ----
    logger.info("Saving tokenizer...")
    save_path = utilities.set_tokenizer(tokenizer_path, tokenizer)
    logger.info(f"Tokenizer saved at: {save_path}")

    del tokenizer
    gc.collect()

    # ---- SPLITTING DATA ----
    logger.info("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.2, random_state=42, shuffle=False)

    del sequences
    gc.collect()

    del labels
    gc.collect()

    # ---- SAVING DATA ----
    logger.info("Saving training and validation data...")
    with open(train_sequences_path, 'wb') as file:
        pickle.dump(X_train, file)
    logger.info(f"Train sequences saved at: {train_sequences_path}")

    with open(val_sequences_path, 'wb') as file:
        pickle.dump(X_val, file)
    logger.info(f"Validation sequences saved at: {val_sequences_path}")

    with open(train_labels_path, 'wb') as file:
        pickle.dump(y_train, file)
    logger.info(f"Train labels saved at: {train_labels_path}")

    with open(val_labels_path, 'wb') as file:
        pickle.dump(y_val, file)
    logger.info(f"Validation labels saved at: {val_labels_path}")

if __name__ == "__main__":
    app()