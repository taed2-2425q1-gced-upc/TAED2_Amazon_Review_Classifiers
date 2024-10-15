"""
Main script for preprocessing the Amazon review data.
"""

from pathlib import Path
import sys
import gc
import tensorflow as tf
import typer
from loguru import logger
from tqdm import tqdm
import mlflow
import numpy as np
import pickle
import subprocess
import gc
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# setting path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.config import MODELS_DIR, RAW_DATA_DIR, EXTERNAL_DATA_DIR, RESOURCES_DIR

app = typer.Typer()

import dagshub
dagshub.init(repo_owner='Benji33', repo_name='TAED2_Amazon_Review_Classifiers', mlflow=True)

@app.command()
def main(

    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    train_data_path: Path = RAW_DATA_DIR / "train.txt",
    tokenizer_path: Path = RESOURCES_DIR / "tokenizer.pkl",
    train_sequences_path: Path = RAW_DATA_DIR/ "train_sequences.pkl",
    val_sequences_path: Path = RAW_DATA_DIR / "val_sequences.pkl",
    train_labels_path: Path = RAW_DATA_DIR / "train_labels.pkl",
    val_labels_path: Path = RAW_DATA_DIR / "val_labels.pkl"
    # -----------------------------------------
):
    """
    Main function to run the Amazon review sentiment classification training.

    Args:
        train_data_path (Path): Path to the training data (default: RAW_DATA_DIR / "train.txt").
    """

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
    num_words=10000

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
    with open(tokenizer_path, 'wb') as f: 
        pickle.dump(tokenizer, f)
    logger.info(f"Tokenizer saved at: {tokenizer_path}")
    
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
    with open(train_sequences_path, 'wb') as f:
        pickle.dump(X_train, f)
    logger.info(f"Train sequences saved at: {train_sequences_path}")
    
    with open(val_sequences_path, 'wb') as f:
        pickle.dump(X_val, f)
    logger.info(f"Validation sequences saved at: {val_sequences_path}")
    
    with open(train_labels_path, 'wb') as f:
        pickle.dump(y_train, f)
    logger.info(f"Train labels saved at: {train_labels_path}")
    
    with open(val_labels_path, 'wb') as f:
        pickle.dump(y_val, f)
    logger.info(f"Validation labels saved at: {val_labels_path}")
    
if __name__ == "__main__":
    app()