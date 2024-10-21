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

from src.config import EXTERNAL_DATA_DIR, RESOURCES_DIR
from src import utilities

app = typer.Typer()

import dagshub
dagshub.init(repo_owner='Benji33', repo_name='TAED2_Amazon_Review_Classifiers', mlflow=True)

def load_glove_embeddings(path, word_index, embedding_dim, num_words=10000):
    """
    Loads only the required GloVe embeddings for the words in the word_index,
    and constructs an embedding matrix in an efficient way.

    Args:
        path (str): Path to the GloVe embeddings file.
        word_index (dict): A dictionary mapping words to their integer indices.
        embedding_dim (int): The dimensionality of the GloVe embeddings.
        num_words (int): Maximum number of words to include from the word_index.

    Returns:
        numpy.ndarray: An embedding matrix with GloVe embeddings for words in the word_index.
    """

    # Prepare the embedding matrix with zeros
    embedding_matrix = np.zeros((min(len(word_index) + 1, num_words), embedding_dim))

    # Open the GloVe file and load only relevant embeddings
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            # Split the line into the word and the corresponding coefficients
            word, *vector = line.split()
            if word in word_index:
                index = word_index[word]
                if index < num_words:  # Check if the word index is within our limit
                    embedding_matrix[index] = np.asarray(vector, dtype='float32')

            # Memory cleanup: forcefully delete the 'vector' after use
            del vector
    # Garbage collection after file processing
    gc.collect()
    return embedding_matrix

@app.command()
def main():
    """
    Main function to run the Amazon review sentiment classification training.
    """

    logger.info("Retrieving Params file.")
    params = utilities.get_params(root_dir)

    # Construct constants
    embeddings_path: Path = EXTERNAL_DATA_DIR / params["embeddings"]
    embedding_matrix_path: Path = EXTERNAL_DATA_DIR / params["embedding_matrix"]
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
    num_words=10000
    embedding_dim=100

    # ---- LOADING TOKENIZER ----
    logger.info("Loading tokenizer...")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    logger.info("Tokenizer loaded successfully.")

    # ---- LOADING GloVe PRE-TRAINED EMBEDDINGS AND CREATING EMBEDDING MATRIX ----
    logger.info("Loading GloVe pre-trained embeddings and creating embedding matrix...")
    embedding_matrix = load_glove_embeddings(embeddings_path, tokenizer.word_index, embedding_dim, num_words=num_words)
    logger.info("Embedding matrix created successfully.")

    # ---- SAVING EMBEDDING MATRIX ----
    logger.info("Saving embedding matrix...")
    with open(embedding_matrix_path, 'wb') as f:
        pickle.dump(embedding_matrix, f)
    logger.info(f"Embedding matrix saved at: {embedding_matrix_path}")


if __name__ == "__main__":
    app()