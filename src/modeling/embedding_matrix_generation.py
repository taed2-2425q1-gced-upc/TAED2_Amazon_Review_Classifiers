"""
This module handles loading GloVe embeddings and generating an embedding matrix for sentiment 
classification models. It checks TensorFlow version compatibility, loads a pre-trained tokenizer, 
and creates the embedding matrix using the GloVe embeddings, which is then saved for future use.

The module supports the following functionalities:
- Checking TensorFlow version and ensuring it matches the required version.
- Loading GloVe embeddings for specific words in the tokenizer's word index.
- Constructing and saving the embedding matrix for model training.
"""


import pickle
from pathlib import Path
import sys
import gc
import dagshub
import typer
from loguru import logger
import numpy as np

# setting path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))


from src.config import EXTERNAL_DATA_DIR, RESOURCES_DIR
from src import utilities

app = typer.Typer()

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

    utilities.check_tensorflow_version()

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
    embedding_matrix = load_glove_embeddings(embeddings_path, tokenizer.word_index,
    embedding_dim, num_words=num_words)
    logger.info("Embedding matrix created successfully.")

    # ---- SAVING EMBEDDING MATRIX ----
    logger.info("Saving embedding matrix...")
    with open(embedding_matrix_path, 'wb') as f:
        pickle.dump(embedding_matrix, f)
    logger.info(f"Embedding matrix saved at: {embedding_matrix_path}")

if __name__ == "__main__":
    app()
