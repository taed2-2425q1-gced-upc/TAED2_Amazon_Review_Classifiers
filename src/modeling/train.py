"""
Main script for training an Amazon review classifier using TensorFlow and MLflow.
This script handles the following tasks:
- TensorFlow version management
- Data loading and preprocessing
- Building and training a sentiment analysis model with LSTM layers
- Logging parameters, metrics, and artifacts to MLflow
- Tracking emissions using CodeCarbon
"""

from pathlib import Path
import sys
import gc
import typer
from loguru import logger
from tqdm import tqdm
import mlflow
from codecarbon import EmissionsTracker
import numpy as np
import tensorflow as tf
import pickle
import subprocess
import gc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Constant
from sklearn.model_selection import train_test_split

# setting path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.config import MODELS_DIR, RAW_DATA_DIR, EXTERNAL_DATA_DIR, RESOURCES_DIR

app = typer.Typer()
tracker = EmissionsTracker()

import dagshub
dagshub.init(repo_owner='Benji33', repo_name='TAED2_Amazon_Review_Classifiers', mlflow=True)

mlflow.set_experiment("amazon-reviews-test")

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
        train_data_path (Path): Path to the training data (default: RAW_DATA_DIR / "train.txt").
        model_path (Path): Path to save the trained model
        		   (default: MODELS_DIR / "sentiment_model.h5").
        embeddings_path (Path): Path to the pre-trained GloVe embeddings
        			(default: EXTERNAL_DATA_DIR / "glove.6B.100d.txt").
    """
    tracker.start()

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

    with mlflow.start_run():
        
        # ---- LOADING EMBEDDING MATRIX ----
        logger.info("Loading the embedding matrix...")
        with open(embedding_matrix_path, 'rb') as f:
            embedding_matrix = pickle.load(f)
        logger.info("Embedding matrix loaded successfully.")
        
        num_words=10000
        maxlen=250
        embedding_dim=100
        mlflow.log_param("max_input_length", maxlen)
        mlflow.log_param("num_words", num_words)
        mlflow.log_param("embedding_dim", embedding_dim)

        
        # ---- BUILDING THE MODEL ----
        logger.info("Building the model...")
        
        # Build the model
        model = Sequential([
            Embedding(10000, embedding_dim, embeddings_initializer=Constant(embedding_matrix),
                    input_length=250, trainable=False),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.5),
            Bidirectional(LSTM(128)),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        logger.info("Compiling the model...")
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("loss", "binary_crossentropy")
        mlflow.log_param("metrics", "accuracy")
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        
        # The embedding matrix is now saved, so you can delete the embedding_matrix if no longer needed
        del embedding_matrix
        gc.collect()
        
        # ---- DATA LOADING----
        logger.info("Loading training data...")
        with open(train_sequences_path, 'rb') as f:
            train_sequences = pickle.load(f)
        logger.info("Training sequences loaded successfully.")
        with open(train_labels_path, 'rb') as f:
            train_labels = pickle.load(f)
        logger.info("Training labels loaded successfully.")
        mlflow.log_param("train_size", len(train_labels))
        
        label_mapping = {'__label__1': 0, '__label__2': 1}
        train_labels = [label_mapping[label] for label in train_labels]
        # Reshape the labels to (num_samples, 1)
        train_labels = np.array(train_labels).reshape(-1, 1)
        
        # Training the model
        logger.info("Training the model...")
        mlflow.log_param("epochs", 5)
        mlflow.log_param("batch_size", 64)
        def data_generator(reviews, labels, batch_size, maxlen):
            total_samples = len(reviews)
            
            while True:
                for i in range(0, total_samples, batch_size):
                    batch_reviews = reviews[i:i+batch_size]
                    batch_labels = labels[i:i+batch_size]
                    
                    padded_sequences = pad_sequences(batch_reviews, padding='post', maxlen=maxlen)
                    
                    yield padded_sequences, batch_labels

        # Use the generator with Keras
        batch_size = 64
        maxlen = 250
        train_gen = data_generator(train_sequences, train_labels, batch_size, maxlen)

        # Use the generator with the Keras model
        model.fit(train_gen, steps_per_epoch=len(train_labels) // batch_size, epochs=5)

        # ---- CLEAN MEMORY ----
        del train_sequences
        del train_labels
        gc.collect()
        
        # ---- LOADING VALIDATION DATA ----
        logger.info("Loading validation data...")
        with open(val_sequences_path, 'rb') as f:
            val_sequences = pickle.load(f)
        logger.info("Validation sequences loaded successfully.")
        with open(val_labels_path, 'rb') as f:
            val_labels = pickle.load(f)
        logger.info("Validation labels loaded successfully.")
        mlflow.log_param("validation_size", len(val_labels))

        val_labels = [label_mapping[label] for label in val_labels]
        val_labels = np.array(val_labels).reshape(-1, 1)
        
        # ---- VALIDATION ----
        logger.info("Padding validation data...")
        padded_val_sequences = pad_sequences(val_sequences, padding='post', maxlen=maxlen)
        
        del val_sequences
        gc.collect()
        
        logger.info("Evaluating the training with the validation set...")
        loss, accuracy = model.evaluate(padded_val_sequences, val_labels)
        logger.info("Validation loss: {:.6f}".format(loss))
        logger.info("Validation accuracy: {:.6f}".format(accuracy))

        mlflow.log_metric("validation_loss", loss)
        mlflow.log_metric("validation_accuracy", accuracy)

        # ---- SAVE THE MODEL AND THE TOKENIZER ----
        logger.info("Saving the model and the tokenizer...")
        model.save(MODELS_DIR / "sentiment_model.h5")

        tracker.stop()


if __name__ == "__main__":
    app()