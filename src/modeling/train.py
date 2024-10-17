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

# Setting path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.config import MODELS_DIR, RAW_DATA_DIR, EXTERNAL_DATA_DIR

app = typer.Typer()
tracker = EmissionsTracker()

import dagshub
dagshub.init(repo_owner='Benji33', repo_name='TAED2_Amazon_Review_Classifiers', mlflow=True)

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
        train_sequences_path (Path): Path to the training sequences (default: RAW_DATA_DIR / "train_sequences.pkl").
        train_labels_path (Path): Path to the training labels (default: RAW_DATA_DIR / "train_labels.pkl").
        val_sequences_path (Path): Path to the validation sequences (default: RAW_DATA_DIR / "val_sequences.pkl").
        val_labels_path (Path): Path to the validation labels (default: RAW_DATA_DIR / "val_labels.pkl").
        embedding_matrix_path (Path): Path to the pre-trained embedding matrix (default: EXTERNAL_DATA_DIR / "embedding_matrix.pkl").
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
        
        # ---- SETTING HYPERPARAMETERS ----
        num_words=10000
        maxlen=250
        embedding_dim=100
        lstm_units=128
        dropout=0.5
        batch_size = 256
        num_epochs = 1
        
        mlflow.log_param("max_input_length", maxlen)
        mlflow.log_param("num_words", num_words)
        mlflow.log_param("embedding_dim", embedding_dim)
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lstm_units", lstm_units)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("num_epochs", num_epochs)

        # ---- LOADING EMBEDDING MATRIX ----
        logger.info("Loading the embedding matrix...")
        with open(embedding_matrix_path, 'rb') as f:
            embedding_matrix = pickle.load(f)
        logger.info("Embedding matrix loaded successfully.")
        
        # ---- BUILDING THE MODEL ----
        logger.info("Building the model...")
        
        model = Sequential([
            Embedding(num_words, embedding_dim, embeddings_initializer=Constant(embedding_matrix),
                    input_length=maxlen, trainable=False),
            Bidirectional(LSTM(lstm_units, return_sequences=True)),
            Dropout(dropout),
            Bidirectional(LSTM(lstm_units)),
            Dropout(dropout),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        logger.info("Compiling the model...")
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("loss", "binary_crossentropy")
        mlflow.log_param("metrics", "accuracy")
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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
        
        # ---- LABEL MAPPING ----
        label_mapping = {'__label__1': 0, '__label__2': 1}
        train_labels = [label_mapping[label] for label in train_labels]
        train_labels = np.array(train_labels).reshape(-1, 1)
        
        # ---- TRAINING ----
        logger.info("Training the model...")
        def data_generator(reviews, labels, batch_size, maxlen):
            total_samples = len(reviews)
            
            while True:
                for i in range(0, total_samples, batch_size):
                    batch_reviews = reviews[i:i+batch_size]
                    batch_labels = labels[i:i+batch_size]
                    
                    padded_sequences = pad_sequences(batch_reviews, padding='post', maxlen=maxlen)
                    
                    yield padded_sequences, batch_labels

        train_gen = data_generator(train_sequences, train_labels, batch_size, maxlen)

        model.fit(train_gen, steps_per_epoch=len(train_labels) // batch_size, epochs=num_epochs)

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

        # ---- SAVE THE MODEL ----
        logger.info("Saving the model...")
        model.save(MODELS_DIR / "sentiment_model.h5")

        tracker.stop()


if __name__ == "__main__":
    app()