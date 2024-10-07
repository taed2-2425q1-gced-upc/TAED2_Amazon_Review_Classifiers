from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import mlflow

from taed2_amazon_review_classifiers import MODELS_DIR, RAW_DATA_DIR, EXTERNAL_DATA_DIR, RESOURCES_DIR

app = typer.Typer()


import dagshub
dagshub.init(repo_owner='Benji33', repo_name='TAED2_Amazon_Review_Classifiers', mlflow=True)

mlflow.set_experiment("amazon-reviews-test")

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    train_data_path: Path = RAW_DATA_DIR / "train.txt",
    #labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "sentiment_model.h5",
    embeddings_path: Path = EXTERNAL_DATA_DIR / "glove.6B.100d.txt",
    # -----------------------------------------
):
    
    import numpy as np
    import tensorflow as tf
    import pickle
    import subprocess
    import sys 
    import gc 
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.initializers import Constant
    from sklearn.model_selection import train_test_split
    
    # ---- TENSORFLOW 2.10.0 INSTALLATION ---- 
    
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
        
        # ---- DATA LOADING ----
        logger.info("Loading data...")
        
        # Log the input file as an artifact
        mlflow.log_artifact(train_data_path)
        
        with open(train_data_path, 'r', encoding='utf-8') as train:
            lines = train.readlines()
        
        # ---- SIMPLE DATA PREPROCESSING ----
        logger.info("Proceeding with simple data preprocessing...")
        
        read_batch_size = 1000  # Adjust based on your memory limits
        labels = []
        reviews = []

        for i in range(0, len(lines), read_batch_size):
            batch = lines[i:i + read_batch_size]
            for line in batch:
                split_line = line.strip().split(' ', 1)
                label = 1 if split_line[0] == '__label__2' else 0
                review = split_line[1]
                labels.append(label)
                reviews.append(review)
                        # Clear batch variable to free memory
            del batch
            gc.collect()

        labels = np.array(labels) # Convert to numpy array    
        
        # Tokenizing and padding text
        num_words=10000
        mlflow.log_param("max_words_tokenizer", num_words)
        tokenizer = Tokenizer(num_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(reviews)
        sequences = tokenizer.texts_to_sequences(reviews)
        maxlen=250
        mlflow.log_param("max_input_length", maxlen)
        padded_sequences = pad_sequences(sequences, padding='post', maxlen=maxlen) # Max length of 250 for model input
        
        # ---- LOADING GloVe PRE-TRAINED EMBEDDINGS ----
        logger.info("Loading GloVe pre-trained embeddings...")
        
        def load_glove_embeddings(path, word_index, embedding_dim):
            embeddings_index = {}
            # Open the file with UTF-8 encoding specified
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    word, coefs = line.split(maxsplit=1)
                    coefs = np.fromstring(coefs, 'f', sep=' ')
                    embeddings_index[word] = coefs

            # Prepare embedding matrix
            num_words = min(len(word_index) + 1, num_words)
            embedding_matrix = np.zeros((num_words, embedding_dim))
            for word, i in word_index.items():
                if i >= num_words:
                    continue
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None: 
                    # Use the embedding vector for words found in the GloVe index
                    embedding_matrix[i] = embedding_vector  # words not found will be all-zeros.

            return embedding_matrix
        
        embedding_dim = 100  # Size of the GloVe vectors you're using
        mlflow.log_param("embedding_dim", embedding_dim)
        embedding_matrix = load_glove_embeddings(embeddings_path, tokenizer.word_index, embedding_dim)    
        
        # ---- BUILDING AND TRAINING THE MODEL ----
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
        
        # Split data into train and validation
        logger.info("Splitting data into train and validation...")
        mlflow.log_param("validation_size", 0.2)
        mlflow.log_param("random_seed", 42)
        X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("validation_size", len(X_val))
    
        # Training the model
        logger.info("Training the model...")
        mlflow.log_param("epochs", 5)
        mlflow.log_param("batch_size", 64)
        model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val), verbose=1)
        
        # Training evaluation
        logger.info("Evaluating the training with the validation set...")
        loss, accuracy = model.evaluate(X_val, y_val)
        logger.info("Validation loss: {:.6f}".format(loss))
        logger.info("Validation accuracy: {:.6f}".format(accuracy))
        
        mlflow.log_metric("validation_loss", loss)
        mlflow.log_metric("validation_accuracy", accuracy)
        
        # ---- SAVE THE MODEL AND THE TOKENIZER ----
        logger.info("Saving the model and the tokenizer...")
        model.save(MODELS_DIR / "sentiment_model.h5")
        with open(RESOURCES_DIR / "tokenizer.pkl", 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        # ---- REPLACE THIS WITH YOUR OWN CODE ----
        logger.info("Training some model...")
        for i in tqdm(range(10), total=10):
            if i == 5:
                logger.info("Something happened for iteration 5.")
        logger.success("Modeling training complete.")
        # -----------------------------------------


if __name__ == "__main__":
    app()