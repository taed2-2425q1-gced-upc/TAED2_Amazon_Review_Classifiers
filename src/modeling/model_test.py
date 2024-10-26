"""
This
"""

from pathlib import Path
import sys
import typing
import tensorflow as tf
import typer
from loguru import logger
import dagshub

# Setting path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.config import MODELS_DIR, RESOURCES_DIR
from src import utilities
from src.modeling.predict import predict_sentiment

app = typer.Typer()

# Initialize DagsHub integration with MLflow
dagshub.init(repo_owner='Benji33', repo_name='TAED2_Amazon_Review_Classifiers', mlflow=True)

def predict_sentiment(text: str, model: tf.keras.Model, tokenizer) -> typing.Tuple[str, float]:
    """
    Predict sentiment for a given text using a pre-trained model and tokenizer.

    Args:
        text (str): The input review text to predict the sentiment for.
        model (tf.keras.Model): The pre-trained TensorFlow model used for sentiment prediction.
        tokenizer: The tokenizer to convert text into sequences for the model.

    Returns:
        Tuple[str, float]: Tuple containing the predicted sentiment label ('Positive' or 'Negative')
        and the model's prediction probability (float).
    """
    # Preprocess the text before predicting (tokenizing and padding)
    sequence = tokenizer.texts_to_sequences([text])  # Convert text to sequence
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
    sequence, padding='post', maxlen=250)
    prediction = model.predict(padded_sequence)[0]
    sentiment_label = 'Negative' if prediction < 0.5 else 'Positive'
    return sentiment_label, prediction

# Function to check if sentiment prediction is conserved
def check_prediction_conservation(sentiment_1, sentiment_2):
    pred_1 = "positive" if sentiment_1 >= 0.5 else "negative"
    pred_2 = "positive" if sentiment_2 >= 0.5 else "negative"
    return pred_1 == pred_2


@app.command()
def main():
    """
    """

    # Check if TensorFlow is already version 2.10.0
    utilities.check_tensorflow_version()

    logger.info("Retrieving Params file.")
    params = utilities.get_params(root_dir)

    # Construct paths to the model, tokenizer, and prediction dataset
    model_path: Path = MODELS_DIR / params['model']
    tokenizer_path: Path = RESOURCES_DIR / params["tokenizer"]

    
    logger.info(f"Loading model and tokenizer.")
    # Load the trained model and tokenizer
    model = tf.keras.models.load_model(model_path)
    tokenizer = utilities.get_tokenizer(tokenizer_path)

    
    # Example reviews and their altered counterparts
    reviews = [
        ("The product arrived quickly and works perfectly.", "The product arrived quickly and dances perfectly."),
        ("Not worth the price, very disappointing quality.", "Not worth the price, very hilarious quality."),
        ("Great value for the money, highly recommend!", "Great value for the monkey, highly recommend!"),
        ("Easy to use and does the job well.", "Easy to hug and does the job well."),
        ("The packaging was damaged, but the item is fine.", "The packaging was damaged, but the tiger is fine.")
    ]

    logger.success("Testing model robustness with original and altered reviews.")
    for original_review, altered_review in reviews:
        # Get sentiment for original review
        sentiment_orig, prob_orig = predict_sentiment(original_review, model, tokenizer)
        prob_orig_value = float(prob_orig)  # Extract the float value from the NumPy array
        logger.info(f"Original Review: {original_review}\nSentiment: {sentiment_orig}, Probability: {prob_orig_value:.4f}")

        # Get sentiment for altered review
        sentiment_alt, prob_alt = predict_sentiment(altered_review, model, tokenizer)
        prob_alt_value = float(prob_alt)  # Extract the float value from the NumPy array
        logger.info(f"Altered Review: {altered_review}\nSentiment: {sentiment_alt}, Probability: {prob_alt_value:.4f}")

        # Check if the prediction is conserved
        is_conserved = check_prediction_conservation(prob_orig_value, prob_alt_value)
        conservation_status = "conserved" if is_conserved else "not conserved"
        logger.success(f"Prediction is {conservation_status}.\n")

    # Example reviews with negation to test model robustness
    negation_reviews = [
        ("The product works perfectly and arrived quickly.", "I wouldn't say the product works bad, and it didn't arrive late."),
        ("The subscription is expensive and not worth it.", "I wouldn't say the subscription is cheap, and it definitely wasn't worth it."),
        ("The product is affordable and does the job well.", "The product is not expensive, I can afford it and does a good job."),
        ("Great value for the money, highly recommend!", "Not a bad price for this product, highly recommended!"),
        ("Not worth the price, very disappointing quality.", "I thought it was worth the price. It wasn’t. The quality is disappointing.")
    ]

    logger.success("Testing model robustness against negation in single and multi-sentence reviews.")
    for original_review, negated_review in negation_reviews:
        # Get sentiment for original review
        sentiment_orig, prob_orig = predict_sentiment(original_review, model, tokenizer)
        prob_orig_value = float(prob_orig)  # Extract the float value from the NumPy array
        logger.info(f"Original Review: {original_review}\nSentiment: {sentiment_orig}, Probability: {prob_orig_value:.4f}")

        # Get sentiment for negated review
        sentiment_neg, prob_neg = predict_sentiment(negated_review, model, tokenizer)
        prob_neg_value = float(prob_neg)  # Extract the float value from the NumPy array
        logger.info(f"Negated Review: {negated_review}\nSentiment: {sentiment_neg}, Probability: {prob_neg_value:.4f}")

        # Check if the prediction changes significantly
        is_conserved = check_prediction_conservation(prob_orig_value, prob_neg_value)
        conservation_status = "conserved" if is_conserved else "not conserved"
        logger.success(f"Prediction is {conservation_status}.\n")


if __name__ == "__main__":
    app()
