"""
This module contains tests for the sentiment analysis model's robustness against negation in 
product reviews. It utilizes pytest to verify that the model maintains consistent predictions 
when the sentiment of reviews is negated.

Tests include:
- Assessing the model's response to various pairs of original and negated reviews, including:
  - Reviews that express a positive sentiment and their corresponding negated counterparts.
  - Reviews that express a negative sentiment and their corresponding negated counterparts.

The tests check if the sentiment prediction remains consistent (conserved) when the sentiment 
is reversed by negation, which is crucial for validating the model's understanding of context 
and negation.

Fixtures:
- `model_and_tokenizer`: A pytest fixture that loads the pre-trained sentiment analysis model 
  and the tokenizer from the specified directories, ensuring that the tests run in a controlled 
  environment.

To run the tests, execute the module directly or use pytest from the command line.

Usage:
    python -m pytest path_to_this_file.py
"""

from pathlib import Path
import sys
import tensorflow as tf
import pytest
import dagshub

# Setting path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.config import MODELS_DIR, RESOURCES_DIR
from src import utilities
from src.modeling.predict import predict_sentiment

# Initialize DagsHub integration with MLflow
dagshub.init(repo_owner='Benji33', repo_name='TAED2_Amazon_Review_Classifiers', mlflow=True)

def check_prediction_conservation(sentiment_1, sentiment_2):
    pred_1 = "positive" if sentiment_1 >= 0.5 else "negative"
    pred_2 = "positive" if sentiment_2 >= 0.5 else "negative"
    return pred_1 == pred_2

# Load the model and tokenizer
@pytest.fixture(scope="module")
def model_and_tokenizer():
    utilities.check_tensorflow_version()
    params = utilities.get_params(root_dir)

    model_path: Path = MODELS_DIR / params['model']
    tokenizer_path: Path = RESOURCES_DIR / params["tokenizer"]

    model = tf.keras.models.load_model(model_path)
    tokenizer = utilities.get_tokenizer(tokenizer_path)
    
    return model, tokenizer

# Test function for original vs. negated reviews
def test_model_robustness_original_vs_negated(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer

    negation_reviews = [
        ("The product works perfectly and arrived quickly.", "I wouldn't say the product works bad, and it didn't arrive late."),
        ("The subscription is expensive and not worth it.", "I wouldn't say the subscription is cheap, and it definitely wasn't worth it."),
        ("The product is affordable and does the job well.", "The product is not expensive, I can afford it and does a good job."),
        ("Great value for the money, highly recommend!", "Not a bad price for this product, highly recommended!"),
        ("Not worth the price, very disappointing quality.", "I thought it was worth the price. It wasn’t. The quality is disappointing.")
    ]

    results = []
    failures = []

    for original_review, negated_review in negation_reviews:
        sentiment_orig, prob_orig = predict_sentiment(original_review, model, tokenizer)
        prob_orig_value = float(prob_orig)

        sentiment_neg, prob_neg = predict_sentiment(negated_review, model, tokenizer)
        prob_neg_value = float(prob_neg)

        # Check if the prediction is conserved
        is_conserved = check_prediction_conservation(prob_orig_value, prob_neg_value)
        result = {
            "Original": original_review,
            "Negated": negated_review,
            "Original Probability": prob_orig_value,
            "Negated Probability": prob_neg_value,
            "Conservation": is_conserved
        }
        results.append(result)

        # Collect failures for reporting
        if not is_conserved:
            failures.append(result)

    # Print the results
    print("\nTest Results:")
    for res in results:
        status = "Conserved" if res["Conservation"] else "Not Conserved"
        print(f"Original: {res['Original']}\n"
              f"Negated: {res['Negated']}\n"
              f"Original Probability: {res['Original Probability']:.4f}\n"
              f"Negated Probability: {res['Negated Probability']:.4f}\n"
              f"Conservation: {status}\n")

    # Raise an assertion error if there are any failures
    if failures:
        raise AssertionError(f"{len(failures)} out of {len(results)} tests failed in the Negation robustness tests:\n" +
                             "\n".join([f"Original: {f['Original']} | Negated: {f['Negated']} | "
                                        f"Orig Prob: {f['Original Probability']:.4f}, "
                                        f"Negated Prob: {f['Negated Probability']:.4f}" for f in failures]))

if __name__ == "__main__":
    pytest.main()
